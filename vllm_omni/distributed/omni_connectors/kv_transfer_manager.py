# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified OmniConnector and KV cache transfer management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch
from vllm.logger import init_logger

from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec

if TYPE_CHECKING:
    from .connectors.base import OmniConnectorBase

logger = init_logger(__name__)


@dataclass
class OmniKVCacheConfig:
    """Configuration for OmniKVTransferManager."""

    connector_config: dict[str, Any] | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    stage_id: str | int | None = None
    engine_input_source: list[str | int] = field(default_factory=list)
    need_recv_cache: bool = False
    need_send_cache: bool = False
    recv_timeout: float = 30.0

    def is_valid(self) -> bool:
        return self.connector_config is not None and self.connector_config.get("type") is not None


@dataclass
class KVCacheTransferData:
    """KV cache data container for transfer between stages."""

    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "layer_blocks": self.layer_blocks,
            "block_ids": self.block_ids,
            "metadata": self.metadata,
        }


class RequestIdResolver(Protocol):
    """Protocol for resolving global request IDs."""

    def __call__(self, req_id: str) -> str: ...


class OmniKVTransferManager:
    """Unified management for OmniConnector and KV cache transfer."""

    def __init__(self, config: OmniKVCacheConfig):
        self.config = config
        self._connector: OmniConnectorBase | None = None

    def get_connector(self) -> OmniConnectorBase | None:
        """Get or create connector instance (lazy initialization)."""
        if self._connector is not None:
            return self._connector

        if not self.config.is_valid():
            return None

        try:
            cfg = self.config.connector_config
            assert cfg is not None
            c_type = cfg.get("type")
            if not c_type:
                return None

            c_extra = {k: v for k, v in cfg.items() if k != "type"}
            self._connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=c_type, extra=c_extra))
            logger.info(f"OmniKVTransferManager: Created connector {c_type}")
            return self._connector
        except Exception as e:
            logger.error(f"Failed to create OmniConnector: {e}")
            return None

    def has_connector(self) -> bool:
        return self.get_connector() is not None

    def extract_kv_cache(
        self,
        req_id: str,
        block_ids: list[int],
        seq_len: int,
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
    ) -> KVCacheTransferData | None:
        """Extract KV cache from GPU blocks for a single request."""
        num_layers = len(kv_caches)
        key_cache: list[torch.Tensor | None] = [None] * num_layers
        value_cache: list[torch.Tensor | None] = [None] * num_layers

        for layer_idx, kv_tensor in enumerate(kv_caches):
            max_block = kv_tensor.shape[1] - 1
            valid_ids = [bid for bid in block_ids if 0 <= bid <= max_block]
            if not valid_ids:
                continue

            selected = kv_tensor[:, valid_ids]
            n_kv, n_blks, blk_sz, n_heads, d_head = selected.shape
            flat = selected.reshape(n_kv, n_blks * blk_sz, n_heads, d_head)
            if seq_len < flat.shape[1]:
                flat = flat[:, :seq_len]

            flat_cpu = flat.detach().cpu().contiguous()
            key_cache[layer_idx] = flat_cpu[0]
            value_cache[layer_idx] = flat_cpu[1]

        if not any(k is not None for k in key_cache):
            return None

        return KVCacheTransferData(
            request_id=req_id,
            layer_blocks={"key_cache": key_cache, "value_cache": value_cache},
            block_ids=block_ids,
            metadata={
                "block_size": block_size,
                "num_layers": num_layers,
                "dtype": str(cache_dtype),
                "seq_len": seq_len,
            },
        )

    def handle_finished_requests_kv_transfer(
        self,
        finished_reqs: dict[str, dict[str, Any]],
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
        request_id_resolver: RequestIdResolver | None = None,
    ) -> list[str]:
        """Handle KV cache transfer for finished requests."""
        if not finished_reqs or not self.has_connector():
            return []

        extracted_ids = []
        for req_id, data in finished_reqs.items():
            try:
                seq_len = data.get("seq_len", 0)
                block_ids = data.get("block_ids", [])
                if not block_ids:
                    continue

                kv_data = self.extract_kv_cache(req_id, block_ids, seq_len, kv_caches, block_size, cache_dtype)
                if kv_data:
                    transfer_req_id = request_id_resolver(req_id) if request_id_resolver else req_id
                    self.transfer_kv_cache(kv_data, transfer_req_id)
            except Exception as e:
                logger.error(f"Failed KV transfer for {req_id}: {e}")
            finally:
                extracted_ids.append(req_id)

        return extracted_ids

    def transfer_kv_cache(self, kv_data: KVCacheTransferData, transfer_req_id: str | None = None) -> bool:
        """Transfer KV cache data to downstream stage."""
        if transfer_req_id is None:
            transfer_req_id = kv_data.request_id
        data_dict = kv_data.to_dict()
        data_dict["request_id"] = transfer_req_id
        return self.send_kv_cache(f"kv_cache_{transfer_req_id}", data_dict)

    def send_kv_cache(self, request_id: str, data: dict[str, Any], max_retries: int = 3) -> bool:
        """Send KV cache to downstream stage with retry."""
        connector = self.get_connector()
        if not connector:
            return False

        from_stage, to_stage = self._get_transfer_stages()
        if from_stage is None or to_stage is None:
            return False

        success, size, _ = self._transfer_with_retry(connector, from_stage, to_stage, request_id, data, max_retries)
        if success:
            logger.info(f"KV cache send OK: {request_id}, {size} bytes")
        else:
            logger.error(f"KV cache send FAILED: {request_id}")
        return success

    def receive_kv_cache(self, request_id: str, timeout: float | None = None) -> tuple[dict[str, Any], int] | None:
        """Receive KV cache from upstream stage with timeout."""
        if not self.config.need_recv_cache:
            return None

        connector = self.get_connector()
        if not connector:
            return None

        from_stage, to_stage = self._get_receive_stages()
        if from_stage is None or to_stage is None:
            return None

        timeout = timeout if timeout is not None else self.config.recv_timeout
        start_time = time.time()
        logger.info(f"Waiting for KV cache for {request_id}...")

        while True:
            try:
                result = connector.get(
                    from_stage=from_stage,
                    to_stage=to_stage,
                    request_id=f"kv_cache_{request_id}",
                )
                if result:
                    data, size = result
                    logger.info(f"Received KV cache for {request_id}, {size} bytes")
                    return data, size
            except Exception as e:
                logger.warning(f"Error receiving KV cache: {e}")

            if time.time() - start_time > timeout:
                logger.error(f"Timeout waiting for KV cache for {request_id}")
                return None
            time.sleep(0.5)

    def _get_transfer_stages(self) -> tuple[str | None, str | None]:
        if self.config.from_stage and self.config.to_stage:
            return str(self.config.from_stage), str(self.config.to_stage)
        return None, None

    def _get_receive_stages(self) -> tuple[str | None, str | None]:
        to_stage = self.config.stage_id
        if self.config.engine_input_source:
            from_stage = self.config.engine_input_source[0]
        elif isinstance(to_stage, int):
            from_stage = to_stage - 1
        else:
            from_stage = self.config.from_stage

        if from_stage is not None and to_stage is not None:
            return str(from_stage), str(to_stage)
        return None, None

    def _transfer_with_retry(
        self,
        connector: OmniConnectorBase,
        from_stage: str,
        to_stage: str,
        request_id: str,
        data: dict[str, Any],
        max_retries: int = 3,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        for attempt in range(max_retries):
            try:
                success, size, metadata = connector.put(
                    from_stage=from_stage, to_stage=to_stage, request_id=request_id, data=data
                )
                if success:
                    return success, size, metadata
            except Exception as e:
                logger.warning(f"Transfer attempt {attempt + 1} exception: {e}")

            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))
        return False, 0, None

    @staticmethod
    def from_omni_kv_config(omni_kv_config: dict[str, Any] | None) -> OmniKVTransferManager:
        if not omni_kv_config:
            return OmniKVTransferManager(OmniKVCacheConfig())

        return OmniKVTransferManager(
            OmniKVCacheConfig(
                connector_config=omni_kv_config.get("connector_config"),
                from_stage=omni_kv_config.get("omni_from_stage"),
                to_stage=omni_kv_config.get("omni_to_stage"),
                stage_id=omni_kv_config.get("stage_id"),
                engine_input_source=omni_kv_config.get("engine_input_source", []),
                need_recv_cache=omni_kv_config.get("need_recv_cache", False),
                need_send_cache=omni_kv_config.get("need_send_cache", False),
                recv_timeout=omni_kv_config.get("recv_timeout", 30.0),
            )
        )

    @staticmethod
    def from_model_config(model_config: Any) -> OmniKVTransferManager:
        omni_kv = getattr(model_config, "omni_kv_config", None)
        if isinstance(omni_kv, dict):
            return OmniKVTransferManager.from_omni_kv_config(omni_kv)
        return OmniKVTransferManager(OmniKVCacheConfig())

    @staticmethod
    def from_od_config(od_config: Any) -> OmniKVTransferManager:
        omni_kv = getattr(od_config, "omni_kv_config", None)
        if isinstance(omni_kv, dict):
            return OmniKVTransferManager.from_omni_kv_config(omni_kv)
        return OmniKVTransferManager(OmniKVCacheConfig())
