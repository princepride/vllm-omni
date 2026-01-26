# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified OmniConnector and KV cache transfer management."""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec

logger = init_logger(__name__)


@dataclass
class OmniKVCacheConfig:
    """Configuration for OmniKVTransferManager."""

    connector_config: dict[str, Any] | None = None
    from_stage: str | None = None
    to_stage: str | None = None
    stage_id: str | int | None = None
    engine_input_source: list[str | int] | None = None
    need_recv_cache: bool = False
    need_send_cache: bool = False
    recv_timeout: float = 30.0


@dataclass
class KVCacheTransferData:
    """Container for KV cache transfer data."""

    request_id: str
    layer_blocks: dict[str, Any]
    block_ids: list[int]
    metadata: dict[str, Any]


class OmniKVTransferManager:
    """Unified management for OmniConnector and KV cache transfer."""

    def __init__(self, config: OmniKVCacheConfig):
        self.config = config
        self._connector = None

        # Pre-calculate transfer stages (src, dst)
        self.send_stages = (
            (str(config.from_stage), str(config.to_stage)) if config.from_stage and config.to_stage else (None, None)
        )

        # Pre-calculate receive stages (src, dst)
        recv_from = config.from_stage
        if config.engine_input_source:
            recv_from = config.engine_input_source[0]
        elif isinstance(config.stage_id, int):
            recv_from = config.stage_id - 1

        self.recv_stages = (
            (str(recv_from), str(config.stage_id))
            if recv_from is not None and config.stage_id is not None
            else (None, None)
        )

    @classmethod
    def _create(cls, cfg: dict | None) -> "OmniKVTransferManager":
        if not cfg:
            return cls(OmniKVCacheConfig())
        return cls(
            OmniKVCacheConfig(
                connector_config=cfg.get("connector_config"),
                from_stage=cfg.get("omni_from_stage"),
                to_stage=cfg.get("omni_to_stage"),
                stage_id=cfg.get("stage_id"),
                engine_input_source=cfg.get("engine_input_source", []),
                need_recv_cache=cfg.get("need_recv_cache", False),
                need_send_cache=cfg.get("need_send_cache", False),
                recv_timeout=cfg.get("recv_timeout", 30.0),
            )
        )

    @classmethod
    def from_omni_kv_config(cls, cfg: dict | None) -> "OmniKVTransferManager":
        return cls._create(cfg)

    @classmethod
    def from_model_config(cls, config: Any) -> "OmniKVTransferManager":
        return cls._create(getattr(config, "omni_kv_config", None))

    @classmethod
    def from_od_config(cls, config: Any) -> "OmniKVTransferManager":
        return cls._create(getattr(config, "omni_kv_config", None))

    @property
    def connector(self):
        if self._connector is None and (cfg := self.config.connector_config) and (c_type := cfg.get("type")):
            extra = {k: v for k, v in cfg.items() if k != "type"}
            self._connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=c_type, extra=extra))
        return self._connector

    def get_connector(self):
        """Compatibility wrapper."""
        return self.connector

    def handle_finished_requests_kv_transfer(
        self,
        finished_reqs: dict[str, dict[str, Any]],
        kv_caches: list[torch.Tensor],
        block_size: int,
        cache_dtype: str,
        request_id_resolver: Callable[[str], str] | None = None,
    ) -> list[str]:
        """Extracts and sends KV cache for finished requests."""
        if not finished_reqs or not self.connector:
            return []

        processed = []
        for req_id, data in finished_reqs.items():
            try:
                if not (block_ids := data.get("block_ids")):
                    continue

                if payload := self._extract(
                    req_id, block_ids, data.get("seq_len", 0), kv_caches, block_size, cache_dtype
                ):
                    target_id = request_id_resolver(req_id) if request_id_resolver else req_id

                    data_dict = asdict(payload)
                    data_dict["request_id"] = target_id

                    self._send(f"kv_cache_{target_id}", data_dict)
            except Exception as e:
                logger.error(f"Failed KV transfer for {req_id}: {e}")
            finally:
                processed.append(req_id)
        return processed

    def _extract(self, req_id, block_ids, seq_len, kv_caches, block_size, cache_dtype) -> KVCacheTransferData | None:
        keys, values = [], []
        for kv in kv_caches:
            valid = [b for b in block_ids if b < kv.shape[1]]
            if not valid:
                continue

            # Extract: [n_kv, n_valid, blk_sz, head, dim] -> reshape -> slice -> cpu
            flat = kv[:, valid].reshape(kv.shape[0], -1, *kv.shape[3:])
            flat = flat[:, :seq_len].detach().cpu().contiguous()
            keys.append(flat[0])
            values.append(flat[1])

        if not keys:
            return None
        return KVCacheTransferData(
            request_id=req_id,
            layer_blocks={"key_cache": keys, "value_cache": values},
            block_ids=block_ids,
            metadata={
                "block_size": block_size,
                "num_layers": len(kv_caches),
                "dtype": str(cache_dtype),
                "seq_len": seq_len,
            },
        )

    def _send(self, req_id: str, data: dict, retries: int = 3) -> bool:
        src, dst = self.send_stages
        if not src or not dst:
            return False

        for i in range(retries):
            try:
                if self.connector.put(from_stage=src, to_stage=dst, request_id=req_id, data=data)[0]:
                    logger.info(f"KV cache send OK: {req_id}")
                    return True
            except Exception:
                pass
            if i < retries - 1:
                time.sleep(0.1 * (2**i))

        logger.error(f"KV cache send FAILED: {req_id}")
        return False

    def receive_kv_cache(self, request_id: str, timeout: float | None = None) -> tuple[dict[str, Any], int] | None:
        if not self.config.need_recv_cache or not self.connector:
            return None

        src, dst = self.recv_stages
        if not src or not dst:
            return None

        key = f"kv_cache_{request_id}"
        timeout = timeout or self.config.recv_timeout
        start = time.time()

        logger.info(f"Waiting for KV cache {request_id}...")
        while time.time() - start < timeout:
            try:
                if res := self.connector.get(from_stage=src, to_stage=dst, request_id=key):
                    logger.info(f"Received KV cache for {request_id}, {res[1]} bytes")
                    return res
            except Exception:
                pass
            time.sleep(0.5)

        logger.error(f"Timeout waiting for KV cache for {request_id}")
        return None
