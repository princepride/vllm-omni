import unittest

import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
)


class MockConnector:
    def __init__(self):
        self.store = {}

    def put(self, from_stage, to_stage, request_id, data):
        # The manager now passes full key as request_id (e.g., omni_stage1_to_stage2_kv_cache_req_test_1)
        key = f"{from_stage}->{to_stage}:{request_id}"
        self.store[key] = data
        return True, len(str(data)), None  # (success, size, metadata)

    def get(self, from_stage, to_stage, request_id, metadata=None):
        # The manager now passes full key as request_id
        key = f"{from_stage}->{to_stage}:{request_id}"
        if key in self.store:
            return self.store[key], len(str(self.store[key]))
        return None


class TestKVFlow(unittest.TestCase):
    def setUp(self):
        # Common constants
        self.num_layers = 2
        self.num_heads = 4
        self.head_dim = 16
        self.block_size = 8
        self.seq_len = 20
        self.req_id = "req_test_1"

        # Test config
        self.config = OmniKVCacheConfig(
            connector_config={"type": "mock"},
            from_stage="stage1",
            to_stage="stage2",
            stage_id="stage2",  # Acting as receiver for some tests
            need_recv_cache=True,
            need_send_cache=True,
            recv_timeout=1.0,  # Short timeout for tests
        )

    def test_manager_extraction(self):
        """Test extraction and sending logic in OmniKVTransferManager."""

        num_blocks = 10
        kv_caches = []
        for _ in range(self.num_layers):
            k_cache = torch.randn(num_blocks, self.block_size, self.num_heads, self.head_dim)
            v_cache = torch.randn(num_blocks, self.block_size, self.num_heads, self.head_dim)
            # Stack K and V to create [2, num_blocks, block_size, n_heads, head_dim]
            layer_cache = torch.stack([k_cache, v_cache], dim=0)
            kv_caches.append(layer_cache)

        block_ids = [1, 3, 5]
        finished_reqs = {self.req_id: {"block_ids": block_ids, "seq_len": self.seq_len}}

        manager = OmniKVTransferManager(self.config)
        # Mock the connector factory or injection
        mock_connector = MockConnector()
        manager._connector = mock_connector

        processed = manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, self.block_size, "float32")

        self.assertIn(self.req_id, processed)

        # Check if data was put into connector
        # Manager builds full key: omni_{from}_to_{to}_kv_cache_{req_id}
        full_request_id = f"omni_stage1_to_stage2_kv_cache_{self.req_id}"
        expected_key = f"stage1->stage2:{full_request_id}"
        self.assertIn(expected_key, mock_connector.store)

        data = mock_connector.store[expected_key]
        self.assertEqual(data["request_id"], self.req_id)
        self.assertIn("layer_blocks", data)
        self.assertEqual(len(data["layer_blocks"]["key_cache"]), self.num_layers)

        # Verify shape of extracted tensor: [seq_len, heads, dim]
        # Note: Manager detaches and moves to CPU
        expected_shape = (self.seq_len, self.num_heads, self.head_dim)
        self.assertEqual(data["layer_blocks"]["key_cache"][0].shape, expected_shape)

        return data  # Return for potential use

    def test_manager_reception(self):
        """Test reception and injection logic in OmniKVTransferManager."""

        expected_shape = (self.seq_len, self.num_heads, self.head_dim)
        key_cache = [torch.randn(expected_shape) for _ in range(self.num_layers)]
        value_cache = [torch.randn(expected_shape) for _ in range(self.num_layers)]

        layer_blocks = {"key_cache": key_cache, "value_cache": value_cache}
        metadata = {
            "block_size": self.block_size,
            "num_layers": self.num_layers,
            "dtype": "float32",
            "seq_len": self.seq_len,
        }

        data_to_receive = {
            "request_id": self.req_id,
            "layer_blocks": layer_blocks,
            "metadata": metadata,
            "block_ids": [],
        }

        # In setUp, from_stage="stage1", stage_id="stage2". recv_stages=("stage1", "stage2")

        manager = OmniKVTransferManager(self.config)
        mock_connector = MockConnector()
        manager._connector = mock_connector

        # Pre-populate connector with data
        # Manager builds full key: omni_{from}_to_{to}_kv_cache_{req_id}
        full_request_id = f"omni_stage1_to_stage2_kv_cache_{self.req_id}"
        store_key = f"stage1->stage2:{full_request_id}"
        mock_connector.store[store_key] = data_to_receive

        req = OmniDiffusionRequest(prompt="test_recv", request_id=self.req_id)
        success = manager.receive_kv_cache(req, target_device=torch.device("cpu"))

        self.assertTrue(success)
        self.assertTrue(hasattr(req, "past_key_values"))
        self.assertTrue(hasattr(req, "kv_metadata"))

        self.assertEqual(len(req.past_key_values.key_cache), self.num_layers)
        self.assertTrue(torch.allclose(req.past_key_values.key_cache[0], key_cache[0]))
        self.assertEqual(req.kv_metadata["seq_len"], self.seq_len)

    def test_integration_flow(self):
        """Simulate extraction -> connector -> reception."""

        sender_config = OmniKVCacheConfig(
            connector_config={"type": "mock"}, from_stage="sender", to_stage="receiver", need_send_cache=True
        )
        sender_manager = OmniKVTransferManager(sender_config)
        connector = MockConnector()
        sender_manager._connector = connector  # Shared connector

        # Create Data
        num_blocks = 5
        kv_caches = []
        for _ in range(self.num_layers):
            layer = torch.randn(2, num_blocks, self.block_size, self.num_heads, self.head_dim)
            kv_caches.append(layer)

        finished_reqs = {self.req_id: {"block_ids": [0, 1], "seq_len": 10}}

        # Send
        sender_manager.handle_finished_requests_kv_transfer(finished_reqs, kv_caches, self.block_size, "float32")

        receiver_config = OmniKVCacheConfig(
            connector_config={"type": "mock"},
            from_stage="sender",
            stage_id="receiver",
            need_recv_cache=True,
            recv_timeout=1.0,
        )
        receiver_manager = OmniKVTransferManager(receiver_config)
        receiver_manager._connector = connector  # Share the same mock connector instance

        req = OmniDiffusionRequest(prompt="test_integ", request_id=self.req_id)

        # Receive
        success = receiver_manager.receive_kv_cache(req)

        # Verify
        self.assertTrue(success)
        self.assertIsNotNone(req.past_key_values)
        self.assertEqual(req.kv_metadata["seq_len"], 10)


if __name__ == "__main__":
    unittest.main()
