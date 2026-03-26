# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MagiHuman DiT model ported into vLLM-Omni."""

import pytest
import torch


class TestDiTModelForward:
    """Unit tests: verify DiT model forward pass with random data (no weights needed)."""

    def test_dit_config_defaults(self):
        from vllm_omni.diffusion.models.magi_human.magi_human_dit import MagiHumanDiTConfig

        config = MagiHumanDiTConfig()
        assert config.num_layers == 40
        assert config.hidden_size == 5120
        assert config.num_heads_q == 40  # 5120 // 128
        assert config.num_heads_kv == 8
        assert config.video_in_channels == 192  # 48 * 4
        assert config.audio_in_channels == 64
        assert config.text_in_channels == 3584

    def test_dit_model_instantiation(self):
        from vllm_omni.diffusion.models.magi_human.magi_human_dit import DiTModel, MagiHumanDiTConfig

        # Use small config for fast testing
        config = MagiHumanDiTConfig(
            num_layers=2,
            hidden_size=256,
            head_dim=64,
            num_query_groups=2,
            video_in_channels=16,
            audio_in_channels=8,
            text_in_channels=32,
            mm_layers=[0, 1],
            gelu7_layers=[0],
        )
        model = DiTModel(model_config=config)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert hasattr(model, "adapter")
        assert hasattr(model, "block")
        assert hasattr(model, "final_norm_video")
        assert hasattr(model, "final_linear_video")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dit_forward_pass(self):
        """Test forward pass with a small model and random data."""
        from vllm_omni.diffusion.models.magi_human.magi_human_dit import (
            DiTModel,
            MagiHumanDiTConfig,
            Modality,
            VarlenHandler,
        )

        config = MagiHumanDiTConfig(
            num_layers=2,
            hidden_size=256,
            head_dim=64,
            num_query_groups=2,
            video_in_channels=16,
            audio_in_channels=8,
            text_in_channels=32,
            mm_layers=[0, 1],
            gelu7_layers=[0],
            enable_attn_gating=False,
        )
        model = DiTModel(model_config=config).cuda().eval()

        # Simulate: 4 video tokens + 3 audio tokens + 2 text tokens = 9 total
        n_video, n_audio, n_text = 4, 3, 2
        seq_len = n_video + n_audio + n_text
        max_ch = max(config.video_in_channels, config.audio_in_channels, config.text_in_channels)

        x = torch.randn(seq_len, max_ch, device="cuda")
        coords_mapping = torch.randn(seq_len, 9, device="cuda")
        modality_mapping = torch.cat(
            [
                torch.full((n_video,), Modality.VIDEO, dtype=torch.int64, device="cuda"),
                torch.full((n_audio,), Modality.AUDIO, dtype=torch.int64, device="cuda"),
                torch.full((n_text,), Modality.TEXT, dtype=torch.int64, device="cuda"),
            ]
        )
        varlen_handler = VarlenHandler(
            cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device="cuda"),
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
        )

        with torch.no_grad():
            out = model(x, coords_mapping, modality_mapping, varlen_handler, None)

        assert out.shape == (seq_len, max(config.video_in_channels, config.audio_in_channels))


class TestDataProxy:
    """Test data preprocessing pipeline."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_data_proxy_roundtrip(self):
        from vllm_omni.diffusion.models.magi_human.magi_human_data_proxy import MagiDataProxy
        from vllm_omni.diffusion.models.magi_human.pipeline_magi_human import EvalInput

        proxy = MagiDataProxy(
            patch_size=2,
            t_patch_size=1,
            frame_receptive_field=-1,  # disable local attn for simplicity
        )

        # Simulate a small video latent: (B=1, C=16, T=2, H=4, W=4)
        z_dim = 16
        x_t = torch.randn(1, z_dim, 2, 4, 4, device="cuda")
        audio_x_t = torch.randn(1, 5, 8, device="cuda")
        txt_feat = torch.randn(1, 3, 32, device="cuda")

        eval_input = EvalInput(
            x_t=x_t,
            audio_x_t=audio_x_t,
            audio_feat_len=[5],
            txt_feat=txt_feat,
            txt_feat_len=[3],
        )

        # Process input
        result = proxy.process_input(eval_input)
        x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler = result

        # Verify shapes
        # video tokens: T/t_patch * H/patch * W/patch = 2*2*2 = 8
        # total = 8 + 5 + 3 = 16
        assert x.shape[0] == 16
        assert coords_mapping.shape[0] == 16
        assert modality_mapping.shape[0] == 16
        assert local_attn_handler is None  # disabled

        # Simulate model output
        max_ch = max(z_dim * 1 * 2 * 2, 8)  # video_ch = z_dim * pT * pH * pW
        fake_output = torch.randn(16, max_ch, device="cuda")
        video_out, audio_out = proxy.process_output(fake_output)

        assert video_out.shape[0] == 1  # batch
        assert audio_out.shape == (1, 5, 8)


class TestScheduler:
    """Test scheduler."""

    def test_scheduler_set_timesteps(self):
        from vllm_omni.diffusion.models.magi_human.magi_human_scheduler import FlowUniPCMultistepScheduler

        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(32, device="cpu", shift=5.0)
        assert len(scheduler.timesteps) == 32
        # Timesteps should be decreasing
        assert scheduler.timesteps[0] > scheduler.timesteps[-1]

    def test_scheduler_step(self):
        from vllm_omni.diffusion.models.magi_human.magi_human_scheduler import FlowUniPCMultistepScheduler

        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(4, device="cpu", shift=5.0)

        sample = torch.randn(1, 4, 2, 4, 4)
        model_output = torch.randn_like(sample)

        result = scheduler.step(model_output, scheduler.timesteps[0], sample, return_dict=False)
        assert result[0].shape == sample.shape

    def test_scheduler_step_ddim(self):
        from vllm_omni.diffusion.models.magi_human.magi_human_scheduler import FlowUniPCMultistepScheduler

        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(4, device="cpu", shift=5.0)

        sample = torch.randn(1, 4, 2, 4, 4)
        velocity = torch.randn_like(sample)

        result = scheduler.step_ddim(velocity, 0, sample)
        assert result.shape == sample.shape


class TestRegistry:
    """Test model registration."""

    def test_registry_lookup(self):
        from vllm_omni.diffusion.registry import DiffusionModelRegistry

        cls = DiffusionModelRegistry._try_load_model_cls("MagiHumanPipeline")
        assert cls is not None
        assert cls.__name__ == "MagiHumanPipeline"

    def test_post_process_func_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        assert "MagiHumanPipeline" in _DIFFUSION_POST_PROCESS_FUNCS

    def test_pre_process_func_registered(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_PRE_PROCESS_FUNCS

        assert "MagiHumanPipeline" in _DIFFUSION_PRE_PROCESS_FUNCS
