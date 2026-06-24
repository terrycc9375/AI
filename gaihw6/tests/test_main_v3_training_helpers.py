import unittest
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from main_v3 import (
    CrossAttention,
    apply_cfg_dropout,
    compute_balanced_snr_weights,
    compute_min_snr_weights,
    create_ema_model,
    generate_save_and_evaluate_fid,
    save_loss_curve,
    to_image_range,
    update_ema,
)


class TrainingHelperTests(unittest.TestCase):
    def test_loss_curve_uses_one_based_epochs_and_saves_file(self):
        pyplot = MagicMock()
        with patch.dict("sys.modules", {"matplotlib.pyplot": pyplot}):
            save_loss_curve([2.0, 1.5], "curve.png")

        pyplot.plot.assert_called_once_with(
            [1, 2], [2.0, 1.5], marker="o"
        )
        pyplot.savefig.assert_called_once_with("curve.png")
        pyplot.close.assert_called_once()

    def test_cfg_dropout_selects_empty_context_per_sample(self):
        context = torch.ones(3, 2, 4)
        empty_context = torch.zeros(1, 2, 4)
        dropout_mask = torch.tensor([True, False, True]).view(3, 1, 1)

        output = apply_cfg_dropout(context, empty_context, dropout_mask)

        torch.testing.assert_close(output[0], empty_context[0])
        torch.testing.assert_close(output[1], context[1])
        torch.testing.assert_close(output[2], empty_context[0])

    def test_min_snr_weights_cap_high_snr_samples(self):
        scheduler = SimpleNamespace(
            alphas_cumprod=torch.tensor([0.5, 0.9, 0.1])
        )
        timesteps = torch.tensor([0, 1, 2])

        weights = compute_min_snr_weights(scheduler, timesteps, gamma=5.0)

        expected = torch.tensor([1.0, 5.0 / 9.0, 1.0])
        torch.testing.assert_close(weights, expected)

    def test_balanced_snr_weights_retain_half_of_unweighted_loss(self):
        scheduler = SimpleNamespace(
            alphas_cumprod=torch.tensor([0.5, 0.9, 0.1])
        )
        timesteps = torch.tensor([0, 1, 2])

        weights = compute_balanced_snr_weights(
            scheduler, timesteps, gamma=5.0, min_unweighted_fraction=0.5
        )

        expected = torch.tensor([1.0, 7.0 / 9.0, 1.0])
        torch.testing.assert_close(weights, expected)

    def test_image_range_clamps_values_before_normalization(self):
        samples = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        output = to_image_range(samples)

        torch.testing.assert_close(
            output, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        )

    def test_generation_defaults_use_balanced_guidance_and_more_steps(self):
        signature = inspect.signature(generate_save_and_evaluate_fid)

        self.assertEqual(signature.parameters["cfg_scale"].default, 2.5)
        self.assertEqual(signature.parameters["ddim_steps"].default, 100)

    def test_cross_attention_preserves_residual_when_projection_is_zero(self):
        attention = CrossAttention(query_dim=8, context_dim=6, heads=2)
        nn.init.zeros_(attention.to_out.weight)
        nn.init.zeros_(attention.to_out.bias)
        image_features = torch.randn(2, 8, 4, 4)
        context = torch.randn(2, 5, 6)

        output = attention(image_features, context)

        torch.testing.assert_close(output, image_features)

    def test_ema_model_is_frozen_and_updates_toward_training_model(self):
        model = nn.Linear(2, 1, bias=False)
        nn.init.constant_(model.weight, 2.0)
        ema_model = create_ema_model(model)
        nn.init.zeros_(ema_model.weight)

        update_ema(ema_model, model, decay=0.75)

        self.assertFalse(ema_model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in ema_model.parameters())
        )
        torch.testing.assert_close(
            ema_model.weight, torch.full_like(ema_model.weight, 0.5)
        )

    def test_ema_default_decay_tracks_a_shorter_training_run(self):
        model = nn.Linear(1, 1, bias=False)
        ema_model = create_ema_model(model)
        nn.init.ones_(model.weight)
        nn.init.zeros_(ema_model.weight)

        update_ema(ema_model, model)

        torch.testing.assert_close(
            ema_model.weight, torch.full_like(ema_model.weight, 0.001)
        )


if __name__ == "__main__":
    unittest.main()
