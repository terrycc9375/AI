# Residual Attention and EMA Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve image features through residual cross-attention and use a frozen EMA UNet for evaluation and image generation.

**Architecture:** `CrossAttention` will pre-normalize image features used for queries and add its projected output back to the original image tensor. Two small helpers will create and update a separate EMA model; the training loop updates it after each optimizer step, while the post-training generation function receives that EMA model directly.

**Tech Stack:** Python, PyTorch, pytest, Hugging Face Transformers and Diffusers

---

### Task 1: Add behavioral regression tests

**Files:**
- Create: `tests/test_main_v3_training_helpers.py`
- Test: `tests/test_main_v3_training_helpers.py`

- [ ] **Step 1: Write focused failing tests**

```python
import torch
from torch import nn

from main_v3 import CrossAttention, create_ema_model, update_ema


def test_cross_attention_preserves_residual_when_projection_is_zero():
    attention = CrossAttention(query_dim=8, context_dim=6, heads=2)
    nn.init.zeros_(attention.to_out.weight)
    nn.init.zeros_(attention.to_out.bias)
    image_features = torch.randn(2, 8, 4, 4)
    context = torch.randn(2, 5, 6)

    output = attention(image_features, context)

    torch.testing.assert_close(output, image_features)


def test_ema_model_is_frozen_and_updates_toward_training_model():
    model = nn.Linear(2, 1, bias=False)
    nn.init.constant_(model.weight, 2.0)
    ema_model = create_ema_model(model)
    nn.init.zeros_(ema_model.weight)

    update_ema(ema_model, model, decay=0.75)

    assert not ema_model.training
    assert all(not parameter.requires_grad for parameter in ema_model.parameters())
    torch.testing.assert_close(ema_model.weight, torch.full_like(ema_model.weight, 0.5))
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest tests/test_main_v3_training_helpers.py -v`

Expected: collection fails because `create_ema_model` and `update_ema` do not exist yet. This establishes that the EMA behavior is missing before implementation.

### Task 2: Implement residual attention and EMA helpers

**Files:**
- Modify: `main_v3.py:1-110`
- Test: `tests/test_main_v3_training_helpers.py`

- [ ] **Step 1: Add the required import and EMA helpers**

Add `import copy`, then define:

```python
def create_ema_model(model):
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_parameter, model_parameter in zip(
        ema_model.parameters(), model.parameters()
    ):
        ema_parameter.lerp_(model_parameter, 1.0 - decay)
```

- [ ] **Step 2: Add normalized residual attention**

Add `self.norm = nn.GroupNorm(8, query_dim)`. In `forward`, preserve `residual = x`, construct queries from `self.norm(x)`, and return:

```python
attention_output = self.to_out(out).reshape(b, h, w, c).permute(0, 3, 1, 2)
return residual + attention_output
```

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run: `pytest tests/test_main_v3_training_helpers.py -v`

Expected: 2 tests pass.

### Task 3: Integrate EMA into training and generation

**Files:**
- Modify: `main_v3.py:391-576`
- Test: `tests/test_main_v3_training_helpers.py`

- [ ] **Step 1: Pass the EMA model into training**

Add `ema_unet` after `unet` in `train_diffusion`. Immediately after `optimizer.step()`, call:

```python
update_ema(ema_unet, unet)
```

- [ ] **Step 2: Construct and use the EMA model**

After constructing `unet`, create:

```python
ema_unet = create_ema_model(unet)
```

Pass `ema_unet` into `train_diffusion`. Preserve the raw `unet.pth` save, then pass `ema_unet` instead of `unet` to `generate_save_and_evaluate_fid`.

- [ ] **Step 3: Run focused tests and static integration checks**

Run: `pytest tests/test_main_v3_training_helpers.py -v`

Expected: 2 tests pass.

Then inspect the diff and confirm `update_ema(ema_unet, unet)` immediately follows `optimizer.step()`, generation receives `ema_unet`, and the per-sample CFG mask remains unchanged.
