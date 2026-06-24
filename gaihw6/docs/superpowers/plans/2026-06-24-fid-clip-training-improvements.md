# FID and CLIP-T Training Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve text-conditioned diffusion training and output validity using empty-prompt CFG conditioning, Min-SNR loss weighting, and bounded image conversion.

**Architecture:** Add small tensor helpers for CFG selection, scheduler-derived Min-SNR weights, and image-range conversion. Cache one frozen CLIP empty-prompt embedding per training/evaluation function, expand it per batch, and pass it explicitly into the sampler.

**Tech Stack:** Python, PyTorch, unittest, Hugging Face Transformers and Diffusers

---

### Task 1: Add failing helper tests

**Files:**
- Modify: `tests/test_main_v3_training_helpers.py`

- [ ] **Step 1: Add tests for CFG selection, Min-SNR weights, and clamping**

Import `SimpleNamespace`, `apply_cfg_dropout`, `compute_min_snr_weights`, and `to_image_range`. Add tests asserting that a Boolean mask independently selects empty context, SNR values above gamma receive `gamma / snr`, and inputs outside `[-1, 1]` map safely into `[0, 1]`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest discover -s tests -v`

Expected: import failure because the three new helpers do not exist.

### Task 2: Implement tensor helpers and Min-SNR loss

**Files:**
- Modify: `main_v3.py:35-55`
- Modify: `main_v3.py:411-454`

- [ ] **Step 1: Add tensor helpers**

Implement:

```python
def apply_cfg_dropout(context, empty_context, dropout_mask):
    return torch.where(dropout_mask, empty_context.expand_as(context), context)


def compute_min_snr_weights(scheduler, timesteps, gamma=5.0):
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    alpha = alphas_cumprod[timesteps].float()
    epsilon = torch.finfo(alpha.dtype).eps
    snr = alpha / (1.0 - alpha).clamp_min(epsilon)
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp_min(epsilon)


def to_image_range(samples):
    return ((samples.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
```

- [ ] **Step 2: Encode and apply empty-prompt conditioning during training**

Before the epoch loop, tokenize one empty string with maximum-length padding, length 77, and truncation. Encode it once without gradients. Add truncation to prompt tokenization and replace `masked_fill` with `apply_cfg_dropout` using the expanded empty context.

- [ ] **Step 3: Replace scalar MSE with Min-SNR-weighted per-sample MSE**

Compute unreduced MSE, average over image dimensions, multiply by `compute_min_snr_weights(scheduler, t)`, and average the batch.

- [ ] **Step 4: Run tests and verify helper behavior is GREEN**

Run: `python -m unittest discover -s tests -v`

Expected: all helper, residual-attention, and EMA tests pass.

### Task 3: Use empty-prompt CFG and bounded outputs during generation

**Files:**
- Modify: `main_v3.py:378-409`
- Modify: `main_v3.py:459-529`

- [ ] **Step 1: Require unconditional context in the sampler**

Add `uncond_context` after `context` in `sample`, remove `torch.zeros_like(context)`, and return `to_image_range(x)`.

- [ ] **Step 2: Cache and pass empty-prompt context in generation**

At the start of `generate_save_and_evaluate_fid`, encode one empty prompt. For each evaluation and export batch, expand it to the context shape and pass it into `sample`. Add truncation to all prompt tokenization calls.

- [ ] **Step 3: Verify complete integration**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

Inspect the diff to confirm that zero unconditional contexts are gone, the 15% per-sample mask remains, Min-SNR gamma is 5.0, sample results are bounded, EMA generation remains active, and training is not started.
