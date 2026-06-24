# Per-sample CFG Dropout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `main_v3.py` from `main_v2.py` with independent 15% classifier-free-guidance dropout for every sample.

**Architecture:** Preserve the complete training program and change only the CFG-dropout mask in `train_diffusion`. Construct a device-local Boolean tensor of shape `(batch_size, 1, 1)` so PyTorch broadcasts each sample's dropout decision across its sequence and embedding dimensions.

**Tech Stack:** Python, PyTorch, NumPy, Hugging Face Transformers and Diffusers

---

### Task 1: Create the new training variant

**Files:**
- Source: `main_v2.py:417-418`
- Create: `main_v3.py`

- [ ] **Step 1: Copy the current source into the new variant**

Create `main_v3.py` as an exact copy of the current `main_v2.py`, preserving the user's existing source state.

- [ ] **Step 2: Replace batch-wide CFG dropout with per-sample masking**

Replace:

```python
if np.random.rand() < 0.15:
    context = torch.zeros_like(context)
```

with:

```python
cfg_dropout_mask = torch.rand(
    context.shape[0], 1, 1, device=context.device
) < 0.15
context = context.masked_fill(cfg_dropout_mask, 0.0)
```

This produces one independent decision for every sample and broadcasts it across the full CLIP context for that sample.

- [ ] **Step 3: Verify the change statically**

Inspect the source diff and confirm that:

- only `main_v3.py` contains the behavior change;
- the probability remains `0.15`;
- the mask shape is `(context.shape[0], 1, 1)`;
- the mask is created on `context.device`;
- no Python program or training process is executed.
