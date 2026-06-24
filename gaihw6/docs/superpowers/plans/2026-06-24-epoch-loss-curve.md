# Epoch Loss Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record mean training loss for every completed epoch and save a labeled loss curve to `loss_curve.png`.

**Architecture:** Keep loss aggregation inside `train_diffusion`, returning a plain list of epoch means. A separate plotting helper lazily uses Matplotlib so plotting remains isolated from model training and can be tested with a mocked pyplot module.

**Tech Stack:** Python, PyTorch, Matplotlib, unittest

---

### Task 1: Add a failing plotting-helper test

**Files:**
- Modify: `tests/test_main_v3_training_helpers.py`

- [ ] **Step 1: Add the test**

Import `patch` and `save_loss_curve`. Mock `matplotlib.pyplot`, call `save_loss_curve([2.0, 1.5], "curve.png")`, and assert that it plots epochs `[1, 2]`, saves to `curve.png`, and closes the figure.

- [ ] **Step 2: Verify RED**

Run: `python -m unittest discover -s tests -v`

Expected: import failure because `save_loss_curve` does not exist.

### Task 2: Implement the plotting helper

**Files:**
- Modify: `main_v3.py:35-70`

- [ ] **Step 1: Add `save_loss_curve`**

Implement a lazy pyplot import, one-based epoch positions, labeled axes, title, grid, tight layout, `savefig(output_path)`, and `close()` without calling `show()`.

- [ ] **Step 2: Verify GREEN**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

### Task 3: Collect epoch loss and invoke plotting

**Files:**
- Modify: `main_v3.py:427-495`
- Modify: `main_v3.py:653-670`

- [ ] **Step 1: Aggregate epoch losses**

Initialize `epoch_losses` before the progress context. For every epoch, initialize `epoch_loss = 0.0` and `batch_count = 0`; after calculating each loss, add `loss.detach().item()` and increment the count. Append `epoch_loss / batch_count` after the epoch when at least one batch was processed, then return `epoch_losses` after training.

- [ ] **Step 2: Save the curve after training**

Capture the list returned by `train_diffusion`, call `save_loss_curve(epoch_losses)`, and print the output filename.

- [ ] **Step 3: Verify integration**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

Inspect the training loop to confirm aggregation uses detached scalar values, EMA still updates after every optimizer step, Min-SNR remains active, and no training is started during verification.
