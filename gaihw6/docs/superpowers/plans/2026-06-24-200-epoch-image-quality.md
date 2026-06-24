# 200-Epoch Image Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve FID within the fixed 200-epoch training budget while retaining v3's text alignment improvements.

**Architecture:** Keep the current model, scheduler, optimizer, empty-prompt CFG, and evaluation pipeline. Change only the loss weighting, EMA horizon, CFG strength, and inference step count, with helper-level tests and static configuration checks.

**Tech Stack:** Python, PyTorch, diffusers DDIM scheduler, unittest

---

### Task 1: Balance the denoising objective

**Files:**
- Modify: `tests/test_main_v3_training_helpers.py`
- Modify: `main_v3.py:55-60`
- Modify: `main_v3.py:501-505`

- [ ] **Step 1: Write the failing blended-weight test**

Import `compute_balanced_snr_weights` and add:

```python
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
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest discover -s tests -p "test_main_v3_training_helpers.py" -k balanced -v`

Expected: ERROR because `compute_balanced_snr_weights` cannot be imported.

- [ ] **Step 3: Implement the balanced weight helper and use it**

Add after `compute_min_snr_weights`:

```python
def compute_balanced_snr_weights(
    scheduler, timesteps, gamma=5.0, min_unweighted_fraction=0.5
):
    min_snr_weights = compute_min_snr_weights(scheduler, timesteps, gamma)
    return min_unweighted_fraction + (
        1.0 - min_unweighted_fraction
    ) * min_snr_weights
```

Change the training call to:

```python
loss_weights = compute_balanced_snr_weights(scheduler, t)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `python -m unittest discover -s tests -p "test_main_v3_training_helpers.py" -k balanced -v`

Expected: PASS.

### Task 2: Shorten the EMA horizon

**Files:**
- Modify: `tests/test_main_v3_training_helpers.py`
- Modify: `main_v3.py:43-48`

- [ ] **Step 1: Write the failing default-decay test**

Add:

```python
def test_ema_default_decay_tracks_a_shorter_training_run(self):
    model = nn.Linear(1, 1, bias=False)
    ema_model = create_ema_model(model)
    nn.init.ones_(model.weight)
    nn.init.zeros_(ema_model.weight)

    update_ema(ema_model, model)

    torch.testing.assert_close(
        ema_model.weight, torch.full_like(ema_model.weight, 0.001)
    )
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest discover -s tests -p "test_main_v3_training_helpers.py" -k default_decay -v`

Expected: FAIL because the current update is `0.0001`.

- [ ] **Step 3: Change the EMA default**

Change the signature to:

```python
def update_ema(ema_model, model, decay=0.999):
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2. Expected: PASS.

### Task 3: Tune sampling configuration

**Files:**
- Modify: `tests/test_main_v3_training_helpers.py`
- Modify: `main_v3.py:521-623`
- Modify: `main_v3.py:687-699`

- [ ] **Step 1: Write failing static configuration tests**

Add:

```python
def test_generation_defaults_use_balanced_guidance_and_more_steps(self):
    import inspect
    from main_v3 import generate_save_and_evaluate_fid

    signature = inspect.signature(generate_save_and_evaluate_fid)
    self.assertEqual(signature.parameters["cfg_scale"].default, 2.5)
    self.assertEqual(signature.parameters["ddim_steps"].default, 100)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest discover -s tests -p "test_main_v3_training_helpers.py" -k generation_defaults -v`

Expected: FAIL because `cfg_scale` is not a function parameter.

- [ ] **Step 3: Centralize and apply generation defaults**

Change the generation function tail to:

```python
    output_dir="generated_images/",
    ddim_steps=100,
    cfg_scale=2.5,
):
```

Replace both hardcoded `cfg_scale=3.0` sample arguments with:

```python
cfg_scale=cfg_scale,
```

Change the main invocation to `ddim_steps=100` and add `cfg_scale=2.5`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the command from Step 2. Expected: PASS.

### Task 4: Verify the complete change

**Files:**
- Verify: `main_v3.py`
- Verify: `tests/test_main_v3_training_helpers.py`

- [ ] **Step 1: Run the complete helper suite**

Run: `python -m unittest discover -s tests -p "test_main_v3_training_helpers.py" -v`

Expected: all tests PASS.

- [ ] **Step 2: Compile production and test code**

Run: `python -m py_compile main_v3.py tests/test_main_v3_training_helpers.py`

Expected: exit code 0 with no output.

- [ ] **Step 3: Check the final diff**

Run: `git diff --check -- main_v3.py tests/test_main_v3_training_helpers.py`

Expected: exit code 0 with no whitespace errors. Confirm no changes outside the agreed files and plan tracking.
