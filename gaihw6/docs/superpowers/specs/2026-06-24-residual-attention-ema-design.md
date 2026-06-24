# Residual attention and EMA generation design

## Objective

Update `main_v3.py` so cross-attention preserves its image input through a residual connection and all post-training evaluation and image generation use exponential-moving-average (EMA) UNet weights.

## Residual attention

Add `GroupNorm(8, query_dim)` to `CrossAttention`. Normalize the image features before constructing queries, retain the original image tensor as the residual, and return the original tensor plus the projected cross-attention output. Keys and values continue to come from the CLIP context. All existing attention call sites remain unchanged.

## EMA lifecycle

Create a frozen deep copy of the initialized UNet on the same device. After each successful optimizer step, update every EMA parameter in place using:

```text
ema = 0.9999 * ema + 0.0001 * trained_parameter
```

The EMA model is never optimized directly and remains in evaluation mode. Pass it to `generate_save_and_evaluate_fid` after training so both the internal FID evaluation and exported images use EMA weights. Preserve `unet.pth` as the raw trained UNet checkpoint and do not add another checkpoint file.

## Scope

- Modify only `main_v3.py` production behavior.
- Preserve per-sample CFG dropout.
- Preserve the current optimizer, learning-rate scheduler, loss, sampling settings, evaluator, and checkpoint filename.
- Do not swap parameters into the live training model.

## Verification

Verify that attention returns a residual sum, EMA parameters are frozen, EMA updates occur immediately after optimizer updates, and generation receives the EMA model. Run only focused validation appropriate to the local environment; do not start training.
