# 200-epoch image-quality improvement design

## Objective

Improve FID under the 200-epoch budget while preserving as much of `main_v3.py`'s CLIP-T improvement as possible. The measured comparison is `CLIP_T: 0.2358, FID: 73.58` for v2 versus `CLIP_T: 0.2592, FID: 89.28` for v3. Lower FID is better.

## Empty-prompt classifier-free guidance

Encode one empty string with the existing frozen CLIP text encoder. During training, expand that embedding to the current batch and use the existing per-sample 15% mask to select between prompt embeddings and empty-prompt embeddings. During sampling, require an expanded empty-prompt embedding as the unconditional context instead of constructing an all-zero tensor.

The conditional and unconditional embeddings use identical tokenization settings: maximum length 77, maximum-length padding, and truncation enabled.

## Balanced denoising loss

Keep epsilon prediction and calculate MSE independently for each sample. Blend ordinary MSE and Min-SNR-weighted MSE with equal contributions:

```text
min_snr_weight = min(SNR, 5) / SNR
weight = 0.5 + 0.5 * min_snr_weight
loss = mean(weight * per_sample_mse)
```

Clamp the SNR denominator to a small positive floating-point value to avoid division by zero. This retains Min-SNR's emphasis on difficult noisy examples without nearly eliminating high-SNR reconstruction gradients that contribute fine detail and FID.

## Faster EMA adaptation

Update EMA parameters after each optimizer step with decay `0.999` instead of `0.9999`. At roughly 29,800 optimizer steps, the shorter EMA horizon follows the useful portion of a 200-epoch run more closely and reduces lag from early, low-quality weights.

## Sampling quality and guidance

Use CFG scale `2.5` instead of `3.0` in evaluation and image export. This reduces over-guidance artifacts while retaining conditional alignment. Use 100 DDIM inference steps instead of 50 to reduce discretization error without increasing training cost.

## Valid sampling range

Clamp the final denoised tensor to `[-1, 1]` before converting it to image space, then return values constrained to `[0, 1]`. Both internal evaluation and exported PNG conversion consume this same bounded result.

## Scope

- Modify `main_v3.py` and its focused tests.
- Preserve residual attention and EMA generation.
- Preserve the linear beta schedule and 15% per-sample conditioning dropout.
- Preserve 200 epochs, batch size 32, optimizer, and learning-rate schedule.
- Do not run model training or download model weights during verification.

## Verification

Add focused tests for blended Min-SNR weighting and the new EMA decay. Retain the existing residual-attention, empty-context, and sample-clamping tests. Verify configuration statically and run syntax compilation without downloading model weights or launching training.

## Expected tradeoff

The changes are intended to reduce FID while limiting CLIP-T regression, but metric improvement cannot be guaranteed without retraining because diffusion training and evaluation are stochastic. A controlled run should use the same seed, dataset, and evaluation image count as the reported baselines.
