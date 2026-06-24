# Per-sample CFG dropout design

## Objective

Create `main_v3.py` from the current `main_v2.py` and change classifier-free-guidance dropout from one decision per batch to one independent decision per sample.

## Scope

- Preserve the current 15% dropout probability.
- Preserve the existing zero-context unconditional representation.
- Generate a Boolean mask with shape `(batch_size, 1, 1)` on the context tensor's device.
- Replace the selected samples' complete context tensors with zeros while leaving other samples unchanged.
- Make no other training, model, sampling, evaluation, or formatting changes.
- Do not execute the Python program.

## Data flow

After CLIP produces `context` with shape `(batch_size, sequence_length, embedding_dim)`, sample one random value for each batch element. Broadcast the resulting mask over the sequence and embedding dimensions, then zero only the selected contexts before the noisy images are passed to the UNet.

## Verification

Verification is static only:

1. Confirm `main_v3.py` otherwise matches `main_v2.py`.
2. Confirm the dropout mask contains one decision per batch element.
3. Confirm broadcasting covers the full context for each selected element.
4. Confirm the probability remains 0.15.
5. Do not run Python or training.
