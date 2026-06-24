# Epoch loss curve design

## Objective

Update `main_v3.py` to record mean denoising loss per training epoch and save a headless loss-versus-epoch plot after training.

## Loss collection

Within `train_diffusion`, accumulate each detached scalar batch loss and the number of batches processed during the current epoch. At the end of the epoch, append the arithmetic mean to an `epoch_losses` list. Return that list after all epochs finish. Raise no new behavior for an empty dataloader; such an epoch records no value.

## Plot generation

Add `save_loss_curve(epoch_losses, output_path="loss_curve.png")`. Import Matplotlib's pyplot lazily inside the function, plot epochs numbered from 1 through the number of recorded values, label both axes, add a title and grid, apply tight layout, save the figure, and close it. Do not call `show()`.

## Integration

Capture the value returned by `train_diffusion` in the main execution block and call `save_loss_curve` immediately after training. The default output is `loss_curve.png` in the working directory.

## Scope

- Modify `main_v3.py` and focused tests only.
- Preserve the optimizer, Min-SNR loss, EMA updates, CFG behavior, generation, and checkpoints.
- Do not start model training during verification.

## Verification

Add focused tests confirming the plot uses one-based epoch numbers, saves to the requested path, and closes the figure. Run the existing helper suite to guard all prior behavior.
