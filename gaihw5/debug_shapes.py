import torch
from main import UNet

model = UNet()
print('model created')

x = torch.randn(1, 4, 32, 32)
timestep = torch.tensor([10])

# Standardize timestep shape
if timestep.ndim == 0:
    timestep = timestep[None]

batch_ts = timestep * torch.ones(x.shape[0], dtype=timestep.dtype)
print('batch_ts', batch_ts.shape)

# time embedding
 t_emb = model.time_proj(batch_ts)
emb = model.time_embedding(t_emb)
print('t_emb', t_emb.shape, 'emb', emb.shape)

sample = model.conv_in(x)
print('conv_in', sample.shape)

down_block_res_samples = (sample,)
for idx, downsample_block in enumerate(model.down_blocks):
    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
    print('down', idx, 'out', sample.shape, 'res', [r.shape for r in res_samples])
    down_block_res_samples += tuple(res_samples)
print('after down', sample.shape, 'states', len(down_block_res_samples))

sample = model.mid_res1(sample, emb)
print('mid1', sample.shape)
sample = model.mid_attn(sample)
print('mid attn', sample.shape)
sample = model.mid_res2(sample, emb)
print('mid2', sample.shape)

for idx, upsample_block in enumerate(model.up_blocks):
    res_samples = down_block_res_samples[-3:]
    down_block_res_samples = down_block_res_samples[:-3]
    print('\nup', idx, 'in', sample.shape, 'res', [r.shape for r in res_samples])
    sample = upsample_block(sample, res_samples, emb)
    print('up', idx, 'out', sample.shape)

sample = model.conv_norm_out(sample)
print('norm out', sample.shape)
sample = model.conv_act(sample)
print('act', sample.shape)
sample = model.conv_out(sample)
print('final', sample.shape)
