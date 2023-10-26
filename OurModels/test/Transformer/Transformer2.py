import torch
import torch.nn as nn

transformer = nn.Transformer(d_model=158, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True)

src = torch.randn(1, 7, 158)
tgt = torch.randn(1, 7, 158)
outputs = transformer(src, tgt)
print(outputs.shape)
