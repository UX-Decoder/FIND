import torch
import torch.nn as nn

from .modules import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP


class ContentAttention(nn.Module):
    def __init__(self, num_layers, hidden_dim, nheads, pre_norm, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
    
    def forward(self, layer_id, content_variables):
        outputs = []
        for key, value in content_variables.items():
            output, _ = self.layers[layer_id](
                value['output'], value['src'],
                memory_mask=value['memory_mask'],
                memory_key_padding_mask=None if 'memory_key_padding_mask' not in value else value['memory_key_padding_mask'],  # here we do not apply masking on padded region
                pos=value['pos'], query_pos=value['query_pos']
            )
            outputs += [output]
        return torch.cat(outputs)

class ConditionAttention(nn.Module):
    def __init__(self, num_layers, hidden_dim, nheads, pre_norm, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
    
    def forward(self, layer_id, output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):        
        output = self.layers[layer_id](
            output, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            query_pos=query_pos
        )
        return output