import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, depth):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        
        # self.layers = nn.ModuleList([
        #     nn.MultiheadAttention(hidden_size, num_heads)
        #     for _ in range(depth)
        # ])
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=depth, num_decoder_layers=depth)
        
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Initialize weights for query_projection to zeros
        nn.init.constant_(self.query_projection.weight, 0)
        nn.init.constant_(self.query_projection.bias, 0)

        # Initialize weights for key_value_projection to ones
        nn.init.constant_(self.key_value_projection.weight, 1)
        nn.init.constant_(self.key_value_projection.bias, 0)  
        
        #initialize the MultiheadAttention layers weights
        # for layer in self.layers:
        #     nn.init.constant_(layer.in_proj_weight[:hidden_size, :], 0)
        #     nn.init.constant_(layer.in_proj_weight[hidden_size:2*hidden_size, :], 1)
        #     nn.init.constant_(layer.in_proj_weight[2*hidden_size:, :], 1)
        #     #modify the bias
        #     nn.init.constant_(layer.in_proj_bias[:hidden_size], 0)
        #     nn.init.constant_(layer.in_proj_bias[hidden_size:2*hidden_size], 0)
        #     nn.init.constant_(layer.in_proj_bias[2*hidden_size:], 0)


    def forward(self, query, key_value):
        # Project the query and key_value
        # query = self.query_projection(query)
        # key_value = self.key_value_projection(key_value)
        
        # # Expand the dimensions of the query and key_value to match the expected shape
        # query_expanded = query.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        # key_value_expanded = key_value.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        # Ensure query has the correct shape
        # if query.dim() == 4:
        #     query = query.squeeze(1)  # Remove the extra dimension if present
        
        # # Ensure key_value has the correct shape
        # if key_value.dim() == 4:
        #     key_value = key_value.squeeze(1)
        
        # # Perform cross attention for each layer
        # for layer in self.layers:
        #     query, _ = layer(query, key_value, key_value, need_weights=False)
        # Project the output and remove the extra dimension
        # output = self.output_projection(query)
        
        # output = key_value
        
        output = self.transformer(query, key_value)
        
        
        return output

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_vision_adopter(config, delay_load=False, **kwargs):
    adopter_type = getattr(config, 'mm_adopter_type', 'cross_attention_1x')

    if adopter_type == 'identity':
        return IdentityMap()
    
    cross_attention_match = re.match(r'^cross_attention_(\d+)x$', adopter_type)
    if cross_attention_match:
        depth = int(cross_attention_match.group(1))
        return CrossAttention(config.hidden_size, 8, depth)

    raise ValueError(f'Unknown adopter type: {adopter_type}')