
__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np



class MaskedLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(MaskedLayerNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, input_x, mask=None):
        if mask is None:
            mask = torch.ones_like(input_x)
        mask = mask.int()
        input_x[torch.isnan(input_x)] = 0
        masked_input = input_x * mask
        mean = masked_input.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True)
        var = ((masked_input - mean) ** 2 * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True)
        output = (masked_input - mean) / torch.sqrt(var + self.eps)
        output = self.weight * output + self.bias
        output[torch.isnan(output)] = 0
        return output * mask


def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, others_args, d_ff=256, store_attn=False,
                attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = Mix_MultiheadAttention(d_model, n_heads,others_args=others_args, d_k=d_k, d_v=d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = MaskedLayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = MaskedLayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, masked:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        B, L, D = src.shape
        att_masked = masked.unsqueeze(1) #.repeat(1, L, 1)
        att_masked = ~att_masked

        ln_masked = masked.unsqueeze(2).repeat(1, 1, D)
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src, ln_masked)

        ## Multi-Head attention
        src2, attn = self.self_attn(src, src, src, attn_mask=att_masked)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src, ln_masked)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src, ln_masked)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src, ln_masked)

        return src

class Mix_MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, others_args,
        d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = Mix_ScaledDotProductAttention(d_model, n_heads, others_args=others_args, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output, attn_weights


class Mix_ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, others_args, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        B, H, L, E = q.shape
        attn_mask = attn_mask.unsqueeze(1).repeat(1, H//2, 1, 1)
        q1 = q[:, :H//2, :, :]
        q2 = q[:, H//2:, :, :]
        k1 = k[:, :H//2, :, :]
        k2 = k[:, H//2:, :, :]
        v1 = v[:, :H//2, :, :]
        v2 = v[:, H//2:, :, :] 
        attn_scores1 = torch.matmul(q1, k1) * self.scale 
        mask_shape = [B, 1, L, L]

        attn_mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(q1.device)
        attn_scores1 = attn_scores1.masked_fill_(attn_mask, -np.inf)
        attn_scores1 = attn_scores1.masked_fill_(attn_mask1, -np.inf)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores1.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        attn_weights1 = F.softmax(attn_scores1, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights1 = self.attn_dropout(attn_weights1)

        # compute the new values given the attention weights
        output1 = torch.matmul(attn_weights1, v1)    

        attn_scores2 = torch.matmul(q2, k2) * self.scale 
        mask_shape = [B, 1, L, L]
        attn_mask2 = torch.tril(torch.ones(mask_shape, dtype=torch.bool), diagonal=-1).to(q2.device)
        attn_scores2 = attn_scores2.masked_fill_(attn_mask, -np.inf)
        attn_scores2 = attn_scores2.masked_fill_(attn_mask2, -np.inf)

        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores2.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        attn_weights2 = F.softmax(attn_scores2, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights2 = self.attn_dropout(attn_weights2)

        # compute the new values given the attention weights
        output2 = torch.matmul(attn_weights2, v2) 

        output = torch.cat((output1, output2), dim=1)
        attn_weights = torch.cat((attn_weights1, attn_weights2), dim=1)
        attn_scores = torch.cat((attn_scores1, attn_scores2), dim=1)
                
        return output, attn_weights


