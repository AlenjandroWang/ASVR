"""
    Modified from https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqtransformer/transformers.py.
"""

import torch
import torch.nn as nn

from collections import OrderedDict
from torch.nn import functional as F
from transformers import PreTrainedModel, AutoConfig, AutoModel

from .attention import AttentionStack
from .configuration_rqtransformer import RQTransformerConfig, AttentionStackConfig, AttentionBlockConfig


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')

    return out


def top_p_probs(probs, p):    
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_idx_remove_cond = cum_probs >= p
    
    sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
    sorted_idx_remove_cond[..., 0] = 0
    
    indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)

    return norm_probs


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    """Take a 2-dim tensor, apply softmax along each row, and sample from
    each multinomial distribution defined by the rows.

    Args:
        logits: 2-dim tensor of shape (n_samples, logit_dim)
        temperature (float): softmax temperature
        top_k (Optional[int]): if given, sample only using `top_k` logits
        top_p (Optional[float]): if given, sample only using `top_p` logits

    Returns:
        samples: 1-dim integer tensor of shape (n_samples,)
    """

    logits = logits.to(dtype=torch.float32)
    logits = logits / temperature

    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    if torch.sum(torch.isnan(logits)):
        print('WARNING... NaN observed')
        logits[torch.isnan(logits)] = -float('Inf')

    probs = F.softmax(logits, dim=-1)
    
    if top_p is not None:
        probs = top_p_probs(probs, top_p)

    try:
        samples = torch.multinomial(probs, num_samples=1)
    except:
        raise RuntimeError

    return samples.view(-1)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class CasualDepthTransformerLayer(nn.Module):
    def __init__(self, config, depth):
        super().__init__()
        self.config = config
        embed_size = config.embed_dim  # 2048
        num_heads = embed_size // 128  # 16
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads,batch_first=True)
        self.layernorm1 = RMSNorm(embed_size)
        self.layernorm2 = RMSNorm(embed_size)
        self.linear1 = nn.Linear(embed_size * depth, 2 * embed_size)  # 8192, 4096
        self.linear2 = nn.Linear(2 * embed_size * depth, embed_size)

    def forward(self, x):
        # 获取输入的序列长度
        seq_len = x.size(1)
        # 创建因果掩码，确保只能看到当前和过去的信息
        # 自注意力层
        res = x
        x = self.layernorm1(x)
        src_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        _x, _ = self.self_attention(x, x, x,  is_causal=True, attn_mask=src_mask)
        res = _x + res  # (bs, sl, d)
        res = self.layernorm2(res) 
        x = torch.einsum('bld,tld->blt', res, torch.reshape(self.linear1.weight, (2 * self.config.embed_dim, -1, self.config.embed_dim)))  # linear1.reshape: 4096, 4, 2048
        x = torch.nn.functional.gelu(x)
        x = torch.einsum('blt,dlt->bld', x, torch.reshape(self.linear2.weight, (self.config.embed_dim, -1, 2 * self.config.embed_dim)))  
        return res + x


# old head
# class RQTransformer(PreTrainedModel):
#     config_class = RQTransformerConfig
#     def __init__(self, config: RQTransformerConfig):
#         super().__init__(config)
#         self.in_mlp_1 = nn.Linear(config.input_embed_dim_1, config.embed_dim)
#         # self.in_mlp_2 = nn.Linear(config.input_embed_dim_2, config.embed_dim)

#         blockconfig = AttentionBlockConfig(embed_dim=config.embed_dim, n_head=config.head["block"]["n_head"])
#         stackconfig = AttentionStackConfig(n_layer=config.head["n_layer"], block=blockconfig)
#         self.head_transformer = AttentionStack(stackconfig)

#         self.pos_emb_d = nn.Parameter(torch.zeros(1, config.block_size[2], config.embed_dim))
#         self.pos_emb_d.data.normal_(mean=0.0, std=0.02)

#         self.classifier_mlp = nn.Sequential(OrderedDict([
#             ('layer_norm', nn.LayerNorm(config.embed_dim)),
#             ('linear', nn.Linear(config.embed_dim, config.vocab_size)),
#         ]))
    
#     def embed_with_model_aux(self, code, model_aux, mode=None):
#         # mode = "visual" or "semantic"
#         xs_emb = model_aux.get_code_emb_with_depth(code, mode=mode)
#         return xs_emb

#     def forward(self, embed_from_body, code, model_aux=None, mode=None):
#         B, seq_len, D = code.shape

#         depth_ctx = self.embed_with_model_aux(code, model_aux, mode=mode)
#         depth_ctx = torch.cumsum(depth_ctx, dim=-2)
#         depth_ctx = self.in_mlp_1(depth_ctx)  # torch.Size([2, 3840, 4, 2560])

#         # embed_from_body = self.in_mlp_2(embed_from_body)

#         depth_ctx_full = torch.cat(
#             [
#                 embed_from_body.view(B, seq_len, 1, -1),
#                 depth_ctx[:, :, :-1, :],  # torch.Size([2, 3840, 3, 2560])
#             ],
#             dim=-2,
#         )

#         depth_ctx_full = depth_ctx_full.reshape(B * seq_len, D, -1)
#         depth_ctx_full = depth_ctx_full + self.pos_emb_d[:, :D, :]

#         head_outputs = self.head_transformer(depth_ctx_full)
#         head_outputs = head_outputs.reshape(B, seq_len, D, -1)
#         head_outputs = self.classifier_mlp(head_outputs)

#         return head_outputs
    
#     def generate(self, embed_from_body, model_aux=None, cfg=3.0, mode=None):
#         top_k = 900
#         top_p = 0.96
#         generate_idx = 1
#         B, seq_len, _ = embed_from_body.shape  # 1, 1, 2048

#         # embed_from_body = self.in_mlp_2(embed_from_body)

#         depth_ctx_full = embed_from_body.view(B, seq_len, 1, -1)
#         depth_ctx_full = depth_ctx_full.reshape(B * seq_len, generate_idx, -1)
#         depth_ctx_full = depth_ctx_full + self.pos_emb_d[:, :generate_idx, :]

#         head_outputs = self.head_transformer(depth_ctx_full)  # torch.Size([1, 1, 2560])
#         head_outputs = head_outputs.reshape(B, -1)

#         logits = self.classifier_mlp(head_outputs)  # 1, 16384

#         logits = logits[B//2:, :] + cfg * (logits[:B//2, :] - logits[B//2:, :])
#         code = sample_from_logits(logits, temperature=1.0, top_p=top_p, top_k=top_k)  # torch.Size([1])
#         code = code.reshape(B//2, seq_len, 1).repeat(2, 1, self.pos_emb_d.shape[1])

#         for i in range(self.pos_emb_d.shape[1]-1):  # self.pos_emb_d.shape[1]-1=4-1=3
#             generate_idx += 1
#             depth_ctx = self.embed_with_model_aux(code, model_aux, mode=mode)
#             depth_ctx = torch.cumsum(depth_ctx, dim=-2)[:, :, :i+1, :]
#             if len(depth_ctx.shape) == 3:
#                 depth_ctx = depth_ctx.unsqueeze(2)
#             depth_ctx = self.in_mlp_1(depth_ctx)

#             depth_ctx_full = torch.cat(
#                 [
#                     embed_from_body.view(B, seq_len, 1, -1),
#                     depth_ctx,
#                 ],
#                 dim=-2,
#             )

#             depth_ctx_full = depth_ctx_full.reshape(B * seq_len, generate_idx, -1)
#             depth_ctx_full = depth_ctx_full + self.pos_emb_d[:, :generate_idx, :]

#             head_outputs = self.head_transformer(depth_ctx_full)  # torch.Size([1, 2/3/4, 2560])
#             head_outputs = head_outputs[:, -1, :]

#             logits = self.classifier_mlp(head_outputs)

#             logits = logits[B//2:, :] + cfg * (logits[:B//2, :] - logits[B//2:, :])
#             code_generate = sample_from_logits(logits, temperature=1.0, top_p=top_p, top_k=top_k)
#             code_generate = code_generate.reshape(B//2, seq_len).repeat(2, 1)  # code.shape=torch.Size([2, 1, 4]) code_generate.shape=torch.Size([2, 1])
#             code[:, :, i+1] = code_generate

#         out_features = self.embed_with_model_aux(code, model_aux, mode=mode)
#         out_features = torch.cumsum(out_features, dim=-2)[:, :, -1, :]

#         return out_features, code

# #new head
class RQTransformer(PreTrainedModel):
    config_class = RQTransformerConfig
    def __init__(self, config: RQTransformerConfig):
        super().__init__(config)
        self.in_mlp_1 = nn.Linear(config.input_embed_dim_1, config.embed_dim)  # 1024, llm_hidden_size(2048)

        self.head_transformer = nn.ModuleList([
            CasualDepthTransformerLayer(config, config.block_size[-1])
            for _ in range(3)
        ])
        self.headnorm = RMSNorm(config.embed_dim) 
        self.heads = nn.ModuleList([
            nn.Linear(config.embed_dim, config.vocab_size[i])
            for i in range(config.block_size[-1])
        ])
        self.gradient_checkpointing = True
    
    def embed_with_model_aux(self, code, model_aux, mode=None):
        # mode = "visual" or "semantic"
        xs_emb = model_aux.get_code_emb_with_depth(code, mode=mode)
        return xs_emb

    def forward(self, embed_from_body, code, model_aux=None, mode=None):
        B, seq_len, D = code.shape  # 59, 256, 4

        depth_ctx = self.embed_with_model_aux(code, model_aux, mode=mode)
        depth_ctx = torch.cumsum(depth_ctx, dim=-2)
        depth_ctx = self.in_mlp_1(depth_ctx)  # torch.Size([59, 256, 4, llm_hidden_size(2048)])
        depth_ctx_full = torch.cat(
            [
                embed_from_body.view(B, seq_len, 1, -1),
                depth_ctx[:, :, :-1, :],
            ],
            dim=-2,
        )  # B, 256, 4, 2048

        depth_ctx_full = depth_ctx_full.reshape(B * seq_len, D, -1)  # B*256, 4, 2048

        for i, tlayer in enumerate(self.head_transformer):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                depth_ctx_full  = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(tlayer), depth_ctx_full,
                )
            else:
                depth_ctx_full  = tlayer(
                    depth_ctx_full,
                )

        depth_ctx_full = self.headnorm(depth_ctx_full)  # B*256, 4, 2048
        logits = [head(depth_ctx_full[:, i]) for i, head in enumerate(self.heads)]  # logits[0].shape = B*256, 16384(codebook_size)

        return logits
    
    def generate(self, embed_from_body, model_aux=None, cfg=3.0, mode=None):
        top_k = 900
        # top_k = 10
        # top_k = 3
        top_p = 0.96

        B, seq_len, _ = embed_from_body.shape  # 1, 1, 2048
        next_token_ids = torch.zeros(B, 1, 8, dtype=torch.long).to(embed_from_body.device)
        for i in range(8):
            logits = self(embed_from_body, next_token_ids, model_aux, mode=mode)
            next_token_logits = logits[i].clone()

            next_token_logits = next_token_logits[B//2:, :] + cfg * (next_token_logits[:B//2, :] - next_token_logits[B//2:, :])
            next_tokens = sample_from_logits(next_token_logits, temperature=1.0, top_p=top_p, top_k=top_k) 
            next_tokens = next_tokens.reshape(B//2, seq_len).repeat(2, 1)
            next_token_ids[:, :, i] = next_tokens

        out_features = self.embed_with_model_aux(next_token_ids, model_aux, mode=mode)
        out_features = torch.cumsum(out_features, dim=-2)[:, :, -1, :]
        # out_features = self.in_mlp_1(out_features)

        return out_features, next_token_ids


AutoConfig.register("rqtransformer_model", RQTransformerConfig)
AutoModel.register(RQTransformerConfig, RQTransformer)