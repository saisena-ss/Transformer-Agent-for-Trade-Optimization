from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import time
import os
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist

@dataclass
class config:
    block_size:int = 1024
    n_layer:int = 6
    n_head:int = 8
    n_embed:int = 512
    input_dim:int = 10
    output_dim:int = 10
    
#multi head attention
class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed,config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        
        #actually it is mask
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
        
        
    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embed,dim=2)
        
        #split into multiple heads
        q = q.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # -> B, nheads, Time, head size
        k = k.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # -> B, nheads, Time, head size 
        v = v.view(B,T,self.n_head, C//self.n_head).transpose(1,2) # -> B, nheads, Time, head size
        
        #attention
        # att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        
        # #mask and softmax
        # att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        # att = F.softmax(att,dim=-1)
        
        # y = att @ v #(B,nheads, T, T) x (B, nheads, T, headsize) -> B, nheads, T, headsize
        
        #flash attention
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        
        y = y.transpose(1,2).contiguous().view(B,T,C)
        
        #last projection
        y = self.c_proj(y)
        
        return y
        
        
#let's define the attention block
class Block(nn.Module):
    """
    Steps:
    1. Layer norm
    2. Attention
    3. Residual connection
    4. layer norm
    5. feed forward
    6. Residual connection
    """
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x
            

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embed,config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self,x):
        return self.c_proj(self.gelu(self.c_fc(x)))
    
        
class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
                                wte = nn.Linear(config.input_dim, config.n_embed),
                                wpe = nn.Embedding(config.block_size,config.n_embed),
                                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                                ln_f = nn.LayerNorm(config.n_embed) 
                                ))
        self.lm_head = nn.Linear(config.n_embed,config.output_dim,bias=False)
        
        #weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        std = 0.02
        
        if isinstance(module,nn.Linear):
            if hasattr(module,"NANOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=std)
    
    
    def forward(self,idx,targets = None):
        if len(idx.shape) == 2:
            idx = idx.unsqueeze(0)
            
        B,T,C = idx.shape
        assert T <= self.config.block_size,f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        tok_emb = self.transformer.wte(idx) #(B,T,C)
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device) #(T)
        pos_emb = self.transformer.wpe(pos) #(T,C)
        x = tok_emb + pos_emb

        #forward pass
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = x[:,-1,:]
        
        out = self.lm_head(x) #(B,T,output_dim)
        
        if len(idx.shape) == 2:
            out = out.squeeze(0)
        # print(out.shape)
        return out
    