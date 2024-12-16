import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from dataclasses import dataclass

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        # TODO: Instead of using tanh, we can use non approximate version also
        #There is not much difference between the time for tanh and real GELU
        self.gelu = nn.GELU(approximate='tanh') # gpt2 uses tanh approximation so we're using it
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed) 
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class RotaryPositionEmbeddings(nn.Module):
    '''Rotary Position Embeddings, as described in the RoPE paper'''
    def __init__(self, config, base=10_000):
        super().__init__()
        self.base = base
        self.dim = config.n_embed
        self.max_seq_len = config.block_size
        self.config = config
        self.rope_init()
    
    def reset_parameters(self):
        self.rope_init()
    
    def rope_init(self):
        '''
        Initialize the RoPE cache with sin and cos values for each position.
        '''
        # Compute thetas for sin and cos
        theta = torch.pow(self.base, -2 * torch.arange(0, self.dim // 2).float() / self.dim)
        self.register_buffer('theta', theta, persistent=False)
        self.build_rope_cache()
    
    def build_rope_cache(self):
        '''
        Build the RoPE cache for the given block size.
        '''
        seq_idx = torch.arange(self.max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        # Compute position * theta for sin and cos
        # idx_theta = seq_idx.view(-1, 1) * self.theta.view(1, -1) # same functionality as einsum
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        hs_half = self.config.n_embed // self.config.n_head // 2  # Calculate hs // 2
        idx_theta = idx_theta[:, :hs_half]  # Slice to match hs_half

        # Precompute sin and cos
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)  # Shape: [block_size, n_emb // 2, 2]
        self.register_buffer('cache', cache, persistent=False)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): Input tensor of shape [b, seq_len, nh, hs].
        
        Returns:
            torch.Tensor: Rotated tensor of the same shape as input.
        '''
        b, seq_len, nh, hs = x.shape  # Extract input dimensions
        # TODO: if n_emb is 32, nh is 2, hs should be 16 (n_emb // nh) but it is 8, why?
        # if master_process: print(f"shape of x before reshaping in RoPE: {x.shape}")

        # Slice the RoPE cache to match the sequence length
        # print(f"shape of cache before slicing in RoPE: {self.cache.shape}")
        # rope_cache = self.cache[:seq_len].to(x.device)  # Shape: [seq_len, n_emb // 2, 2]
        rope_cache = self.cache[:seq_len, :hs // 2].to(x.device)
        # Reshape input for rotation (split last dim into pairs)
        x = x.reshape(*x.shape[:-1], -1, 2)  # Shape: [b, seq_len, nh, hs // 2, 2]
        # print(f"shape of x after reshaping in RoPE: {x.shape}")
        # this or x = x.view(b, seq_len, nh, hs // 2, 2) # Shape: [b, seq_len, nh, hs // 2, 2]
        # Add singleton dimensions to rope_cache for broadcasting
        rope_cache = rope_cache.unsqueeze(0).unsqueeze(2)  # Shape: [1, seq_len, 1, h_s // 2, 2]
        # rope_cache = rope_cache.view(-1, x.size(1), 1, x.size(3), 2)
        # print(f"Shape of rope_cache in RoPE: {rope_cache.shape}")
        
        # Perform the RoPE rotation
        rotated = torch.stack([
            x[..., 0] * rope_cache[..., 0] - x[..., 1] * rope_cache[..., 1],  # cos * even - sin * odd
            x[..., 1] * rope_cache[..., 0] + x[..., 0] * rope_cache[..., 1]   # sin * even + cos * odd
        ], dim=-1)  # Shape: [b, seq_len, nh, hs // 2, 2]

        # Flatten the last two dimensions back into the original shape
        # print(f"shape of rotated before flattening in RoPE: {rotated.shape}")
        # print(f"shape of rotated after flattening in RoPE: {rotated.flatten(-2).shape}")
        return rotated.flatten(-2).type_as(x)  # Shape: [b, seq_len, nh, hs]

# This is for self attention block
class CausalSelfAttention(nn.Module):
    def __init__(self, config):

        super().__init__() 
        assert config.n_embed % config.n_head == 0
        
        # This is for query, key and value projections
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)  # combined linear projection for all three Q, K, V
        # This is for output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # This is for scaling the weights
        self.c_proj.NANOGPT_SCALE_INIT = 1.0


        # Regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # not really a bias but a mask, but following OpenAI naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) 

        self.rope = RotaryPositionEmbeddings(config)

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence Length, Embedding dimensionality (n_embed)
        # d_k = d_v = n_embed // n_head
        # n_head -> Number of heads in the multi-head attention
        # create query, key, value matrices
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)  # B x T x n_embed -> B x T x 3*n_embed -> q, k, v each of shape B x T x n_embed
        # hs (head size) = n_embed // n_head
        # C == n_emb == n_head * hs == n_head * d_k == n_head * d_v
        q = q.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # q (BxTxC) reshaped to B x T x n_head x hs then transposed to B x n_head x T x hs
        k = k.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # same for k
        v = v.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # same for v

        q = self.rope(q)
        k = self.rope(k)

        # Attention mechanism
        # att = (q @ k.transpose(-2, -1)) * (1.0 / ((k.size(-1)) ** 0.5))
        # # Masked Attention
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # wow who knew flash attention was so easy to implement
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * (self.n_embed // self.n_head))
        # Output projection
        y = self.c_proj(y)
        return y

# This is for transformer block
class Block(nn.Module):
    # write 

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # didn't realize it was this easy to implement residual connections
        x = x + self.attn(self.ln_1(x)) # clean residual connections are desrable for deep models form an optimization perspective
        x = x + self.mlp(self.ln_2(x)) # also we perform layer normalization before self attention and MLP, in contrast to the original transformer
        # this is because it is more stable to normalize the input to each sub-layer, rather than the output
        # this is called pre-normalization and is used in the "An Image is Worth 16x16 Words" paper
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # Maximum sequence length
    vocab_size: int = 50257 # 50k "Byte Pair Encodings" (BPE) vocab size + 256 bytes tokens + 1 <|endoftoken|>
    # special end of sequence token delimits document boundaries and can start generation as well
    n_layer: int = 12 # Number of transformer blocks (how deep is the model)
    n_head: int = 12 # Number of heads in the multi-head attention (how wide is the model)
    n_embed: int = 768 # Embedding dimensionality

class rope_GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        # Developing Transformer
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embed), # Token embedding weights
            'wpe': nn.Embedding(config.block_size, config.n_embed), # Positional embedding weights
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # All transformer blocks
            'ln_f': nn.LayerNorm(config.n_embed)
        })

        # Final Linear layer after all transformer blocks
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    
        # Weight sharing scheme
        # This is for sharing the weights between token and positional embeddings
        # Reason: Since they are semantically similar, they should have similar weights
        self.lm_head.weight = self.transformer['wte'].weight

        # Initialize parameters with mean 0 and standard deviation 0.02 because 1/sqrt(768), 1/sqrt(1600)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # Here 2 is because one is for MLP and other is for attention mechanism
                # config.n_layer is number of transformer blocks
                # Why this? 
                # We send each layer after adding the residual connection and normalization
                # To make results' distribution normal with standard deviation = 0.02
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            # if module.padding_idx is not None:
            #     torch.nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, model block size is {self.config.block_size}"

        # IMP: Token and Positional Embeddings
        # pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        # pos_emb = self.transformer.wpe(pos)  #Positional Embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx)  #Token Embeddings of shape (B, T, n_embed)
        x = tok_emb # + pos_emb # broadcast along the batch dimension

        # Forward pass through each transformer block
        for block in self.transformer.h:
            x = block(x)
        
        # Final Linear layer
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)

        # Loss function
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss


    @classmethod
    def from_pretrained(cls, model_type):
        ## This is for loading the pretrained model
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import AutoModelForCausalLM
        print("Loading weights from pretrained gpt:", model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embed=768),           #124M parameters
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),   #345M parameters
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),    #774M parameters
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600)        #1558M parameters
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('attn.bias')]

        # init a huggingface GPT2 model
        model_hf = AutoModelForCausalLM.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Check parameters match
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('attn.bias')]
        assert set(sd_keys) == set(sd_keys_hf), f"Length mismatch - keys: {len(sd_keys)} and huggingface keys: {len(sd_keys_hf)}"

        # Copy shared weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys:
            if any(k.endswith(x) for x in transposed):
                assert sd[k].shape == sd_hf[k].shape[::-1]
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  #This is transpose but will not give warnings
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, lr, device):
        # We need all the named parameters that require gradients
        param_dict = {k: v for k, v in self.named_parameters()}
        param_dict = {k: v for k, v in param_dict.items() if v.requires_grad}
        
        # Bias does not need weight decay 
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # Optimizer
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        # Return number of elements (numel), which is the number of parameters
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        # Check AdamW optimizer and use the fused verison if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay, fused=use_fused)
        if master_process:
            print(100*'-')
            print(f"Using Fused AdamW: {fused_available}")
            print(f"# Decayed parameter tensors: {len(decay_params)} with {num_decay_params} parameters")
            print(f"# No Decay parameter tensors: {len(no_decay_params)} with {num_no_decay_params} parameters")
        return optimizer