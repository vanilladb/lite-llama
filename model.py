import os
import time
import math
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
torch.manual_seed(1337)
import torch.nn as nn
from torch.nn import functional as F

from utils import Timing

def sample(logits, temperature):
    if temperature < 1e-6:
        return int(logits.item().argmax())
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis, device=None):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device='cuda'))
        self.dim = dim

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    def load_weight(self, data, offset, device=None) -> int:
        self.weight = nn.Parameter(torch.tensor(data[offset : offset+self.dim], dtype=torch.float16, device='cuda'))
        return offset + self.dim

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.wq, self.wk, self.wv, self.wo = [nn.Linear(dim, dim, bias=False) for _ in range(4)]
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk, xv = [x.view(bsz, seqlen, self.n_heads, self.head_dim) for x in (xq, xk, xv)]
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # TODO: kv caching is broken
        # if start_pos == 0:
        #     keys, values = xk, xv
        # else:
        #     assert hasattr(self, 'cache_k'), "no cache"
        #     keys, values = torch.cat((self.cache_k, xk), dim=1), torch.cat((self.cache_v, xv), dim=1)
        
        # self.cache_k = keys
        # self.cache_v = values
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)
    
    def load_weight(self, data, offset) -> int:
        n_param = self.dim * self.dim
        self.wq.weight = nn.Parameter(torch.tensor(data[offset : offset+n_param], dtype=torch.float16, device='cuda').view(self.dim, self.dim))
        offset += n_param
        self.wk.weight = nn.Parameter(torch.tensor(data[offset : offset+n_param], dtype=torch.float16, device='cuda').view(self.dim, self.dim))
        offset += n_param
        self.wv.weight = nn.Parameter(torch.tensor(data[offset : offset+n_param], dtype=torch.float16, device='cuda').view(self.dim, self.dim))
        offset += n_param
        self.wo.weight = nn.Parameter(torch.tensor(data[offset : offset+n_param], dtype=torch.float16, device='cuda').view(self.dim, self.dim))
        return offset + n_param
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.dim = dim
        self.hidden_dim = hidden_dim

    def __call__(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    def load_weight(self, data, offset) -> int:
        n_param = self.dim * self.hidden_dim
        self.w1.weight = nn.Parameter(torch.tensor(data[offset : offset+n_param], dtype=torch.float16, device='cuda').view(self.hidden_dim, self.dim))
        offset += n_param
        self.w2.weight = nn.Parameter(torch.tensor(data[offset : offset+n_param], dtype=torch.float16, device='cuda').view(self.dim, self.hidden_dim))
        offset += n_param
        self.w3.weight = nn.Parameter(torch.tensor(data[offset : offset+n_param], dtype=torch.float16, device='cuda').view(self.hidden_dim, self.dim))
        return offset + n_param
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, multiple_of, n_heads, norm_eps, param):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, 4*dim, multiple_of)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.serialized_dir = f'serialized/{param}'

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))
    
    def fetch(self, layer: int):
        data = np.memmap(os.path.join(self.serialized_dir, f'layer{layer}.bin'), dtype=np.float16, mode='r')

        offset = 0
        offset = self.attention.load_weight(data, offset)
        offset = self.feed_forward.load_weight(data, offset)
        offset = self.attention_norm.load_weight(data, offset)
        self.ffn_norm.load_weight(data, offset)

class Transformer(nn.Module):
    def __init__(self, dim, multiple_of, n_heads, n_layers, norm_eps, vocab_size, max_batch_size=32, max_seq_len=2048, param='7B'):
        super().__init__()
        self.n_layers = n_layers
        self.cache_size = 2
        self.layer_cache = [TransformerBlock(dim, multiple_of, n_heads, norm_eps, param) for _ in range(self.cache_size)]
        self.cache_state = [False for _ in range(self.cache_size)]
        self.norm = RMSNorm(dim, norm_eps)
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len)

    def prefetcher(self):
        while True:
            for layer in range(self.n_layers):
                cache = layer % self.cache_size
                while self.cache_state[cache] is True:
                    time.sleep(0.001)
                    continue
                self.layer_cache[cache].fetch(layer)
                self.cache_state[cache] = True

    def forward(self, tokens, start_pos, device=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(device)

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos+1)

        for layer in range(self.n_layers):
            cache = layer % self.cache_size
            while self.cache_state[cache] is False:
                time.sleep(0.001)
                continue
            h = self.layer_cache[cache](h, start_pos, freqs_cis, mask)
            self.cache_state[cache] = False

        return self.output(self.norm(h)[:, -1, :]) # only compute the last logits
    

# **** files and arguments ****

PARAM = '13B'
WEIGHTS_DIR = Path("weights")
TOKENIZER_FILENAME = WEIGHTS_DIR / "tokenizer.model"
VOCAB_SIZE = 32000

args_7B = {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE, "param": "7B"}
args_13B = {"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE, "param": "13B"}
args_30B = {"dim": 6656, "multiple_of": 256, "n_heads": 52, "n_layers": 60, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE, "param": "30B"}
args_65B = {"dim": 8192, "multiple_of": 256, "n_heads": 64, "n_layers": 80, "norm_eps": 1e-06, "vocab_size": VOCAB_SIZE, "param": "65B"}

args = {
    '7B': args_7B,
    '13B': args_13B,
    '30B': args_30B,
    '65B': args_65B
}

if __name__ == '__main__':
    from sentencepiece import SentencePieceProcessor
    sp_model = SentencePieceProcessor(model_file=str(TOKENIZER_FILENAME))
    assert sp_model.vocab_size() == VOCAB_SIZE

    device = 'cuda' if torch.cuda.is_available else 'cpu'
    print(f'Device: {device}')
    
    model = Transformer(**args[PARAM]).half().to(device)
    model.load_state_dict(torch.load(f'serialized/{PARAM}/io.pt'))

    executor = concurrent.futures.ThreadPoolExecutor()
    prefetcher = executor.submit(model.prefetcher)

    prompt = "Elon Musk is "
    toks = [sp_model.bos_id()] + sp_model.encode(prompt)

    while True:
        with torch.inference_mode(), Timing('== '):
            logits = model(torch.tensor(toks).unsqueeze(dim=0).to(device), 0, device)
        tok = sample(logits, 0.7)
        start_pos = len(toks)
        toks.append(tok)
        cur = sp_model.decode(toks)
        print(cur)
