import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads , dim_head):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x ,labels, con):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if labels!=None:
            attn_mask = get_attn_pad_mask(labels, labels).cuda()
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
            dots.masked_fill_(attn_mask, -1e9)
        con=con.unsqueeze(1).repeat(1, self.heads, 1, 1)
        dots=dots+torch.mul(con, dots) 
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.dropout(self.to_out(out)),attn.to('cpu')

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, labels, con):
        attenlist=[]
        for attn, ff in self.layers:
            o=attn(x,labels, con)
            x = o[0] + x
            x = ff(x) + x
            attenlist.append(o[1])
        return x,attenlist


class NMRformer(nn.Module):
    def __init__(self, input_dim, num_classes, dim, mlp_dim, depth=6, dim_head = 64, heads=8):
        super().__init__()


        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
            nn.Dropout(0.1)
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            # nn.GELU(),
            # nn.Linear(512, num_classes),
        )

    def forward(self, series, con, labels=None):
        x = self.to_patch_embedding(series)
        x = self.transformer(x,labels, con)
        return self.linear_head(x[0]).transpose(-1, -2),x[1]
    
    
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)