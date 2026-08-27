import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F






# -------------------------------------------------------------------------------------------------
# Positional Encoding (1D)
# -------------------------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        half = dim//2
        div  = torch.exp(torch.arange(half).float() * -(np.log(10000.0)/half))
        pe[:, :half]       = torch.sin(pos*div)
        pe[:, half:2*half] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x, start, end):
        # x: [B, N, dim]
        return x + self.pe[:,start:end,:]  


# -------------------------------------------------------------------------------------------------
# Initialize topological features: patch -> filtration embedding -> CLS
# -------------------------------------------------------------------------------------------------
class TopoPatchEmbeddings(nn.Module):
    def __init__(self,combination,num_statis,h_dim,patch_size ):
        super().__init__()
        self.combination = combination
        self.num_statis = num_statis
        self.patch_size = (patch_size,num_statis)
        self.projection = nn.Conv2d(combination, h_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, topological_features):
        B, combination, L, num_statis = topological_features.shape
        
        if combination != self.combination:
            raise ValueError(
                "Make sure that the element-specific number of the Topological features match with the one set in the configuration."
            )
        if num_statis != self.num_statis:
            raise ValueError(
                "Make sure that the statistical property number of the Topological features match with the one set in the configuration."
            )
        
        x = self.projection(topological_features).flatten(2).transpose(1, 2)
        return x


class TopoEmbeddings(nn.Module):
    def __init__(self,combination,num_statis,h_dim,patch_size,max_len):
        super().__init__()
        self.patch_embed = TopoPatchEmbeddings(combination,num_statis,h_dim,patch_size)
        # filtration embedding
        self.position_emb = PositionalEncoding(dim=h_dim, max_len=max_len)
        
        # cls
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)

    def encode(self,topological_features):
        x = self.patch_embed(topological_features)
        B, L, D = x.shape
        cls_tokens = self.cls_token.expand(B, 1, D) 
        embeddings = torch.cat((cls_tokens, x), dim=1) 

        # position for whole sequence (cls at position 0)
        embeddings = self.position_emb(embeddings, 0, embeddings.size(1))
        return embeddings
        

# -------------------------------------------------------------------------------------------------
# Sheaf Transformer Components
# -------------------------------------------------------------------------------------------------

class SheafValueTransformLinear(nn.Module):
    def __init__(self, D, H, d):
        super().__init__()
        self.D = D # dim
        self.H = H # heads
        self.d = d # stalk_dim 

        # W_i: (H, D, d), W_j: (H, D, d), W_v: (H, d, d), b: (H, d)
        self.Wi = nn.Parameter(torch.empty(H, D, d))
        self.Wj = nn.Parameter(torch.empty(H, D, d))
        self.Wv = nn.Parameter(torch.empty(H, d, d))
        self.b  = nn.Parameter(torch.zeros(H, d))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wi)
        nn.init.xavier_uniform_(self.Wj)
        nn.init.xavier_uniform_(self.Wv)
        nn.init.zeros_(self.b)

    def forward(self, x, v, attn):
        """
        x:    (B, N, D)
        v:    (B, H, N, d)   
        attn: (B, H, N, N)     
        return out: (B, H, N, d)
        """
        # Wi@Xi + b
        # x @ Wi: (B,N,D) with (H,D,d) -> (B,H,N,d)
        base_i = torch.einsum('bnd,hdk->bhnk', x, self.Wi) + self.b[None, :, None, :]

        # \sum_{j}alpha(ij)Wj@Xj
        xj_proj = torch.einsum('bnd,hdk->bhnk', x, self.Wj)              # (B,H,N,d)
        out_j   = torch.matmul(attn, xj_proj)                            # (B,H,N,d)

        # \sum_{j}alpha(ij)Wv@Vj
        vj_proj = torch.einsum('bhnk,hkm->bhnm', v, self.Wv)              # (B,H,N,d)
        out_v   = torch.matmul(attn, vj_proj)                            # (B,H,N,d)

        return base_i + out_j + out_v


class SheafValueTransformNonlinear(nn.Module):
    def __init__(self, D, H, d, r = 8, bias= True):
        super().__init__()
        self.D = D # dim
        self.H = H # heads
        self.d = d # stalk_dim 
        self.r = r # low rank dimension

        # Per-head low-rank factors generated from x:
        # U_net: x_i -> U_i (H, d, r)
        # V_net: x_j -> V_j (H, d, r)
        #
        # We implement as linear layers producing H*d*r, then reshape.
        self.U_net = nn.Linear(D, H * d * r, bias=bias)
        self.V_net = nn.Linear(D, H * d * r, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U_net.weight)
        nn.init.xavier_uniform_(self.V_net.weight)
        if self.U_net.bias is not None:
            nn.init.zeros_(self.U_net.bias)
        if self.V_net.bias is not None:
            nn.init.zeros_(self.V_net.bias)

    def forward(self, x, v, attn):
        B, N, D = x.shape

        # Generate per-head low-rank factors and constrain coefficients to [-1, 1]
        # U: (B, N, H, d, r) -> (B, H, N, d, r)
        U = torch.tanh(self.U_net(x)).view(B, N, self.H, self.d, self.r).permute(0, 2, 1, 3, 4).contiguous()
        V = torch.tanh(self.V_net(x)).view(B, N, self.H, self.d, self.r).permute(0, 2, 1, 3, 4).contiguous()

        # s_j = V^T@v_j 
        # V: (B,H,N,d,r), v: (B,H,N,d)  -> (B, H, N, r)
        s = torch.einsum("bhn dr, bhn d -> bhn r", V, v)

        # attention-weighted mixing over j: sum_j attn_{i,j} * s_j  -> (B, H, N, r)
        s_mix = torch.matmul(attn, s)

        # out_i = U_i @ s_mix_i  -> (B, H, N, d)
        out = torch.einsum("bhn dr, bhn r -> bhn d", U, s_mix)
        return out


class SheafTransformerLayer(nn.Module):
    def __init__(self, dim, heads, stalk_dim, low_rank, norm_typ,dropout):
        super().__init__()
        assert dim % heads == 0
        self.norm_typ = norm_typ
        self.heads,self.stalk_dim = heads,stalk_dim
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(dim, heads*stalk_dim)
        self.W_k = nn.Linear(dim, heads*stalk_dim)
        self.W_v = nn.Linear(dim, heads*stalk_dim)
        self.W_o = nn.Linear(heads*stalk_dim, dim)
        self.norm1= nn.LayerNorm(dim)
        self.norm2= nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(
            nn.Linear(dim, dim*4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim*4, dim)
        )
        
        
        #self.sheaf_transform = SheafValueTransformLinear(dim,heads,stalk_dim)
        self.sheaf_transform = SheafValueTransformNonlinear(dim,heads,stalk_dim,low_rank)
        
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def post_norm_forward(self,x):
        # post-norm
        B,L,D = x.shape
        q = self.W_q(x).view(B,L,self.heads,self.stalk_dim).transpose(1,2)
        k = self.W_k(x).view(B,L,self.heads,self.stalk_dim).transpose(1,2)
        v = self.W_v(x).view(B,L,self.heads,self.stalk_dim).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2,-1)) / (math.sqrt(self.stalk_dim))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        out = self.sheaf_transform(x, v, attn)   # (B,H,L,D)
        out = out.transpose(1,2).reshape(B, L, -1)
        out    = self.W_o(out)

        x2     = self.norm1(x + self.dropout(out))
        x3     = self.ffn(x2)
        return self.norm2(x2 + self.dropout(x3))
    
    def pre_norm_forward(self,x1):
        # pre-norm
        x = self.norm1(x1)
        B,L,D = x.shape
        q = self.W_q(x).view(B,L,self.heads,self.stalk_dim).transpose(1,2)
        k = self.W_k(x).view(B,L,self.heads,self.stalk_dim).transpose(1,2)
        v = self.W_v(x).view(B,L,self.heads,self.stalk_dim).transpose(1,2)
        
        scores = torch.matmul(q, k.transpose(-2,-1)) / (math.sqrt(self.stalk_dim))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        out = self.sheaf_transform(x, v, attn)   # (B,H,L,D)
        out = out.transpose(1,2).reshape(B, L, -1)
        out    = self.W_o(out)

        x = x1 + self.dropout(out)
        return x + self.dropout(self.ffn(self.norm2(x)))
    
    def forward(self,x):
        if self.norm_typ=='pre_norm':
            x = self.pre_norm_forward(x)
            return x
        elif self.norm_typ=='post_norm':
            x = self.post_norm_forward(x)
            return x




# -------------------------------------------------------------------------------------------------
# Encoder
# -------------------------------------------------------------------------------------------------
class TopoEncoder(nn.Module):
    def __init__(self, dim, heads, stalk_dim, low_rank, dropout, num_layers, norm_typ):
        super().__init__()
        
        self.layers = nn.ModuleList([ SheafTransformerLayer(dim, heads, stalk_dim, low_rank, norm_typ,dropout) for _ in range(num_layers) ])
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x


# -------------------------------------------------------------------------------------------------
# CoPresheaf Transformer Backbone
# ------------------------------------------------------------------------------------------------- 
def advance_released_ctnn_rng(para):
    """Keep released CTNN seed initialization unchanged after removing unused modules."""
    init = nn.Linear(para.encoder_h_dim, para.rng_h_dim)
    token = torch.empty(1, 1, para.rng_h_dim)
    nn.init.xavier_uniform_(init.weight)
    nn.init.normal_(token, std=0.02)
    for _ in range(para.rng_num_layers):
        SheafTransformerLayer(
            para.rng_h_dim,
            para.rng_heads,
            para.rng_stalk_dim,
            para.low_rank,
            para.norm_typ,
            para.rng_dropout,
        )
    back = nn.Linear(para.rng_h_dim, para.combination*para.num_statis*para.patch_size)
    nn.init.xavier_uniform_(back.weight)


class CoPresheafBackbone(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.patch_size = para.patch_size
        self.topo_embed = TopoEmbeddings(para.combination,para.num_statis,para.encoder_h_dim,para.patch_size,para.max_len)
        self.encoder = TopoEncoder(para.encoder_h_dim,para.encoder_heads,para.encoder_stalk_dim,para.low_rank,para.encoder_dropout,para.encoder_num_layers,para.norm_typ)
        advance_released_ctnn_rng(para)
        
    def encode(self,topological_features):
        # prepare encoder input
        x = self.topo_embed.encode(topological_features)
        
        # encoder
        encoder_out = self.encoder(x)
        return encoder_out

        

# -------------------------------------------------------------------------------------------------
# Finetune
# -------------------------------------------------------------------------------------------------
class Finetune(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.dim = para.encoder_h_dim
        self.backbone = CoPresheafBackbone(para)
        
        self.fc = nn.Sequential(
            nn.Linear(para.encoder_h_dim, para.encoder_h_dim*2),
            nn.ReLU(),
            nn.Linear(para.encoder_h_dim*2, 1)
        )
    
    def forward(self,topological_features,pool='cls'):
        x = self.backbone.encode(topological_features)
        if pool=='cls':
            x = x[:,0,:]
        elif pool=='average':
            x = x.mean(axis=1)
        else:
            x = x[:,0,:]
        x = x.view(-1,self.dim) 
        x = self.fc(x)         
        return x  
        
        
        
        
        
        
