import torch
import timm
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Helper

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def create_enc(backbone_name, num_slices, layer_num:int=1):
    assert backbone_name in ["resnet18", "resnet34", "resnet50"], \
        f"Backbone f{backbone_name} is not in the list"
    model = timm.create_model(model_name=backbone_name, pretrained=True, 
                              in_chans=num_slices)
    img_enc = nn.Sequential(*list(model.children())[:(layer_num-6)])
    return img_enc


# Pre-layernorm
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# Feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# Attention
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            # Cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1) 
        
        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Transformer encoder, for multi-planes
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), 
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# Cross attention transformer
class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), 
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), 
            ]))
    
    def forward(self, ax_tokens, cor_tokens):
        (ax_cls, ax_patch_tokens), (cor_cls, cor_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (ax_tokens, cor_tokens))

        for ax_attend_cor, cor_attend_ax in self.layers:
            ax_cls = ax_attend_cor(ax_cls, context=cor_patch_tokens, kv_include_self=True) + ax_cls
            cor_cls = cor_attend_ax(cor_cls, context=ax_patch_tokens, kv_include_self=True) + cor_cls
            
        ax_tokens = torch.cat((ax_cls, ax_patch_tokens), dim=1)
        cor_tokens = torch.cat((cor_cls, cor_patch_tokens), dim=1)

        return ax_tokens, cor_tokens


# Multi-plane encoder
class MultiPlaneEncoder(nn.Module):
    def __init__(
        self, 
        *, 
        depth, 
        dim,  
        cross_attn_heads,
        plane_enc_params, 
        cross_attn_depth, 
        cross_attn_dim_head=64, 
        dropout=0., 
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=dim, dropout=dropout, **plane_enc_params), 
                Transformer(dim=dim, dropout=dropout, **plane_enc_params), 
                CrossTransformer(
                    dim=dim, depth=cross_attn_depth, heads=cross_attn_heads, 
                    dim_head=cross_attn_dim_head, dropout=dropout)
            ]))
    
    def forward(self, ax_tokens, cor_tokens):
        for ax_enc, cor_enc, cross_attend in self.layers:
            ax_tokens, cor_tokens = ax_enc(ax_tokens), cor_enc(cor_tokens)
            ax_tokens, cor_tokens = cross_attend(ax_tokens, cor_tokens)
        
        return ax_tokens, cor_tokens


# Patch_based image to shared position embedding
class SharedPostionEmbedder(nn.Module):
    def __init__(
        self, 
        *, 
        dim, 
        image_size, 
        in_channels, 
        patch_size, 
        dropout=0.,
    ):
        super().__init__()
        assert image_size % patch_size == 0, \
            "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), 
            nn.LayerNorm(patch_dim), 
            nn.Linear(patch_dim, dim), 
            nn.LayerNorm(dim)
        )
        
        self.shared_pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.ax_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cor_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, ax_out, cor_out):
        ax_out = self.to_patch_embedding(ax_out)
        cor_out = self.to_patch_embedding(cor_out)
        b, n, _ = ax_out.shape # ax and cor are same
        
        ax_cls_tokens = repeat(self.ax_cls_token, '() n d -> b n d', b=b)
        cor_cls_tokens = repeat(self.cor_cls_token, '() n d -> b n d', b=b)
        
        ax_tokens = torch.cat((ax_cls_tokens, ax_out), dim=1)
        ax_tokens += self.shared_pos_embedding[:, :(n + 1)]

        cor_tokens = torch.cat((cor_cls_tokens, cor_out), dim=1)
        cor_tokens += self.shared_pos_embedding[:, :(n + 1)]

        return self.dropout(ax_tokens), self.dropout(cor_tokens)


class AnkleNet(nn.Module):
    def __init__(
        self, 
        *, 
        backbone_name,
        image_size, 
        num_slices,
        num_classes,
        layer_num, 
        plane_dim, 
        plane_patch_size, 
        plane_enc_depth, 
        plane_enc_heads, 
        plane_enc_mlp_dim, 
        plane_enc_dim_head, 
        cross_attn_depth=2, 
        cross_attn_heads=8,
        cross_attn_dim_head=64, 
        depth=3, 
        droput=0.1, 
        emb_dropout=0.1,
    ):
        super().__init__()
        self.ax_img_enc = create_enc(backbone_name=backbone_name, num_slices=num_slices, layer_num=layer_num) # (B, 16, 256, 256) --> (B, 256, 16, 16)
        self.cor_img_enc = create_enc(backbone_name=backbone_name, num_slices=num_slices, layer_num=layer_num)
        
        in_channels = num_slices * 2**(layer_num+1)  # resnet18/34
        # in_channels = num_slices * 4 * 2**(layer_num+1) # resnet50
        image_size = image_size // 2**(layer_num+1)
        
        self.shared_pos_embedder = SharedPostionEmbedder(
            dim=plane_dim, image_size=image_size, in_channels=in_channels, patch_size=plane_patch_size, dropout=emb_dropout)
        
        self.multi_plane_encoder = MultiPlaneEncoder(
            depth=depth, 
            dim=plane_dim, 
            cross_attn_heads=cross_attn_heads, 
            cross_attn_dim_head=cross_attn_dim_head, 
            cross_attn_depth= cross_attn_depth, 
            plane_enc_params=dict(
                depth=plane_enc_depth, 
                heads=plane_enc_heads, 
                mlp_dim=plane_enc_mlp_dim, 
                dim_head=plane_enc_dim_head,
            ), 
            dropout=droput
        )
        
        self.ax_cls_head = nn.Sequential(nn.LayerNorm(plane_dim), nn.Linear(plane_dim, num_classes))
        self.cor_cls_head = nn.Sequential(nn.LayerNorm(plane_dim), nn.Linear(plane_dim, num_classes))
    
    def forward(self, x):
        ax_img = x["axial_img"].squeeze(1)
        cor_img = x["coronal_img"].squeeze(1)
        ax_out = self.ax_img_enc(ax_img)
        cor_out = self.cor_img_enc(cor_img)

        ax_tokens, cor_tokens = self.shared_pos_embedder(ax_out, cor_out)
        ax_tokens, cor_tokens = self.multi_plane_encoder(ax_tokens, cor_tokens)

        ax_cls, cor_cls = map(lambda t: t[:, 0], (ax_tokens, cor_tokens))

        ax_logits = self.ax_cls_head(ax_cls)
        cor_logits = self.cor_cls_head(cor_cls)

        return ax_logits + cor_logits
        

    
