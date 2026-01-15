"""
Vision Transformer (ViT) implementation for RSST pruning
Adapted for CIFAR-100 (32x32 images)
"""
import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    """将图像分割成patches并进行embedding"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head Self-Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 保存为属性，供剪枝使用
        self.scale = self.head_dim ** -0.5
        
        # 用于剪枝的线性层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        # 动态计算embed_dim，因为结构化剪枝后会改变
        embed_dim = self.num_heads * self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP模块"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=192, depth=9, num_heads=12, mlp_ratio=2.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token & Position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification head (use cls token)
        x = x[:, 0]
        x = self.head(x)
        
        return x


def vit_small(num_classes=100, img_size=32, pretrained=False):
    """ViT-Small for CIFAR"""
    model = VisionTransformer(
        img_size=img_size,
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=num_classes
    )
    
    if pretrained:
        print("⚠️  Note: 加载预训练权重需要安装timm库 (pip install timm)")
        try:
            import timm
            # 加载ImageNet预训练的ViT-Small
            pretrained_model = timm.create_model('vit_small_patch16_224', pretrained=True)
            # 复制可以迁移的权重（除了patch embedding和分类头）
            load_pretrained_weights(model, pretrained_model, num_classes)
            print("✓ 成功加载预训练权重")
        except ImportError:
            print("✗ 未安装timm库，使用随机初始化")
        except Exception as e:
            print(f"✗ 加载预训练权重失败: {e}，使用随机初始化")
    
    return model


def vit_tiny(num_classes=100, img_size=32, pretrained=False):
    """ViT-Tiny for CIFAR (更小，训练更快)"""
    model = VisionTransformer(
        img_size=img_size,
        patch_size=4,
        embed_dim=192,
        depth=9,
        num_heads=3,
        mlp_ratio=2,
        qkv_bias=True,
        num_classes=num_classes
    )
    
    if pretrained:
        print("⚠️  Note: ViT-Tiny通常没有ImageNet预训练权重，使用随机初始化")
    
    return model


def vit_base(num_classes=100, img_size=32, pretrained=False):
    """ViT-Base for CIFAR"""
    model = VisionTransformer(
        img_size=img_size,
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=num_classes
    )
    
    if pretrained:
        print("⚠️  Note: 加载预训练权重需要安装timm库 (pip install timm)")
        try:
            import timm
            # 加载ImageNet预训练的ViT-Base
            pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            # 复制可以迁移的权重
            load_pretrained_weights(model, pretrained_model, num_classes)
            print("✓ 成功加载预训练权重")
        except ImportError:
            print("✗ 未安装timm库，使用随机初始化")
        except Exception as e:
            print(f"✗ 加载预训练权重失败: {e}，使用随机初始化")
    
    return model


def load_pretrained_weights(model, pretrained_model, num_classes):
    """
    从预训练模型加载权重到自定义ViT模型
    
    Args:
        model: 自定义的ViT模型
        pretrained_model: timm预训练模型
        num_classes: 目标类别数
    """
    print("正在迁移预训练权重...")
    model_dict = model.state_dict()
    pretrained_dict = pretrained_model.state_dict()
    
    # 只加载形状匹配的权重
    transferred = 0
    skipped = 0
    
    for name, param in pretrained_dict.items():
        if name in model_dict:
            if model_dict[name].shape == param.shape:
                model_dict[name] = param
                transferred += 1
            else:
                skipped += 1
                # print(f"  跳过 {name}: 形状不匹配 {model_dict[name].shape} vs {param.shape}")
        else:
            skipped += 1
    
    model.load_state_dict(model_dict)
    print(f"  ✓ 迁移了 {transferred} 个参数")
    print(f"  ⚠ 跳过了 {skipped} 个参数（形状不匹配或不存在）")
    print(f"  ℹ 分类头和patch embedding已重新初始化以适配CIFAR")

