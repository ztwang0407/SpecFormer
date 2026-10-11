from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from dinov2.vit import DinoVisionTransformer, vit_base, vit_large

from .mixstyle import MixStyle # 导入MixStyle

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, conv_14_28, use_mixstyle=False, mixstyle_p=0.5, mixstyle_alpha=0.1):
        super().__init__(backbone, position_embedding, conv_14_28)
        # dino的stride为14
        # self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        # 也即最后一层输出的名字
        self.vit_feat_name = f'res{backbone.n_blocks - 1}'

        # 根据参数决定是否初始化MixStyle
        self.use_mixstyle = use_mixstyle
        if self.use_mixstyle:
            self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
            
    def forward(self, tensor_list: NestedTensor, VPT_enable = False):
        out_tmp: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        if VPT_enable:
            x = self[0](tensor_list.tensors)[self.vit_feat_name]
        else:
            with torch.no_grad(): x = self[0](tensor_list.tensors)[self.vit_feat_name]
        # conv_14_28 ↓
        x = self[2](x)
        # conv_14_28 ↑
        
        # 在特征图被下采样到 stride 28 之后应用
        if self.use_mixstyle and self.training:
            x_mixed = self.mixstyle(x)
            x_augmented = torch.cat([x, x_mixed], dim=0) # 形状变为 (B=2, C, H, W)
            
            # 3. 复制原始掩码 m
            m_augmented = m.repeat(2, 1, 1) # 形状变为 (B=2, H_orig, W_orig)
        else:
            x_augmented = x
            m_augmented = m   
        mask = F.interpolate(m_augmented[None].float(), size=x_augmented.shape[-2:]).to(torch.bool)[0]
        out_tmp['0'] = NestedTensor(x_augmented, mask)

        xs = out_tmp
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

    def forward_supp_branch(self, tensor_list: NestedTensor,  return_interm_layers = False, VPT_enable = False,sam_masks=None):
        out_tmp: Dict[str, NestedTensor] = {}
        
        m = tensor_list.mask
        if VPT_enable:
            x = self[0](tensor_list.tensors)[self.vit_feat_name]
        else:
            with torch.no_grad(): x = self[0](tensor_list.tensors)[self.vit_feat_name]

        # conv_14_28 ↓
        x = self[2](x)
        # conv_14_28 ↑
        x_out =x 
        m_out =m
         # 对support image的特征也应用MixStyle
        if self.use_mixstyle and self.training:
            x_mixed = self.mixstyle(x)
            if sam_masks is not None:
                # x_orig: 原始特征 (Foreground Content)
                x_orig = x.clone()
                
                # 调整掩码大小以匹配特征图
                # sam_masks: [B, 1, 518, 518] -> Resize -> [B, 1, 37, 37]
                mask_resized = F.interpolate(sam_masks, size=x.shape[-2:], mode='nearest')
                
                # 融合: 前景保持 x_orig，背景使用 x_mixed
                # mask=1 (前景), mask=0 (背景)
                x_aug = x_orig * mask_resized + x_mixed * (1 - mask_resized)
            else:
                # 兜底：如果没有掩码，全图混合
                x_aug = x_mixed 
            x_out = torch.cat([x,x_aug],dim=0)
            m_out = torch.cat([m,m],dim=0)
        mask_final = F.interpolate(m_out[None].float(), size=x_out.shape[-2:]).to(torch.bool)[0]
        out_tmp['0'] = NestedTensor(x_out, mask_final)

        xs = out_tmp
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

def build_dino_v2_vit(args, input_shape, VPT_enable):
    # 指定在模型的哪些层输出特征图。如果为 None，则不输出中间层特征。
    out_indices = None

    if out_indices is not None:
        if isinstance(out_indices, str):
            out_indices = [int(m) for m in out_indices.split(",")]
    
    if args.dino_type == 'small':
        return DinoVisionTransformer(
        patch_size=14,
        img_size=518,
        init_values=1,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        out_indices=out_indices,
        VPT_enable=VPT_enable,
    )
    elif args.dino_type == 'base':
        return vit_base(out_indices=out_indices, VPT_enable=VPT_enable)
    elif args.dino_type == "large":
        return vit_large(img_size=518, patch_size=14, init_values=1, out_indices=out_indices, VPT_enable=VPT_enable)
    else:
        raise NotImplementedError()
    
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # 这个3应该没啥用
    backbone = build_dino_v2_vit(args, 3, VPT_enable=args.VPT_enable)
    # 禁止更新参数
    for p in backbone.parameters(): p.requires_grad = False
    backbone.eval()
    if args.VPT_enable:
        # 确保 prompt_dropout 和 prompt_embeddings 可训练
        backbone.prompt_dropout.requires_grad = True
        backbone.prompt_embeddings.requires_grad = True
        backbone.prompt_dropout.train()
        """
        nn.Parameter 自身并没有 train() 或 eval() 方法。这是因为 train() 和 eval() 是 nn.Module 类的方法，它们用于设置模块的训练模式或评估模式，主要影响的是 Dropout 和 BatchNorm 等层的行为。
        """
        # backbone.prompt_embeddings.train()
        missing_keys, unexpected_keys = backbone.load_state_dict(torch.load(args.dino_weight_path), strict=False)
    else:
        missing_keys, unexpected_keys = backbone.load_state_dict(torch.load(args.dino_weight_path))
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    """
    missing_keys 是检查点文件中缺失但在模型中需要的键（即模型参数）。
    unexpected_keys 是检查点文件中有的但模型中没有的键。
    """
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    # 让dino的14stride->28stride
    conv_14_28 = nn.Sequential(
        nn.Conv2d(backbone.num_channels[0], backbone.num_channels[0], kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, backbone.num_channels[0]),
        )
    # 按照detr中一样的初始化
    nn.init.xavier_uniform_(conv_14_28[0].weight, gain=1)
    nn.init.constant_(conv_14_28[0].bias, 0)
    model = Joiner(backbone, position_embedding, conv_14_28, 
                   use_mixstyle=args.use_mixstyle,
                   mixstyle_p=args.mixstyle_p,
                   mixstyle_alpha=args.mixstyle_alpha)
    return model