import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import datasets
import util.misc as utils
from util.lr_scheduler import WarmupMultiStepLR
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.dataset_support import build_support_dataset
from engine import evaluate, train_one_epoch
from models import build_model

torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('CDFormer', add_help=False)

    # Basic Training and Inference Setting
    parser.add_argument('--lr', default=2e-4, type=float)
    # backbone.0在resnet中指的是resnet的参数,在dino中就是dino的参数，比较普适
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--embedding_related_names', default=['level_embed', 'query_embed'], type=str, nargs='+')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop_milestones', default=[45], type=int, nargs='+')
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--warmup_factor', default=0.1, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    # parser.add_argument('--resume', default='exps1/dino_coco_80_size_5_eposide/base_train/checkpoint0019.pth', help='resume from checkpoint, empty for training from scratch')
    parser.add_argument('--resume', default='', help='resume from checkpoint, empty for training from scratch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='only perform inference and evaluation')
    # parser.add_argument('--eval', default=True, action='store_true', help='only perform inference and evaluation')
    # 原本为10
    parser.add_argument('--eval_every_epoch', default=10, type=int, help='eval every ? epoch')
    parser.add_argument('--save_every_epoch', default=10, type=int, help='save model weights every ? epoch')

    # Few-shot Learning Setting
    parser.add_argument('--fewshot_finetune', default=False, action='store_true')
    parser.add_argument('--fewshot_seed', default=1, type=int)
    parser.add_argument('--num_shots', default=10, type=int)

    # Meta-Task Construction Settings
    parser.add_argument('--episode_num', default=5, type=int, help='The number of episode(s) for each iteration')
    parser.add_argument('--episode_size', default=5, type=int, help='The episode size')
    # 改成voc的20，我全都要用
    parser.add_argument('--total_num_support', default=15, type=int, help='used in training: each query image comes with ? support image(s)')
    parser.add_argument('--max_pos_support', default=10, type=int, help='used in training: each query image comes with at most ? positive support image(s)')

    # Model parameters
    # * Model Variant
    parser.add_argument('--with_box_refine', default=False, action='store_true')

    # * Backbone
    """
    resnet50
    如果要切换回resnet50记得更改nested_tensor_from_tensor_list,同时更改CDFormer中的build_backbone
    """
    parser.add_argument('--backbone', default='dinov2', type=str, help="Name of the ResNet backbone")
    parser.add_argument('--dilation', action='store_true', help="If true, ResNet backbone DC5 mode enabled")
    parser.add_argument('--freeze_backbone_at_layer', default=2, type=int, help='including the provided layer')
    parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels, 1 or 4')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate dim of the FC in transformer")
    parser.add_argument('--hidden_dim', default=256, type=int, help="dimension of transformer")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads for transformer")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="no aux loss @ each decoder layer")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2.0, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5.0, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2.0, type=float, help="GIoU box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1.0, type=float)
    parser.add_argument('--dice_loss_coef', default=1.0, type=float)
    parser.add_argument('--cls_loss_coef', default=2.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float)
    parser.add_argument('--giou_loss_coef', default=2.0, type=float)
    parser.add_argument('--category_codes_cls_loss_coef', default=5.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    # 基类训练:voc_base1  
    # 新类训练：uodd、deepfish、neu、clipart、artaxor、dior、dataset1
    parser.add_argument('--dataset_file', default='uodd')
    parser.add_argument('--remove_difficult', action='store_true')

    # Misc
    parser.add_argument('--output_dir', default='', help='path to where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing, only cuda supported')
    parser.add_argument('--seed', default=6666, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # dino相关
    """
    dino改进点:
    (1)data处理,需要为14的倍数,改进在nested_tensor_from_tensor_list中
    (2)dino特征提取器,改进在整个的dino_backbone.py中
    注意:dataset并没有改,体现在nested_tensor_from_tensor_list中
    (3)正向传播时要将self.backbone.backbone.eval(),因为在train时会.train()整个模型,体现在forward中
    (4)出现爆显存问题:
        1.compute_category_codes后加torch.cuda.empty_cache()
        2.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
    """
    # small base large
    parser.add_argument('--dino_type', default='large')
    parser.add_argument('--dino_weight_path', default='model_pt/dinov2/dinov2_vitl14_pretrain.pth')

    # category_codes_cls_loss相关
    """
    cam_all指的是要不要encoder的每一层都用cam,若为false,那只对layer0里加入cam
    multi_category_loss指的是要不要对每一层提取的类别特征用loss,若为false,那只对layer0提取的类别特征用loss
    那么有三种组合，
    (1)cam_all = true,  multi_category_loss = false, 每一层都有cam,但只对第0层的类别特征做loss
    (2)cam_all = true,  multi_category_loss = true,  每一层都有cam,并对每一层的类别特征都做loss
    (3)cam_all = false, multi_category_loss = false, 只有第一层都有cam,并对第0层的类别特征做loss(最基础的版本)
    (4)cam_all = false, multi_category_loss = true,  只有第一层都有cam,并对每一层的类别特征都做loss(这个感觉没必要)
    """
    parser.add_argument('--category_codes_cls_loss', action='store_true', help="if set, enable category codes cls loss")
    # parser.add_argument('--category_codes_cls_loss', default=False, action='store_true', help="if set, enable category codes cls loss")
    parser.add_argument('--cam_all', default=False)
    # distLinear必须当multi_category_loss为True时才能为True，因为这个参数相当于对每一层提取的类别特征构建一个特征空间
    parser.add_argument('--multi_category_loss', default=False)
    parser.add_argument('--all_distLinear', default=False)
    
    # VPT相关
    parser.add_argument('--VPT_enable', default=False)
    parser.add_argument('--VPT_method', default='shallow')

    # 微调特征空间保留相关,只有当微调时才能生效
    parser.add_argument('--feature_space_enable', default=False)
    # coco/voc
    parser.add_argument('--base_dataset_name', default='coco')
    """
    uodd:4 deepfish:2 neu:7 artaxor:8 clipart:21 dior:21
    """
    parser.add_argument('--novel_class_num_hh', default=7, type=int)
    #
    parser.add_argument('--frozen_base_feature_enable', default=False)
    parser.add_argument('--frozen_name', default=['category_codes_cls.L.weight_g', 'category_codes_cls.L.weight_v'], type=str, nargs='+')

    # MixStyle parameters for Domain Adaptation
    parser.add_argument('--use_mixstyle', action='store_true', 
                        help="Whether to use MixStyle for domain adaptation")
    parser.add_argument('--mixstyle_p', default=0.5, type=float, 
                        help='Probability of applying MixStyle')
    parser.add_argument('--mixstyle_alpha', default=0.1, type=float, 
                        help='Alpha parameter for the Beta distribution in MixStyle')
    # 新增的参数，用于控制只训练MixStyle
    parser.add_argument('--train_only_mixstyle', action='store_true',
                        help='Freeze all model weights except for the new MixStyle module')
    return parser


def main(args):
    # 打印出所有参数
    print("Training configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
        
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    import logging
    import os

    rank = int(os.environ.get('RANK', 0))
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - Rank {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"debug_rank_{rank}.log"),
            logging.StreamHandler()
        ]
    )

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)
    # from thop import profile

    # inputs = torch.randn(1, 3, 640, 640)  # 模拟输入
    # flops, params = profile(model, inputs=(inputs,inputs))
    # print(f"FLOPs: {flops}, Params: {params}")
    # 加载预训练权重
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

    image_set = 'fewshot' if args.fewshot_finetune else 'train'
    dataset_train = build_dataset(image_set=image_set, args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_support = build_support_dataset(image_set=image_set, args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            sampler_support = samplers.NodeDistributedSampler(dataset_support)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
            sampler_support = samplers.DistributedSampler(dataset_support)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_support = torch.utils.data.RandomSampler(dataset_support)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)

    loader_train = DataLoader(dataset_train,
                              batch_sampler=batch_sampler_train,
                              collate_fn=utils.collate_fn,
                              num_workers=args.num_workers,
                              pin_memory=True)

    loader_val = DataLoader(dataset_val,
                            batch_size=args.batch_size,
                            sampler=sampler_val,
                            drop_last=False,
                            collate_fn=utils.collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=True)

    loader_support = DataLoader(dataset_support,
                                batch_size=1,
                                sampler=sampler_support,
                                drop_last=False,
                                num_workers=args.num_workers,
                                pin_memory=False)
    # for i, batch in enumerate(loader_support):
    #     print(f"thie is epoch {i}")
    #     # print(batch)  # 看看能否正常输出每个批次的数据

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    if not args.fewshot_finetune:
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                     if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
                "initial_lr": args.lr,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
                "initial_lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
                "initial_lr": args.lr * args.lr_linear_proj_mult,
            }
        ]
    else:
        # For few-shot finetune stage, do not train sampling offsets, reference points, and embedding related parameters
        param_dicts = [
                {
                    "params":
                        [p for n, p in model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names) and \
                            not match_name_keywords(n, args.lr_linear_proj_names) and \
                            not match_name_keywords(n, args.embedding_related_names) and p.requires_grad],
                    "lr": args.lr,
                    "initial_lr": args.lr,
                },
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                    "lr": args.lr_backbone,
                    "initial_lr": args.lr_backbone,
                },
            ]
        # if not args.frozen_base_feature_enable:
        #     param_dicts = [
        #         {
        #             "params":
        #                 [p for n, p in model_without_ddp.named_parameters()
        #                 if not match_name_keywords(n, args.lr_backbone_names) and \
        #                     not match_name_keywords(n, args.lr_linear_proj_names) and \
        #                     not match_name_keywords(n, args.embedding_related_names) and p.requires_grad],
        #             "lr": args.lr,
        #             "initial_lr": args.lr,
        #         },
        #         {
        #             "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
        #             "lr": args.lr_backbone,
        #             "initial_lr": args.lr_backbone,
        #         },
        #     ]
        # elif args.frozen_base_feature_enable:
        #     param_dicts = [
        #         {
        #             "params":
        #                 [p for n, p in model_without_ddp.named_parameters()
        #                 if  not match_name_keywords(n, args.lr_backbone_names) and \
        #                     not match_name_keywords(n, args.lr_linear_proj_names) and \
        #                     not match_name_keywords(n, args.embedding_related_names) and \
        #                     not match_name_keywords(n, args.frozen_name) and \
        #                     p.requires_grad],
        #             "lr": args.lr,
        #             "initial_lr": args.lr,
        #         },
        #         # 这里p.requires_grad都为true，因为model初始化还在下面，不过无所谓
        #         {
        #             "params":
        #                 [p[:args.novel_class_num_hh] for n, p in model_without_ddp.named_parameters()
        #                 if  match_name_keywords(n, args.frozen_name) and \
        #                     p.requires_grad],
        #             "lr": args.lr,
        #             "initial_lr": args.lr,
        #         },
        #         {
        #             "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
        #             "lr": args.lr_backbone,
        #             "initial_lr": args.lr_backbone,
        #         },
        #     ]


    optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)
    lr_scheduler = WarmupMultiStepLR(optimizer,
                                     args.lr_drop_milestones,
                                     gamma=0.1,
                                     warmup_epochs=args.warmup_epochs,
                                     warmup_factor=args.warmup_factor,
                                     warmup_method='linear',
                                     last_epoch=args.start_epoch - 1)

    # find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass
    # 这个提示确实意味着如果你确定在模型的前向传播过程中不存在未使用的参数，你可以安全地取消 find_unused_parameters=True 的设置，以避免不必要的计算开销。
    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # print(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.dataset.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        # 特征空间相关
        if args.feature_space_enable:
            old_weight_g = checkpoint['model']['category_codes_cls.L.weight_g']  # torch.Size([21, 1])
            old_weight_v = checkpoint['model']['category_codes_cls.L.weight_v']  # torch.Size([21, 256])
        #删除不匹配的键,微调时用（预训练时不用）
        del checkpoint['model']['category_codes_cls.L.weight_g']
        del checkpoint['model']['category_codes_cls.L.weight_v']

        if args.multi_category_loss and args.all_distLinear:
            del checkpoint['model']['category_codes_cls1.L.weight_g']
            del checkpoint['model']['category_codes_cls1.L.weight_v']
            del checkpoint['model']['category_codes_cls2.L.weight_g']
            del checkpoint['model']['category_codes_cls2.L.weight_v']
            del checkpoint['model']['category_codes_cls3.L.weight_g']
            del checkpoint['model']['category_codes_cls3.L.weight_v']
            del checkpoint['model']['category_codes_cls4.L.weight_g']
            del checkpoint['model']['category_codes_cls4.L.weight_v']
            del checkpoint['model']['category_codes_cls5.L.weight_g']
            del checkpoint['model']['category_codes_cls5.L.weight_v']
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        """
        missing_keys 是检查点文件中缺失但在模型中需要的键（即模型参数）。
        unexpected_keys 是检查点文件中有的但模型中没有的键。
        """
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
            
        # ############################ 更新后的冻结逻辑 ############################
        if args.train_only_mixstyle:
            print("="*40)
            print("Training mode: Training MixStyle and subsequent layers.")
            print("Freezing backbone and other unrelated parameters...")
            print("="*40)
            
            # 首先冻结所有参数
            for param in model_without_ddp.parameters():
                param.requires_grad = False
                
            # 需要解冻的模块关键字列表
            unfreeze_keywords = ['mixstyle', 'meta_decoder', 'class_embed', 'bbox_embed']
            
            # 根据关键字解冻相关参数
            for name, param in model_without_ddp.named_parameters():
                # 检查参数名是否包含任何一个需要解冻的关键字
                if any(keyword in name for keyword in unfreeze_keywords):
                    param.requires_grad = True
                    print(f"Unfrozen for training: {name}")
        # ####################################################################
    
        if args.fewshot_finetune:
            if args.feature_space_enable:
                # 1. 提取新模型的参数
                new_weight_g = model_without_ddp.category_codes_cls.L.weight_g.data # torch.Size([26, 1]),假设新类空间为6(5+1)
                new_weight_v = model_without_ddp.category_codes_cls.L.weight_v.data # torch.Size([26, 256]) 
                # 2. 根据base数据集决定使用旧模型的哪些参数
                from datasets import (coco_base_class_ids, coco_novel_class_ids, voc_base1_class_ids, voc_novel1_class_ids)
                if args.base_dataset_name == 'coco':
                    indices = coco_base_class_ids + coco_novel_class_ids
                elif args.base_dataset_name == 'voc':
                    indices = voc_base1_class_ids + voc_novel1_class_ids
                indices.sort()
                # 3. 初始化新模型的前 6 个参数使用随机初始化（已默认初始化），后20/80个参数用旧模型的参数(old_weight_g有21or91个)
                new_weight_g[args.novel_class_num_hh:] = old_weight_g[indices]
                new_weight_v[args.novel_class_num_hh:] = old_weight_v[indices]
                # 4. 将更新后的参数赋值回新模型
                model_without_ddp.category_codes_cls.L.weight_g.data = new_weight_g
                model_without_ddp.category_codes_cls.L.weight_v.data = new_weight_v
                # 5. 检查参数是否相等
                # print("Old weight_g: ", old_weight_g[indices])
                # print("New weight_g (imported): ", model_without_ddp.category_codes_cls.L.weight_g.data[args.novel_class_num_hh:])
                # print(torch.allclose(old_weight_g[indices].cuda(), model_without_ddp.category_codes_cls.L.weight_g.data[args.novel_class_num_hh:], atol=1e-6))
                # print("Old weight_v: ", old_weight_v[indices])
                # print("New weight_v (imported): ", model_without_ddp.category_codes_cls.L.weight_v.data[args.novel_class_num_hh:])
                # print(torch.allclose(old_weight_v[indices].cuda(), model_without_ddp.category_codes_cls.L.weight_v.data[args.novel_class_num_hh:]))
                if args.frozen_base_feature_enable:
                    with torch.no_grad():
                        # 设置后 20/80 个参数的 requires_grad 为 False
                        model_without_ddp.category_codes_cls.L.weight_g[args.novel_class_num_hh:].requires_grad = False
                        model_without_ddp.category_codes_cls.L.weight_v[args.novel_class_num_hh:].requires_grad = False


            if args.category_codes_cls_loss:
                # Re-init weights of novel categories for few-shot finetune
                novel_class_ids = datasets.get_class_ids(args.dataset_file, type='novel')
                if args.num_feature_levels == 1:
                    # 对于新类的特征向量进行初始化，每一行都进行符合正态分布的初始化，base类保留
                    # 因为该特征向量是要与support类编码进行相似度计算以进行分类的
                    # 而在base类训练时该矩阵(nn.linear)没办法训练到新类行的特征，而微调时数据量又太少
                    # 所以需要手动先进行初始化
                    """
                    这种方法特别适用于那些需要特定初始化策略的情况，
                    如在处理新类别或在迁移学习场景中引入未曾训练过的类别。这样的初始化有助于确保这些新类别的权重在训练开始时具有合适的分布，可能有助于改善模型对这些类别的学习效果 
                    """
                    """
                    在我们的方法(跨域少样本)中,假如基类20类,新类三类
                    那么在微调初始化时,self.category_codes_cls = distLinear(self.hidden_dim, self.num_classes)相当于会重置
                    那么(1)missing_keys和unexpected_keys会有提示(2)模型初始化时为kaiming初始化,这里是normal_初始化
                    """
                    for novel_class_id in novel_class_ids:
                        nn.init.normal_(model_without_ddp.category_codes_cls.L.weight[novel_class_id])
                elif args.num_feature_levels > 1:
                    for classifier in model_without_ddp.category_codes_cls:
                        for novel_class_id in novel_class_ids:
                            nn.init.normal_(classifier.L.weight[novel_class_id])
                else:
                    raise RuntimeError

    if args.eval:
        # # Evaluate only base categories
        # test_stats, coco_evaluator = evaluate(
        #     args, model, criterion, postprocessors, loader_val, loader_support, base_ds, device, type='base'
        # )
        # if args.output_dir:
        #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval_base.pth")

        # Evaluate only novel categories
        test_stats, coco_evaluator = evaluate(
            args, model, criterion, postprocessors, loader_val, loader_support, base_ds, device, type='novel'
        )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval_novel.pth")

        return

    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, criterion, loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()

        # Saving Checkpoints after each epoch
        if args.output_dir and (not args.fewshot_finetune):
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Saving Checkpoints every args.save_every_epoch epoch(s)
        if args.output_dir:
            checkpoint_paths = []
            if (epoch + 1) % args.save_every_epoch == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Evaluation and Logging
        if (epoch + 1) % args.eval_every_epoch == 0:
            if 'base' in args.dataset_file:
                evaltype = 'base'
            else:
                evaltype = 'all'
            if args.fewshot_finetune:
                evaltype = 'novel'

            test_stats, coco_evaluator = evaluate(
                args, model, criterion, postprocessors, loader_val, loader_support, base_ds, device, type=evaltype
            )

        # if (epoch + 1) % args.epochs == 0:

        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                  **{f'test_{k}': v for k, v in test_stats.items()},
        #                  'epoch': epoch,
        #                  'n_parameters': n_parameters,
        #                  'evaltype': evaltype}

        #     if args.output_dir and utils.is_main_process():
        #         with (output_dir / "results.txt").open("a") as f:
        #             f.write(json.dumps(log_stats) + "\n")
        #         # for evaluation logs
        #         if coco_evaluator is not None:
        #             (output_dir / 'eval').mkdir(exist_ok=True)
        #             if "bbox" in coco_evaluator.coco_eval:
        #                 filenames = ['latest.pth']
        #                 filenames.append(f'{epoch:03}.pth')
        #                 for name in filenames:
        #                     torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CDFormer', parents=[get_args_parser()])
    args = parser.parse_args()
    assert args.max_pos_support <= args.total_num_support
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
