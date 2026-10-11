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
from models import build_model
from datasets.dataset_inference_val import build, evaluate
from datasets import (coco_base_class_ids, coco_novel_class_ids, \
                      voc_base1_class_ids, voc_novel1_class_ids, \
                      voc_base2_class_ids, voc_novel2_class_ids, \
                      voc_base3_class_ids, voc_novel3_class_ids, \
                      uodd_class_ids, deepfish_class_ids,        \
                      neu_class_ids, clipart_class_ids,          \
                      artaxor_class_ids, dior_class_ids
                    )



torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('CDFormer', add_help=False)

    # Basic Training and Inference Setting
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    #FINAL_save_pt/uodd/vitl_0049_test/seed01_10shot_01/checkpoint0049.pth
    parser.add_argument('--resume', default='exps/uodd/epoch50/seed01_10shot_01/checkpoint0049.pth', help='resume from checkpoint, empty for training from scratch')
    parser.add_argument('--eval', default=True, action='store_true', help='only perform inference and evaluation')

    # Meta-Task Construction Settings
    parser.add_argument('--episode_num', default=5, type=int, help='The number of episode(s) for each iteration')
    parser.add_argument('--episode_size', default=5, type=int, help='The episode size')
    parser.add_argument('--total_num_support', default=15, type=int, help='used in training: each query image comes with ? support image(s)')
    parser.add_argument('--max_pos_support', default=10, type=int, help='used in training: each query image comes with at most ? positive support image(s)')

    # * Backbone
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
    parser.add_argument('--category_codes_cls_loss', default=True, action='store_true', help="if set, enable category codes cls loss")

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
    parser.add_argument('--dataset_file', default='uodd')
    parser.add_argument('--remove_difficult', action='store_true')

    # Misc
    parser.add_argument('--output_dir', default='visual', help='path to where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing, only cuda supported')
    parser.add_argument('--seed', default=6666, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # * Model Variant
    parser.add_argument('--with_box_refine', default=False, action='store_true')

    # Few-shot Learning Setting
    parser.add_argument('--fewshot_finetune', default=True, action='store_true')
    parser.add_argument('--fewshot_seed', default=1, type=int)
    parser.add_argument('--num_shots', default=10, type=int)

    # dino相关
    # small base large
    parser.add_argument('--dino_type', default='large')
    parser.add_argument('--dino_weight_path', default='model_pt/dinov2/dinov2_vitl14_pretrain.pth')

    # category_codes_cls_loss相关
    # parser.add_argument('--category_codes_cls_loss', action='store_true', help="if set, enable category codes cls loss")
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
    parser.add_argument('--novel_class_num_hh', default=4, type=int)
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
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    image_set = 'val'
    if args.dataset_file in ['coco', 'coco_base']:
        root = Path('data/coco')
        img_folder = root / "val2017"
        ann_file = root / "annotations" / 'instances_val2017.json'
        class_ids = coco_base_class_ids + coco_novel_class_ids
        class_ids.sort()
        dataset_val = build(args, img_folder, ann_file, image_set, activated_class_ids=class_ids, with_support=False)
    if args.dataset_file in ['voc', 'voc_base1', 'voc_base2', 'voc_base3']:
        root = Path('data/voc')
        img_folder = root / "images"
        ann_file = root / "annotations" / 'pascal_test2007.json'
        dataset_val = build(args, img_folder, ann_file, image_set, activated_class_ids=list(range(1, 20+1)), with_support=False)
    if args.dataset_file in ['uodd']:
        root = Path('data/UODD')
        img_folder = root / "test"
        ann_file = root / f'annotations' / f'test.json'
        dataset_val = build(args, img_folder, ann_file, image_set, activated_class_ids=uodd_class_ids, with_support=False)
    if args.dataset_file in ['deepfish']:
        root = Path('data/FISH')
        img_folder = root / "test"
        ann_file = root / f'annotations' / f'test.json'
        dataset_val = build(args, img_folder, ann_file, image_set, activated_class_ids=deepfish_class_ids, with_support=False)
    if args.dataset_file in ['neu']:
        root = Path('data/NEU-DET')
        img_folder = root / "test"
        ann_file = root / f'annotations' / f'test.json'
        dataset_val = build(args, img_folder, ann_file, image_set, activated_class_ids=neu_class_ids, with_support=False)
    if args.dataset_file in ['clipart']:
        root = Path('data/clipart1k')
        img_folder = root / "test"
        ann_file = root / f'annotations' / f'test.json'
        dataset_val = build(args, img_folder, ann_file, image_set, activated_class_ids=clipart_class_ids, with_support=False)
    if args.dataset_file in ['artaxor']:
        root = Path('data/ArTaxOr')
        img_folder = root / "test"
        ann_file = root / f'annotations' / f'test.json'
        dataset_val = build(args, img_folder, ann_file, image_set, activated_class_ids=artaxor_class_ids, with_support=False)
    if args.dataset_file in ['dior']:
        root = Path('data/DIOR')
        img_folder = root / "test"
        ann_file = root / f'annotations' / f'test.json'
        dataset_val = build(args, img_folder, ann_file, image_set, activated_class_ids=dior_class_ids, with_support=False)

    image_set = 'fewshot' 
    dataset_support = build_support_dataset(image_set=image_set, args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            sampler_support = samplers.NodeDistributedSampler(dataset_support)
        else:
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
            sampler_support = samplers.DistributedSampler(dataset_support)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_support = torch.utils.data.RandomSampler(dataset_support)

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

    for n, p in model_without_ddp.named_parameters():
        print(n)

    if args.distributed:
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
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    if args.eval:
        # # Evaluate only base categories
        # test_stats, coco_evaluator, _ = evaluate(
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CDFormer', parents=[get_args_parser()])
    args = parser.parse_args()
    assert args.max_pos_support <= args.total_num_support
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
