import sys
import math
import random
import itertools
import copy
import gc
import logging
from typing import Iterable
import numpy as np

import torch
import util.misc as utils
from datasets.eval_detection import DetectionEvaluator
from datasets import (coco_base_class_ids, coco_novel_class_ids, \
                      voc_base1_class_ids, voc_novel1_class_ids, \
                      voc_base2_class_ids, voc_novel2_class_ids, \
                      voc_base3_class_ids, voc_novel3_class_ids, \
                      uodd_class_ids, deepfish_class_ids, \
                      neu_class_ids,  clipart_class_ids,  \
                      artaxor_class_ids, dior_class_ids,  \
                      dataset1_class_ids, dataset2_class_ids,  \
                      dataset3_class_ids
                    )


@torch.no_grad()
def sample_support_categories(args, targets, support_images, support_class_ids, support_targets):
    """
    This function is used during training. It does the followings:
    1. Samples the support categories (total num: args.total_num_support; maximum positive num: args.max_pos_support)
       (Insufficient positive support categories will be replaced with negative support categories.)
    2. Filters ground truths of the query images.
       We only keep ground truths whose labels are sampled as support categories.
    3. Samples and pre-processes support_images, support_class_ids, and support_targets.
    """
    support_images = list(itertools.chain(*support_images))
    support_class_ids = torch.cat(support_class_ids, dim=0).tolist()
    support_targets = list(itertools.chain(*support_targets))

    positive_labels = torch.cat([target['labels'] for target in targets], dim=0).unique()
    num_positive_labels = positive_labels.shape[0]
    positive_labels_list = positive_labels.tolist()
    negative_labels_list = list(set(support_class_ids) - set(positive_labels_list))
    num_negative_labels = len(negative_labels_list)

    positive_label_indexes = [i for i in list(range(len(support_images))) if support_class_ids[i] in positive_labels_list]
    negative_label_indexes = [i for i in list(range(len(support_images))) if support_class_ids[i] in negative_labels_list]

    meta_support_images, meta_support_class_ids, meta_support_targets = list(), list(), list()
    for _ in range(args.episode_num):
        """
        这里的策略是因为原本的代码没有背景类,所以其下限要在max(0, args.episode_size - num_negative_labels)之间确定
        但是在类别多的时候，args.episode_size - num_negative_labels一定是<0的
        但现在我们微调时如果就三类，那么会出现num_negative_labels = 0的情况，就会报错
        """
        # NUM_POS = random.randint(max(0, args.episode_size - num_negative_labels),
        #                          min(num_positive_labels, args.episode_size))
        """
        那么问题又来了，可能会出现NUM_POS=0并且num_negative_labels=0的情况(比如三新类且num_positive_labels=3)
        此时就会出现support全为空的情况，所以这里我们将0->1
        也即至少保证有一个正例
        """
        # NUM_POS = random.randint(0, min(num_positive_labels, args.episode_size))
        """
        那么问题又来了，如果min(num_positive_labels, args.episode_size) = 1，会报错
        """
        # NUM_POS = random.randint(1, min(num_positive_labels, args.episode_size))
        # 同样，这里也有问题，比如在uodd数据集中，若batch=2，并且取到了003719.jpg和002117.jpg
        # num_positive_labels会等于0，这里的问题出现在dataset_fewshot中的数据增强
        '''
        # 其用了T.RandomSizeCrop(384, 600),这导致了目标的裁剪,故先去掉(在dataset_fewshot中)
        但目前又不用去了，因为有了if num_positive_labels == 0判断条件
        '''
        # print(f"num_positive_labels: {num_positive_labels}")
        if num_positive_labels == 0:
            NUM_POS = 0
        elif min(num_positive_labels, args.episode_size) == 1:
            NUM_POS = 1
        else:
            NUM_POS = random.randint(1, min(num_positive_labels, args.episode_size))
        NUM_NEG = args.episode_size - NUM_POS

        # Sample positive support classes: make sure in every episode, there is no repeated category
        while True:
            pos_support_indexes = random.sample(positive_label_indexes, NUM_POS)
            if NUM_POS == len(set([support_class_ids[i] for i in pos_support_indexes])):
                break

        # Sample negative support classes: try our best to ensure in every episode there is no repeated category
        # 上面正样本的提取一定不会有问题，因为用了min(num_positive_labels, args.episode_size)来限制
        # 这里负样本的提取就需要注意了
        num_trial = 0
        """
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        while num_trial < 50:
        """
        while num_trial < 50:
            # 如果需要的负样本类别小于所拥有的负样本类别，那么就可以直接提取NUM_NEG个
            if NUM_NEG <= num_negative_labels:
                neg_support_indexes = random.sample(negative_label_indexes, NUM_NEG)
            # 反之只提取num_negative_labels个，剩下的用bg类补全
            else:
                neg_support_indexes = random.sample(negative_label_indexes, num_negative_labels)
            if NUM_NEG == len(set([support_class_ids[i] for i in neg_support_indexes])):
                break
            else:
                num_trial += 1
       
        """
        这里原本索引是100,但由于batch=8索引最大到160
        所以这里类别号仍为100,但索引改为1000
        """
        if NUM_NEG > 0:
            # 如果需要的负样本数量大于了已经提取的index，那么需要加入bg类来达到NUM_NEG的要求
            if NUM_NEG > len(neg_support_indexes):
                for i in range(NUM_NEG - len(neg_support_indexes)):
                    neg_support_indexes.append(1000)
            else:
                # 从0到NUM_NEG-1中任意取一个数a(-1是因为如果NUM_NEG=5，那么a有可能是5，这样就寄了)
                a = random.randint(0, NUM_NEG - 1)
                # 从列表中任意取a个不同的索引
                indexes_to_update = random.sample(range(NUM_NEG), a)
                # 将取到的索引处的值赋值为1000
                for index in indexes_to_update:
                    neg_support_indexes[index] = 1000

        support_indexes = pos_support_indexes + neg_support_indexes
        random.shuffle(support_indexes)

        selected_support_images = [support_images[i] for i in support_indexes if i != 1000]
        selected_support_class_ids = [support_class_ids[i] if i != 1000 else 100 for i in support_indexes]
        selected_support_targets = [support_targets[i] for i in support_indexes if i != 1000]

        # if 100 in selected_support_class_ids:
        #     raise ValueError("Debug: Variable x is 0, stopping program for debugging.")

        meta_support_images += selected_support_images
        meta_support_class_ids += selected_support_class_ids
        meta_support_targets += selected_support_targets

    """
    在这里选择挑选完后调整为大小一致的batch
    """
    meta_support_images = utils.nested_tensor_from_tensor_list(meta_support_images)
    meta_support_class_ids = torch.tensor(meta_support_class_ids)

    return targets, meta_support_images, meta_support_class_ids, meta_support_targets


def train_one_epoch(args,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    dataloader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, targets, support_images, support_class_ids, support_targets in metric_logger.log_every(dataloader, print_freq, header):

        # * Sample Support Categories;
        # * Filters Targets (only keep GTs within support categories);
        # * Samples Support Images and Targets
        """
        这里虽然是tensor,但还是在cpu上的,防止OOM
        下面在放到GPU的显存上
        """
        targets, support_images, support_class_ids, support_targets = \
            sample_support_categories(args, targets, support_images, support_class_ids, support_targets)

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        support_images = support_images.to(device)
        support_class_ids = support_class_ids.to(device)
        support_targets = [{k: v.to(device) for k, v in t.items()} for t in support_targets]

        outputs = model(samples, targets=targets, supp_samples=support_images, supp_class_ids=support_class_ids, supp_targets=support_targets)
        loss_dict = criterion(outputs)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is NaN - {}. \nTraining terminated unexpectedly.\n".format(loss_value))
            print("loss dict:")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # for i in range(torch.cuda.device_count()):
        #     allocated = torch.cuda.memory_allocated(i) / 1024 ** 2  # 实际使用的显存
        #     reserved = torch.cuda.memory_reserved(i) / 1024 ** 2    # 预留的显存（包括缓存）
        #     logging.info(f"GPU {i} Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        #     # print(f"GPU {i} Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    del support_images
    del support_class_ids
    del support_targets
    del samples
    del targets
    del outputs
    del weight_dict
    del grad_total_norm
    del loss_value
    del losses
    del loss_dict
    del loss_dict_reduced
    del loss_dict_reduced_scaled
    del loss_dict_reduced_unscaled
    torch.cuda.empty_cache()
    gc.collect()


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, model, criterion, postprocessors, dataloader, support_data_loader, base_ds, device, type='all'):
    model.eval()
    criterion.eval()

    # First: Obtain Category Codes for All Categories to Detect
    support_iter = iter(support_data_loader)
    all_category_codes_final = []
    print("Extracting support category codes...")
    number_of_supports = 100  # This is the number of support images to use for each category. Need be large enough.
    for i in range(number_of_supports):
        try:
            support_images, support_class_ids, support_targets = next(support_iter)
        except:
            support_iter = iter(support_data_loader)
            support_images, support_class_ids, support_targets = next(support_iter)
        support_images = [support_image.squeeze(0) for support_image in support_images]
        support_class_ids = support_class_ids.squeeze(0).to(device)
        support_targets = [{k: v.squeeze(0) for k, v in t.items()} for t in support_targets]
        num_classes = support_class_ids.shape[0]
        num_episode = math.ceil(num_classes / args.episode_size)
        category_codes_final = []
        support_class_ids_final = []
        for i in range(num_episode):
            if (args.episode_size * (i + 1)) <= num_classes:
                support_images_ = utils.nested_tensor_from_tensor_list(
                    support_images[(args.episode_size * i): (args.episode_size * (i + 1))]
                ).to(device)
                support_targets_ = [
                    {k: v.to(device) for k, v in t.items()} for t in support_targets[(args.episode_size * i): (args.episode_size * (i + 1))]
                ]
                support_class_ids_ = support_class_ids[(args.episode_size * i): (args.episode_size* (i + 1))]
            else:
                support_images_ = utils.nested_tensor_from_tensor_list(
                    support_images[-args.episode_size:]
                ).to(device)
                support_targets_ = [
                    {k: v.to(device) for k, v in t.items()} for t in support_targets[-args.episode_size:]
                ]
                support_class_ids_ = support_class_ids[-args.episode_size:]
                # 加上背景类
                # 如要添加bg
                new_value = torch.tensor([100], device=support_class_ids_.device)
                for i in range(args.episode_size - len(support_class_ids_)):
                    support_class_ids_ = torch.cat((support_class_ids_, new_value))
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                category_code = model.module.compute_category_codes(support_images_, support_targets_, support_class_ids_)
            else:
                category_code = model.compute_category_codes(support_images_, support_targets_, support_class_ids_)
            category_code = torch.stack(category_code, dim=0)   # (num_enc_layer, args.total_num_support, d)
            category_codes_final.append(category_code)
            support_class_ids_final.append(support_class_ids_)
        support_class_ids_final = torch.cat(support_class_ids_final, dim=0)
        category_codes_final = torch.cat(category_codes_final, dim=1)  # (num_enc_layer, num_episode x args.total_num_support, d)
        all_category_codes_final.append(category_codes_final)

    if args.num_feature_levels == 1:
        all_category_codes_final = torch.stack(all_category_codes_final, dim=0)  # (number_of_supports, num_enc_layer, num_episode x args.total_num_support, d)
        all_category_codes_final = torch.mean(all_category_codes_final, 0, keepdims=False)
        all_category_codes_final = list(torch.unbind(all_category_codes_final, dim=0))
    elif args.num_feature_levels == 4:
        raise NotImplementedError
    else:
        raise NotImplementedError
    print("Completed extracting category codes. Start Inference...")

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('bbox',) if k in postprocessors.keys())
    evaluator = DetectionEvaluator(base_ds, iou_types)
    if type == 'all':
        pass  # To evaluate all categories, no need to change params of the evaluator
    elif type == 'base':
        if args.dataset_file == 'coco_base':
            evaluator.coco_eval['bbox'].params.catIds = coco_base_class_ids
        elif args.dataset_file == 'voc_base1':
            evaluator.coco_eval['bbox'].params.catIds = voc_base1_class_ids
        elif args.dataset_file == 'voc_base2':
            evaluator.coco_eval['bbox'].params.catIds = voc_base2_class_ids
        elif args.dataset_file == 'voc_base3':
            evaluator.coco_eval['bbox'].params.catIds = voc_base3_class_ids
        else:
            raise ValueError
    elif type == 'novel':
        if args.dataset_file == 'coco_base' or args.dataset_file == 'coco':
            evaluator.coco_eval['bbox'].params.catIds = coco_novel_class_ids
        elif args.dataset_file == 'voc_base1':
            evaluator.coco_eval['bbox'].params.catIds = voc_novel1_class_ids
        elif args.dataset_file == 'voc_base2':
            evaluator.coco_eval['bbox'].params.catIds = voc_novel2_class_ids
        elif args.dataset_file == 'voc_base3':
            evaluator.coco_eval['bbox'].params.catIds = voc_novel3_class_ids
        elif args.dataset_file == 'uodd':
            evaluator.coco_eval['bbox'].params.catIds = uodd_class_ids
        elif args.dataset_file == 'deepfish':
            evaluator.coco_eval['bbox'].params.catIds = deepfish_class_ids
        elif args.dataset_file == 'neu':
            evaluator.coco_eval['bbox'].params.catIds = neu_class_ids
        elif args.dataset_file == 'clipart':
            evaluator.coco_eval['bbox'].params.catIds = clipart_class_ids
        elif args.dataset_file == 'artaxor':
            evaluator.coco_eval['bbox'].params.catIds = artaxor_class_ids
        elif args.dataset_file == 'dior':
            evaluator.coco_eval['bbox'].params.catIds = dior_class_ids
        elif args.dataset_file == 'dataset1':
            evaluator.coco_eval['bbox'].params.catIds = dataset1_class_ids
        elif args.dataset_file == 'dataset2':
            evaluator.coco_eval['bbox'].params.catIds = dataset2_class_ids
        elif args.dataset_file == 'dataset3':
            evaluator.coco_eval['bbox'].params.catIds = dataset3_class_ids
        else:
            raise ValueError
    else:
        raise ValueError("Type must be 'all', 'base' or 'novel'!")

    print_freq = 50

    for samples, targets in metric_logger.log_every(dataloader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, targets=targets, supp_class_ids=support_class_ids_final, category_codes=all_category_codes_final)
        loss_dict = criterion(outputs)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if evaluator is not None:
            evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if evaluator is not None:
        evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if evaluator is not None:
        if type == 'all':
            print("\n\n\n\n * ALL Categories:")
        elif type == 'base':
            print("\n\n\n\n * Base Categories:")
        elif type == 'novel':
            print("\n\n\n\n * Novel Categories:")
        else:
            raise ValueError("Type must be 'all', 'base' or 'novel'!")
        evaluator.accumulate()
        evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = evaluator.coco_eval['bbox'].stats.tolist()

    del support_images
    del support_class_ids
    del support_targets
    del samples
    del targets
    del outputs
    del weight_dict
    del loss_dict
    del loss_dict_reduced
    del loss_dict_reduced_scaled
    del loss_dict_reduced_unscaled
    del category_code
    del category_codes_final
    del all_category_codes_final
    del orig_target_sizes
    del res
    del results
    torch.cuda.empty_cache()

    return stats, evaluator