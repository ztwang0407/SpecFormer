import os
import random
from PIL import Image
import torch
import torch.utils.data
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import math
import warnings
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from .visual import visualize_predictions
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
from datasets import (coco_base_class_ids, coco_novel_class_ids, \
                      voc_base1_class_ids, voc_novel1_class_ids, \
                      voc_base2_class_ids, voc_novel2_class_ids, \
                      voc_base3_class_ids, voc_novel3_class_ids, \
                      uodd_class_ids, deepfish_class_ids,        \
                      neu_class_ids, clipart_class_ids,          \
                      artaxor_class_ids, dior_class_ids
                    )

#画图相关
import util.misc as utils
from datasets.eval_detection import DetectionEvaluator
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms

class DetectionDataset(TvCocoDetection):
    def __init__(self, args, img_folder, ann_file, transforms, support_transforms, return_masks, activated_class_ids,
                 with_support, cache_mode=False, local_rank=0, local_size=1):
        super(DetectionDataset, self).__init__(img_folder, ann_file, cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.with_support = with_support
        self.activated_class_ids = activated_class_ids
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ori_prepare = Ori_ConvertCocoPolysToMask()

    def __getitem__(self, idx):
        img, target = super(DetectionDataset, self).__getitem__(idx)
        target = [anno for anno in target if anno['category_id'] in self.activated_class_ids]
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        ori_img, ori_target = self.ori_prepare(img, target)
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self.with_support:
            support_images, support_class_ids, support_targets = self.sample_support_samples(target)
            return img, target, support_images, support_class_ids, support_targets
        else:
            return img, target, ori_img, ori_target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

class Ori_ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        return image, target
    

def make_transforms(image_set):
    """
    Transforms for query images.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomColorJitter(p=0.3333),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1152),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1152),
                ])
            ),
            normalize,
        ])
    # 在val时就不需要那么多数据增强了，resize就好(target也会变的)
    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1152),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_support_transforms():
    """
    Transforms for support images during the training phase.
    For transforms for support images during inference, please check dataset_support.py
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672]

    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomColorJitter(p=0.25),
        T.RandomSelect(
            T.RandomResize(scales, max_size=672),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=672),
            ])
        ),
        normalize,
    ])


def build(args, img_folder, ann_file, image_set, activated_class_ids, with_support):
    return DetectionDataset(args, img_folder, ann_file,
                            transforms=make_transforms(image_set),
                            support_transforms=make_support_transforms(),
                            return_masks=False,
                            activated_class_ids=activated_class_ids,
                            with_support=with_support,
                            cache_mode=args.cache_mode,
                            local_rank=get_local_rank(),
                            local_size=get_local_size())


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.5):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf  # 类别置信度
        self.iou_thres = iou_thres  # IoU置信度

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None or detections.size(0) == 0:  # 处理 detections 为空的情况
            gt_classes = labels[:, 0].int()  # 实际类别
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # 预测为背景,但实际为目标
            return

        # 过滤置信度低于阈值的检测结果
        detections = detections[detections[:, 4] > self.conf]
        if detections.size(0) == 0:  # 过滤后仍然为空
            gt_classes = labels[:, 0].int()  # 实际类别
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # 预测为背景,但实际为目标
            return

        gt_classes = labels[:, 0].int()  # 实际类别
        detection_classes = detections[:, 5].int()  # 预测类别
        iou = box_iou(labels[:, 1:], detections[:, :4])  # 计算所有结果的IoU

        x = torch.where(iou > self.iou_thres)  # 根据IoU匹配结果,返回满足条件的索引 x(dim0), (dim1)
        if x[0].shape[0]:
            # shape:[n, 3], 3->[label, detect, iou]
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]  # 根据IoU从大到小排序
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 若一个dt匹配多个gt,保留IoU最高的gt匹配结果
                matches = matches[matches[:, 2].argsort()[::-1]]  # 根据IoU从大到小排序
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 若一个gt匹配多个dt,保留IoU最高的dt匹配结果
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0  # 是否存在和gt匹配成功的dt
        m0, m1, _ = matches.transpose().astype(int)  # m0:gt索引 m1:dt索引
        for i, gc in enumerate(gt_classes):  # 实际的结果
            j = m0 == i  # 预测为该目标的预测结果序号
            if n and sum(j) == 1:  # 该实际结果预测成功
                self.matrix[detection_classes[m1[j]], gc] += 1  # 预测为目标,且实际为目标
            else:  # 该实际结果预测失败
                self.matrix[self.nc, gc] += 1  # 预测为背景,但实际为目标

        if n:
            for i, dc in enumerate(detection_classes):  # 对预测结果处理
                if not any(m1 == i):  # 若该预测结果没有和实际结果匹配
                    self.matrix[dc, self.nc] += 1  # 预测为目标,但实际为背景


    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn
        plt.rc('font', family='Times New Roman', size=15)
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = 0.00  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            h = sn.heatmap(array,
                           ax=ax,
                           annot=nc < 30,
                           annot_kws={
                               'size': 5},
                           cmap='Reds',
                           fmt='.2f',
                           linewidths=2,
                           square=True,
                           vmin=0.0,
                           xticklabels=ticklabels,
                           yticklabels=ticklabels,
                           )
            h.set_facecolor((1, 1, 1))

            cb = h.collections[0].colorbar  # 显示colorbar
            cb.ax.tick_params(labelsize=10)  # 设置colorbar刻度字体大小。

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.rcParams["font.sans-serif"] = ["SimSun"]
        plt.rcParams["axes.unicode_minus"] = False
        ax.set_xlabel('ground truth')
        ax.set_ylabel('prediction')
        ax.set_title('Confusion Matrix', fontsize=20)
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=100)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))

# VOC数据集的类别名称
clipart_names = [
    'sheep', 'chair', 'boat', 'bottle',
    'diningtable', 'sofa', 'cow', 'motorbike',
    'car', 'aeroplane', 'cat', 'train',
    'person', 'bicycle', 'pottedplant', 'bird',
    'dog', 'bus', 'tvmonitor', 'horse'
]

artaxor_names = [
    'Araneae', 'Coleoptera', 'Diptera', 'Hemiptera',
    'Hymenoptera', 'Lepidoptera', 'Odonata'
]

uodd_names = [
    'seacucumber', 'seaurchin', 'scallop'
]

dior_names = [
    'Expressway-Service-area','Expressway-toll-station','airplane',
    'airport','baseballfield','basketballcourt','bridge','chimney',
    'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 
    'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 
    'vehicle', 'windmill'
]

deepfish_names = [
    'fish'
]

neu_names = [
    'crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'
]
def xywh_to_xyxy(box):
    x, y, w, h = box
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]

def format_detections(results, confidence_threshold=0.1):
    formatted_detections = []
    res = results

    boxes = res['boxes']
    scores = res['scores']
    # 如果id不是从0开始的需要 - 1(也即除了uodd)
    labels = res['labels'] - 1
    # labels = res['labels']


    # Loop through the boxes and apply the confidence threshold
    for i in range(len(boxes)):
        conf = scores[i].item()
        
        # Filter out detections with confidence lower than the threshold
        if conf >= confidence_threshold:
            box = boxes[i]
            cls = labels[i].item()
            formatted_detections.append([*box, conf, cls])
    
    return torch.tensor(formatted_detections)

def format_labels(ori_target):
    formatted_labels = []
    tgt = ori_target
    # for tgt in ori_target:
    boxes = tgt['boxes']
    # 如果id不是从0开始的需要 - 1(也即除了uodd)
    labels = tgt['labels'] - 1
    # labels = tgt['labels']
    
    # Convert each box from [x, y, w, h] to [x1, y1, x2, y2] and concatenate with class
    for i in range(len(boxes)):
        cls = labels[i]
        # if cls == 0:
        #     print("ship")
        box = xywh_to_xyxy(boxes[i])
        formatted_labels.append([cls.item(), *box])
    
    return torch.tensor(formatted_labels)


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
        else:
            raise ValueError
    else:
        raise ValueError("Type must be 'all', 'base' or 'novel'!")

    print_freq = 50
    if args.dataset_file == 'artaxor':
        num_classes = 7
    elif args.dataset_file == 'uodd':
        num_classes = 3
    elif args.dataset_file == 'deepfish':
        num_classes = 1
    elif args.dataset_file == 'neu':
        num_classes = 6
    else:
        num_classes = 20

    conf_threshold = 0.2
    iou_threshold = 0.5

    # 创建混淆矩阵对象
    cm = ConfusionMatrix(nc=num_classes, conf=conf_threshold, iou_thres=iou_threshold)
    # 创建保存文件夹
    folder_path = 'visual/vis_result/uodd'
    # 选择要画的目标框
    draw_gt = False
    draw_pre = True
    if not os.path.exists(folder_path):
        # 如果不存在，则创建文件夹
        os.makedirs(folder_path)

    for samples, targets, ori_image, ori_target in metric_logger.log_every(dataloader, print_freq, header):

        samples = samples.to(device)
        images = samples.tensors
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 开始计时
        start_time = time.time()
        outputs = model(samples, targets=targets, supp_class_ids=support_class_ids_final, category_codes=all_category_codes_final)
        # 结束计时
        end_time = time.time()
        # 计算并输出推理时间
        inference_time = end_time - start_time
        # print(f"Inference time: {inference_time:.6f} seconds")
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
        # 混淆矩阵
        for img, ori_target_single, result_single in zip(ori_image, ori_target, results):
            formatted_detections = format_detections(result_single)
            formatted_labels = format_labels(ori_target_single)
            cm.process_batch(formatted_detections, formatted_labels)

        # 可视化每张图片的预测结果
        for img, target, result in zip(ori_image, ori_target, results):
            predictions = result
            visualize_predictions(img, target, predictions, folder_path, draw_gt=draw_gt, draw_pre=draw_pre)

    # 混淆矩阵绘制
    # 找出TP和FP
    tp, fp = cm.tp_fp()

    # 打印混淆矩阵
    cm.print()

    # 绘制混淆矩阵
    # 假设类别名称是 ['class0', 'class1', 'class2']
    cm.plot(normalize=True, save_dir='./', names=uodd_names)


    #
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
