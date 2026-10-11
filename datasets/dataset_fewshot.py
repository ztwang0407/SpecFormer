from pathlib import Path
import os
import numpy as np
from PIL import Image
import torch
from datasets.dataset import DetectionDataset
import datasets.transforms as T
from util.misc import get_local_rank, get_local_size

# -----------------------------------------------------------------------------
# 【新增】 自定义数据集类，用于加载 SAM 掩码
# -----------------------------------------------------------------------------
class DetectionDatasetWithSAM(DetectionDataset):
    """
    继承自 DetectionDataset，重写 __getitem__ 以支持加载离线生成的 SAM 掩码。
    """
    def __getitem__(self, idx):
        # 1. 获取图片 ID 和 文件名
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        # 2. 加载图片 (逻辑与父类一致)
        img_path = Path(self.root) / file_name
        img = Image.open(img_path).convert('RGB')
        
        # 3. 加载标注 (逻辑与父类一致)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        target = {'image_id': img_id, 'annotations': anns}
        
        # 4. 预处理 (Poly -> Box/Mask)
        # 注意：self.prepare 会返回一个新的 target 字典，所以我们必须在它之后、transforms 之前注入掩码
        img, target = self.prepare(img, target)
        
        # -----------------------------------------------------------
        # 【核心修改】 加载 SAM 掩码
        # -----------------------------------------------------------
        # 假设掩码存放在数据集根目录同级的 sam2_masks 文件夹下
        # 例如: data/NEU/train -> data/NEU/sam2_masks
        dataset_root = os.path.dirname(str(self.root)) 
        mask_dir = os.path.join(dataset_root, "sam2_masks")
        
        # 构造掩码路径 (替换后缀为 .png)
        img_basename = os.path.basename(file_name)
        mask_name = os.path.splitext(img_basename)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            # 读取掩码并转为 Tensor [1, H, W]
            mask_pil = Image.open(mask_path).convert('L')
            mask_np = np.array(mask_pil)
            # 二值化 (确保是 0/1)
            mask_np = (mask_np > 128).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0) # [1, H, W]
            
            # 放入 target['masks']，让 transforms 自动处理翻转/裁剪/缩放
            target['masks'] = mask_tensor
        else:
            # 如果没找到掩码，生成全 1 掩码 (保护全图)
            w, h = img.size
            target['masks'] = torch.ones((1, h, w), dtype=torch.uint8)
        # -----------------------------------------------------------

        # 5. 数据增强 (Transforms)
        # 此时 target['masks'] 会跟随 img 一起进行 RandomFlip/Resize/Crop
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        # 6. 转存掩码键名
        # 为了配合 CDFormer.py 中的读取逻辑 (if 'sam_mask' in target)，我们需要重命名或复制
        if 'masks' in target:
            target['sam_mask'] = target['masks']
            
        return img, target

# -----------------------------------------------------------------------------
def make_transforms():
    """
    Transforms for query images during the few-shot fine-tuning stage.
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

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


def make_support_transforms():
    """
    Transforms for support images during the few-shot fine-tuning stage.
    For transforms for support images during the base training stage, please check dataset.py
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


def build(args, image_set, activated_class_ids, with_support=True):
    assert image_set == "fewshot"
    activated_class_ids.sort()

    if args.dataset_file in ['coco_base']:
        root = Path('data/coco_fewshot')
        img_folder = "/home/csy/datasets/mscoco/train2017"
        ann_file = root / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())

    if args.dataset_file == "voc_base1":
        root = Path('data/voc_fewshot_split1')
        img_folder = root.parent / 'voc' / "images"
        ann_file = root / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())

    if args.dataset_file == "voc_base2":
        root = Path('data/voc_fewshot_split2')
        img_folder = root.parent / 'voc' / "images"
        ann_file = root / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())

    if args.dataset_file == "voc_base3":
        root = Path('data/voc_fewshot_split3')
        img_folder = root.parent / 'voc' / "images"
        ann_file = root / f'seed{args.fewshot_seed}' / f'{args.num_shots}shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())
    # uodd
    if args.dataset_file == "uodd":
        root = Path('data/UODD')
        img_folder = root / "train"
        ann_file = root / f'annotations' / f'{args.num_shots}_shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())

    # deepfish
    if args.dataset_file == "deepfish":
        root = Path('data/FISH')
        img_folder = root / "train"
        ann_file = root / f'annotations' / f'{args.num_shots}_shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())
    # neu
    if args.dataset_file == "neu":
        root = Path('data/NEU-DET')
        img_folder = root / "train"
        ann_file = root / f'annotations' / f'{args.num_shots}_shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())
    # clipart
    if args.dataset_file == "clipart":
        root = Path('data/clipart1k')
        img_folder = root / "train"
        ann_file = root / f'annotations' / f'{args.num_shots}_shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())
    # artaxor
    if args.dataset_file == "artaxor":
        root = Path('data/ArTaxOr')
        img_folder = root / "train"
        ann_file = root / f'annotations' / f'{args.num_shots}_shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())
    # dior
    if args.dataset_file == "dior":
        root = Path('data/DIOR')
        img_folder = root / "train"
        ann_file = root / f'annotations' / f'{args.num_shots}_shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())
    # dataset1
    if args.dataset_file == "dataset1":
        root = Path('data/dataset1')
        img_folder = root / "train"
        ann_file = root / f'annotations' / f'{args.num_shots}_shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())
    # dataset2
    if args.dataset_file == "dataset2":
        root = Path('data/dataset2')
        img_folder = root / "train"
        ann_file = root / f'annotations' / f'{args.num_shots}_shot.json'
        return DetectionDataset(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())
    # dataset3
    if args.dataset_file == "dataset3":
        root = Path('data/dataset3')
        img_folder = root / "train"
        ann_file = root / f'annotations' / f'{args.num_shots}_shot.json'
        return DetectionDatasetWithSAM(args, img_folder, str(ann_file),
                                transforms=make_transforms(),
                                support_transforms=make_support_transforms(),
                                return_masks=False,
                                activated_class_ids=activated_class_ids,
                                with_support=with_support,
                                cache_mode=args.cache_mode,
                                local_rank=get_local_rank(),
                                local_size=get_local_size())


    raise ValueError
