#画图相关
import util.misc as utils
from datasets.eval_detection import DetectionEvaluator
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision.transforms
import os
import random
from PIL import Image
import matplotlib.colors as mcolors
import matplotlib.colors as mplc
import colorsys

def change_color_brightness(color, brightness_factor = -0.7):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
                0 will correspond to no change, a factor in [-1.0, 0) range will result in
                a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color (tuple[double]): a tuple containing the RGB values of the
                modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

def draw_text(ax, text, position, font_size=None, color="g", horizontal_alignment="left", rotation=0):
    """
    在给定的Axes上绘制文本。

    Args:
        ax: 要绘制的matplotlib Axes对象。
        text (str): 要绘制的文本。
        position (tuple): 文本位置的x和y坐标。
        font_size (int): 文本字体大小。
        color: 文本颜色。
        horizontal_alignment (str): 水平对齐方式。
        rotation: 旋转角度（单位为度）。
    """
    # 确保文本颜色明亮
    color = np.maximum(list(mcolors.to_rgb(color)), 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))

    x, y = position
    ax.text(
        x,
        y,
        text,
        size=font_size,
        family="sans-serif",
        bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.5, "edgecolor": "none"},
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color,
        zorder=10,
        rotation=rotation,
    )


def visualize_predictions(image, targets, predictions, output_dir, draw_gt, draw_pre):
    UODD_TEST = ['seacucumber', 'seaurchin', 'scallop']
    clipart1k_TEST = ['sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse']
    NEUDET_TEST = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    ArTaxOr_TEST = ['Araneae', 'Coleoptera', 'Diptera', 'Hemiptera', 'Hymenoptera', 'Lepidoptera', 'Odonata']
    DIOR_TEST = ['Expressway-Service-area','Expressway-toll-station','airplane','airport','baseballfield','basketballcourt','bridge','chimney','dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']
    FISH_TEST = ['fish']

    class_names = UODD_TEST
    # CUSTOM_COLORS = [
    #     (220/255, 20/255, 60/255), (0/255, 128/255, 0/255), (0/255, 0/255, 255/255), 
    #     (255/255, 140/255, 0/255), (128/255, 0/255, 128/255),
    #     (0/255, 255/255, 255/255), (255/255, 0/255, 255/255), (75/255, 0/255, 130/255), 
    #     (0/255, 255/255, 127/255), (255/255, 69/255, 0/255),
    #     (255/255, 228/255, 181/255), (255/255, 20/255, 147/255), 
    #     (154/255, 205/255, 50/255), (139/255, 69/255, 19/255), 
    #     (255/255, 215/255, 0/255),
    #     (0/255, 191/255, 255/255), (47/255, 79/255, 79/255), 
    #     (188/255, 143/255, 143/255), (255/255, 99/255, 71/255), 
    #     (205/255, 92/255, 92/255),
    #     (144/255, 238/255, 144/255), (30/255, 144/255, 255/255), 
    #     (128/255, 128/255, 0/255), (107/255, 142/255, 35/255), 
    #     (255/255, 127/255, 80/255)
    #     ]
    #DIOR
    CUSTOM_COLORS = [
        (220/255, 20/255, 60/255), (205/255, 92/255, 92/255), (255/255, 215/255, 0/255), 
        (255/255, 140/255, 0/255), (255/255, 215/255, 0/255),
        (255/255, 215/255, 0/255), (255/255, 0/255, 255/255), (75/255, 0/255, 130/255), 
        (0/255, 255/255, 127/255), (255/255, 69/255, 0/255),
        (255/255, 228/255, 181/255), (255/255, 20/255, 147/255), 
        (154/255, 205/255, 50/255), (139/255, 69/255, 19/255), 
        (255/255, 215/255, 0/255),
        (0/255, 191/255, 255/255), (47/255, 79/255, 79/255), 
        (188/255, 143/255, 143/255), (255/255, 99/255, 71/255), 
        (205/255, 92/255, 92/255),
        (144/255, 238/255, 144/255), (30/255, 144/255, 255/255), 
        (128/255, 128/255, 0/255), (107/255, 142/255, 35/255), 
        (255/255, 127/255, 80/255)
        ]
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # 获取图像尺寸
    img_width, img_height = image.size
    default_font_size = max(np.sqrt(img_width * img_height) // 90, 10)

    if draw_gt:
        # 绘制gt框
        boxes = targets["boxes"].cpu().numpy()
        labels = targets["labels"].cpu().numpy()
        for i, box in enumerate(boxes):
            x, y, w, h = box
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            # 需要减一是uodd以外的数据集
            if class_names != UODD_TEST:
                label = class_names[labels[i] - 1]
                label_index = labels[i] - 1
            else:
                label = class_names[labels[i]]
                label_index = labels[i]
            # rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', line_style="-", alpha=0.5)
            # ax.add_patch(rect)
            # ax.text(x, y - 10, f'{label}', color='r', fontsize=12, backgroundcolor="none") 
            instance_area = w * h
            if instance_area < 1000 or h < 40:
                if ymax >= h - 5:
                    text_pos = (xmax, ymin)
                else:
                    text_pos = (xmin, ymax)

            height_ratio = h / np.sqrt(h * w)
            lighter_color = change_color_brightness(CUSTOM_COLORS[label_index % len(CUSTOM_COLORS)], brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                * 0.5
                * default_font_size
            )
            rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                        edgecolor=CUSTOM_COLORS[label_index % len(CUSTOM_COLORS)], facecolor='none')
            ax.add_patch(rect)
            draw_text(ax, f'{label}', (x, y), color=lighter_color, font_size=font_size)  # 使用 draw_text 绘制文本

    if draw_pre:
        # 绘制预测框
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        for i, box in enumerate(boxes):
            score = scores[i]
            if score > 0.2:  # 只可视化置信度大于0.5的预测
                xmin, ymin, xmax, ymax = box
                x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin
                # 需要减一是uodd以外的数据集
                if class_names != UODD_TEST:
                    label = class_names[labels[i] - 1]
                    label_index = labels[i] - 1
                else:
                    label = class_names[labels[i]]
                    label_index = labels[i]
                # rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', line_style="-", alpha=0.5)
                # ax.add_patch(rect)
                # ax.text(x, y - 10, f'{label}', color='r', fontsize=12, backgroundcolor="none") 
                instance_area = w * h
                if instance_area < 1000 or h < 40:
                    if ymax >= h - 5:
                        text_pos = (xmax, ymin)
                    else:
                        text_pos = (xmin, ymax)

                height_ratio = h / np.sqrt(h * w)
                lighter_color = change_color_brightness(CUSTOM_COLORS[label_index % len(CUSTOM_COLORS)], brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * default_font_size
                )
                rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                                        edgecolor=CUSTOM_COLORS[label_index % len(CUSTOM_COLORS)], facecolor='none')
                ax.add_patch(rect)
                draw_text(ax, f'{label}', (x, y), color=lighter_color, font_size=font_size)  # 使用 draw_text 绘制文本


    plt.axis('off')

    # 获取 image_id 并构造输出文件路径
    image_id = targets["image_id"].item()
    output_path = os.path.join(output_dir, f'{image_id}.jpg')

    # 保存为 jpg 文件
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='jpg')
    plt.close(fig)
    print(f"Image saved to {output_path}")
