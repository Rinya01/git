"""
离线数据增强脚本 - 修正版（确保输出标注文件）
"""
import os
import cv2
import albumentations as A
import numpy as np
from pathlib import Path

# ================== 配置区域 ==================
IMAGE_DIR = r"D:\demo\images"          # 原始图片目录
LABEL_DIR = r"D:\demo\labels"          # 原始标注目录
OUTPUT_DIR = r"D:\demo\augmented_data" # 输出目录
AUG_PER_IMAGE = 9                      # 每张图生成变体数
SAVE_ORIGINAL = True                   # 是否保留原图
# =============================================

def read_yolo_labels(label_path, img_w, img_h):
    """读取 YOLO 格式标注，返回 bboxes 和 class_labels"""
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            # 转换为 pascal_voc 格式 [x_min, y_min, x_max, y_max] 像素坐标
            x_min = (x_c - w / 2) * img_w
            y_min = (y_c - h / 2) * img_h
            x_max = (x_c + w / 2) * img_w
            y_max = (y_c + h / 2) * img_h
            bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(cls)
    return bboxes, class_labels

def save_yolo_labels(save_path, bboxes, class_labels, img_w, img_h):
    """保存为 YOLO 归一化格式"""
    with open(save_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            x_min, y_min, x_max, y_max = bbox
            x_c = ((x_min + x_max) / 2) / img_w
            y_c = ((y_min + y_max) / 2) / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h
            # 裁剪到 [0,1] 范围
            x_c = max(0, min(1, x_c))
            y_c = max(0, min(1, y_c))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def get_pipeline():
    return A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill_value=114, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussianBlur(blur_limit=(3, 3), p=0.3),
        A.RandomScale(scale_limit=(-0.2, 0.0), p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipeline = get_pipeline()

    img_files = list(Path(IMAGE_DIR).glob("*.*"))
    img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    total_generated = 0

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告：无法读取 {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 对应的标注文件路径
        label_path = Path(LABEL_DIR) / img_path.with_suffix('.txt').name
        bboxes, class_labels = read_yolo_labels(str(label_path), w, h)

        base_name = img_path.stem

        # 保存原图及标注（如果存在）
        if SAVE_ORIGINAL:
            out_img = os.path.join(OUTPUT_DIR, f"{base_name}_orig.jpg")
            cv2.imwrite(out_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if bboxes:
                out_label = os.path.join(OUTPUT_DIR, f"{base_name}_orig.txt")
                save_yolo_labels(out_label, bboxes, class_labels, w, h)
            total_generated += 1

        # 生成增强变体
        for i in range(AUG_PER_IMAGE):
            if bboxes:
                augmented = pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
            else:
                augmented = pipeline(image=img)
                aug_img = augmented['image']
                aug_bboxes, aug_labels = [], []

            out_img = os.path.join(OUTPUT_DIR, f"{base_name}_aug{i}.jpg")
            cv2.imwrite(out_img, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            if aug_bboxes:
                out_label = os.path.join(OUTPUT_DIR, f"{base_name}_aug{i}.txt")
                save_yolo_labels(out_label, aug_bboxes, aug_labels, w, h)

            total_generated += 1

        print(f"处理完成：{img_path.name} -> 生成 {AUG_PER_IMAGE} 变体")

    print(f"\n总计生成 {total_generated} 个文件至 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()