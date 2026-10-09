#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Worker - 面单信息结构化提取（增强版）
调用方式：python ocr_worker.py <image_path>
输出：JSON 字符串，包含 name, phone, address, tracking_no 四个字段
"""

import sys
import json
import re
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import config

# ================== 配置参数（统一维护于 config.py） ==================
ADDRESS_MERGE_Y_THRESHOLD = config.ADDRESS_MERGE_Y_THRESHOLD   # 地址合并垂直距离阈值
ADDRESS_TYPE = config.ADDRESS_TYPE        # 'receiver' 收件地址，'sender' 寄件地址

# 关键词列表
NAME_KEYWORDS = config.NAME_KEYWORDS
PHONE_KEYWORDS = config.PHONE_KEYWORDS
TRACKING_KEYWORDS = config.TRACKING_KEYWORDS

# 忽略无关文本关键词（含包裹分类类别）
IGNORE_KEYWORDS = config.IGNORE_KEYWORDS

# 正则表达式
MOBILE_PATTERN = re.compile(r'1[3-9]\d[\s\-]?\d{4}[\s\-]?\d{4}')
PHONE_PATTERN = re.compile(r'\d{3,4}[\s\-]?\d{7,8}')
TRACKING_PATTERN = re.compile(r'[A-Z0-9]{10,30}', re.IGNORECASE)
ADDRESS_PATTERN = re.compile(r'(省|市|区|县|镇|乡|村|路|街|道|巷|弄|号|幢|栋|单元|室|楼|小区|花园|苑|广场|大厦|中心|学院|大学|工业园|开发区|组|队)')
DIGITS_ONLY = re.compile(r'^\d+$')


def preprocess_image(image_path):
    """图像预处理：灰度化、CLAHE增强、锐化，返回预处理后的图像路径"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 如果图片尺寸过小，放大至短边800像素
    h, w = img.shape[:2]
    if min(h, w) < config.OCR_MIN_SIDE:
        scale = config.OCR_MIN_SIDE / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # 轻度锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    # 转回三通道
    processed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(image_path, processed)
    return image_path


def extract_name_phone_from_text(text):
    phone_match = MOBILE_PATTERN.search(text)
    if phone_match:
        phone = phone_match.group()
        name = text.replace(phone, '').strip()
        name = re.sub(r'[\s,，、]', '', name)
        if 2 <= len(name) <= 4 and re.fullmatch(r'[\u4e00-\u9fa5]+', name):
            return name, phone
    return None, None


def classify_text_block(text, box, all_boxes):
    text_clean = text.strip()
    if not text_clean:
        return 'unknown'

    # 过滤无关关键词
    if any(kw in text_clean for kw in IGNORE_KEYWORDS):
        return 'ignore'

    # 电话特征优先
    if MOBILE_PATTERN.search(text_clean) or any(kw in text_clean for kw in PHONE_KEYWORDS):
        return 'phone'

    # 运单号特征
    if DIGITS_ONLY.fullmatch(text_clean) and len(text_clean) >= 12:
        return 'tracking_no'
    if any(kw in text_clean for kw in TRACKING_KEYWORDS) or TRACKING_PATTERN.fullmatch(text_clean):
        return 'tracking_no'

    # 姓名特征
    if any(kw in text_clean for kw in NAME_KEYWORDS):
        return 'name'
    if re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', text_clean):
        if not ADDRESS_PATTERN.search(text_clean):
            return 'name'

    # 地址分类（区分收/寄件地址）
    if '收' in text_clean:
        return 'receiver_address'
    if '寄' in text_clean:
        return 'sender_address'
    if ADDRESS_PATTERN.search(text_clean):
        return 'address'
    if len(text_clean) >= 6 and not DIGITS_ONLY.fullmatch(text_clean):
        return 'address'

    return 'unknown'


def merge_boxes(boxes, texts, classes, y_threshold=ADDRESS_MERGE_Y_THRESHOLD):
    if not boxes:
        return []
    items = list(zip(boxes, texts, classes))
    items.sort(key=lambda x: x[0][0][1])
    merged = []
    used = [False] * len(items)
    for i, (box_i, text_i, cls_i) in enumerate(items):
        if used[i]:
            continue
        if cls_i not in ('address', 'receiver_address', 'sender_address'):
            merged.append((box_i, text_i, cls_i))
            used[i] = True
            continue
        current_box = list(box_i)
        current_text = text_i
        current_cls = cls_i
        used[i] = True
        for j, (box_j, text_j, cls_j) in enumerate(items):
            if used[j] or cls_j not in ('address', 'receiver_address', 'sender_address'):
                continue
            y_i = (current_box[0][1] + current_box[2][1]) / 2
            y_j = (box_j[0][1] + box_j[2][1]) / 2
            if abs(y_i - y_j) < y_threshold:
                if max(current_box[0][0], box_j[0][0]) < min(current_box[2][0], box_j[2][0]):
                    new_x1 = min(current_box[0][0], box_j[0][0])
                    new_y1 = min(current_box[0][1], box_j[0][1])
                    new_x2 = max(current_box[2][0], box_j[2][0])
                    new_y2 = max(current_box[2][1], box_j[2][1])
                    current_box = [[new_x1, new_y1], [new_x2, new_y1], [new_x2, new_y2], [new_x1, new_y2]]
                    current_text += text_j
                    used[j] = True
        merged.append((current_box, current_text, current_cls))
    return merged


# ================== OCR 引擎单例 ==================
_OCR_ENGINE = None


def get_ocr_engine():
    """初始化并返回 PaddleOCR 单例。
    模型仅在进程内加载一次，常驻服务模式下多次识别复用同一引擎。"""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False,
                                det_model_dir=config.PADDLE_DET_DIR,
                                rec_model_dir=config.PADDLE_REC_DIR,
                                cls_model_dir=config.PADDLE_CLS_DIR)
    return _OCR_ENGINE


def extract_structured_info(image_path):
    # 图像预处理
    processed_path = preprocess_image(image_path)
    if processed_path is None:
        return {'name': '', 'phone': '', 'address': '', 'tracking_no': '', 'error': '图片读取失败'}

    ocr = get_ocr_engine()

    try:
        result = ocr.ocr(processed_path, cls=True)
    except Exception as e:
        return {'name': '', 'phone': '', 'address': '', 'tracking_no': '', 'error': str(e)}

    if not result or not result[0]:
        return {'name': '', 'phone': '', 'address': '', 'tracking_no': ''}

    boxes, texts = [], []
    for line in result[0]:
        if line[1][1] >= config.OCR_CONF_THRES:
            boxes.append(line[0])
            texts.append(line[1][0])

    if not texts:
        return {'name': '', 'phone': '', 'address': '', 'tracking_no': ''}

    # 调试输出（可选）
    # print("===== OCR识别文本 =====", file=sys.stderr)
    # for i, t in enumerate(texts):
    #     print(f"{i}: {t}", file=sys.stderr)
    # print("=======================", file=sys.stderr)

    classes = [classify_text_block(t, b, boxes) for t, b in zip(texts, boxes)]
    merged_items = merge_boxes(boxes, texts, classes)

    info = {'name': '', 'phone': '', 'address': '', 'tracking_no': ''}
    address_candidates = []

    for box, text, cls in merged_items:
        if cls in ('ignore',):
            continue

        # 姓名电话
        if cls == 'phone' and not info['phone']:
            name, phone = extract_name_phone_from_text(text)
            if name and phone:
                info['name'] = name
                info['phone'] = phone
            else:
                m = MOBILE_PATTERN.search(text)
                if m:
                    info['phone'] = m.group()
        elif cls == 'name' and not info['name']:
            info['name'] = text.strip()

        # 运单号
        if cls == 'tracking_no' and not info['tracking_no']:
            m = TRACKING_PATTERN.search(text) or DIGITS_ONLY.search(text)
            info['tracking_no'] = m.group() if m else text.strip()

        # 地址候选
        if cls in ('address', 'receiver_address', 'sender_address'):
            clean_text = re.sub(r'^[收发]', '', text).strip()
            if len(clean_text) >= 6 and ADDRESS_PATTERN.search(clean_text):
                address_candidates.append(clean_text)
    if address_candidates:
        info['address'] = max(address_candidates, key=len)
    else:
        for t in texts:
            if ADDRESS_PATTERN.search(t) and len(t) > len(info['address']):
                info['address'] = t
    info['address'] = re.sub(r'[\x00-\x1f\x7f]', '', info['address'])
    info['address'] = re.sub(r'\s+', '', info['address'])
    for box, text, cls in merged_items:
        if cls != 'unknown':
            continue
        if not info['name'] and re.fullmatch(r'[\u4e00-\u9fa5]{2,4}', text):
            info['name'] = text
        elif not info['phone'] and MOBILE_PATTERN.search(text):
            info['phone'] = MOBILE_PATTERN.search(text).group()
        elif not info['tracking_no'] and DIGITS_ONLY.fullmatch(text) and len(text) >= 12:
            info['tracking_no'] = text
        elif not info['address'] and ADDRESS_PATTERN.search(text):
            info['address'] = text

    return info
def serve():
    """常驻服务模式：PaddleOCR 只初始化一次，循环从 stdin 读取图片路径，
    识别结果以单行 JSON 写回 stdout（每次识别免去进程启动与模型加载开销）。
    协议：请求 = 一行图片路径；'exit' 或 EOF = 退出；响应 = 一行 JSON。"""
    # Windows 下强制 UTF-8，避免默认 GBK 编码破坏 JSON 中文
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

    try:
        get_ocr_engine()
    except Exception as e:
        print(json.dumps({'ready': False, 'error': str(e)}, ensure_ascii=False), flush=True)
        return

    print(json.dumps({'ready': True}, ensure_ascii=False), flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        if path.lower() in ('exit', 'quit'):
            break
        try:
            info = extract_structured_info(path)
        except Exception as e:
            info = {'name': '', 'phone': '', 'address': '', 'tracking_no': '', 'error': str(e)}
        print(json.dumps(info, ensure_ascii=False), flush=True)


def main():
    if '--serve' in sys.argv:
        serve()
        return
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) < 1:
        print(json.dumps({'name': '', 'phone': '', 'address': '', 'tracking_no': ''}, ensure_ascii=False))
        return
    path = args[0]
    if not os.path.exists(path):
        print(json.dumps({'name': '', 'phone': '', 'address': '', 'tracking_no': '', 'error': '文件不存在'}, ensure_ascii=False))
        return
    info = extract_structured_info(path)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == '__main__':
    main()