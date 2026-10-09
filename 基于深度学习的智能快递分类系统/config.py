# -*- coding: utf-8 -*-
"""
全局配置文件
集中管理所有路径与可调参数，避免散落在各模块中的硬编码。
所有路径均基于项目根目录解析为绝对路径，与运行时工作目录无关。
"""
import os
from pathlib import Path

# ---------------- 项目根目录 ----------------
ROOT = Path(__file__).resolve().parent

# ---------------- 模型权重 ----------------
# 包裹分类模型（第一阶段检测）
PACKAGE_WEIGHTS = str(ROOT / 'runs' / 'train' / 'exp3' / 'weights' / 'best.pt')
# 面单检测模型（第二阶段检测）
WAYBILL_WEIGHTS = str(ROOT / 'runs' / 'train' / 'waybill_detect4' / 'weights' / 'best.pt')

# ---------------- PaddleOCR 模型目录 ----------------
# 注：PaddleOCR 在 Windows 上无法打开包含中文字符的路径，
# 因此模型文件放在项目外的纯 ASCII 路径下。
_PADDLE_MODELS_ROOT = Path(r'E:\paddle_ocr_models')
PADDLE_DET_DIR = str(_PADDLE_MODELS_ROOT / 'ch_PP-OCRv4_det_infer')
PADDLE_REC_DIR = str(_PADDLE_MODELS_ROOT / 'ch_PP-OCRv4_rec_infer')
PADDLE_CLS_DIR = str(_PADDLE_MODELS_ROOT / 'ch_ppocr_mobile_v2.0_cls_infer')

# ---------------- 数据库 ----------------
DATABASE_DIR = ROOT / 'database'
DB_PATH = str(DATABASE_DIR / 'app_data.db')

# ---------------- UI 资源与临时文件 ----------------
UI_DIR = str(ROOT / 'images' / 'UI')
# 注：cv2 (OpenCV) 和 PaddleOCR 在 Windows 上无法处理包含中文字符的路径，
# 因此临时文件放在纯 ASCII 路径下（项目外的 E:\ocr_temp）。
TMP_DIR = r'E:\ocr_temp'
# 登录/主窗口图标（若文件不存在，Qt 会静默忽略，此处启动时给出告警）
UI_ICON = os.path.join(UI_DIR, 'lufei.png')
UI_IMG_LEFT = os.path.join(UI_DIR, 'up.jpeg')      # 图片检测左侧占位图
UI_IMG_RIGHT = os.path.join(UI_DIR, 'right.jpeg')  # 图片检测右侧占位图
UI_IMG_VIDEO = os.path.join(UI_DIR, 'c.jpeg')      # 视频检测占位图

# 临时文件（上传预览 / 检测结果 / 面单裁剪图）
TMP_UPLOAD = os.path.join(TMP_DIR, 'tmp_upload.')
TMP_UPLOAD_SHOW = os.path.join(TMP_DIR, 'upload_show_result.jpg')
TMP_SINGLE_RESULT = os.path.join(TMP_DIR, 'single_result.jpg')
TMP_VID_RESULT = os.path.join(TMP_DIR, 'single_result_vid.jpg')
TMP_WAYBILL_CROP = os.path.join(TMP_DIR, 'waybill_crop.jpg')

# OCR worker 脚本（由 window.py 通过 subprocess 调用）
OCR_WORKER_SCRIPT = str(ROOT / 'ocr_worker.py')

# ---------------- 推理参数 ----------------
IMG_SIZE = [640, 640]          # 推理输入尺寸
MAX_DET = 1000                 # 单张图最大检测框数
# 包裹分类模型
CONF_THRES_PACKAGE = 0.25
IOU_THRES_PACKAGE = 0.45
# 面单检测模型
CONF_THRES_WAYBILL = 0.33
IOU_THRES_WAYBILL = 0.50
# 面单裁剪外扩像素
WAYBILL_CROP_MARGIN = 50
# 结果展示图高度（像素）
OUTPUT_SIZE = 480

# ---------------- OCR 参数 ----------------
OCR_CONF_THRES = 0.5           # OCR 识别置信度过滤阈值
OCR_MIN_SIDE = 800             # 裁剪面单短边小于该值时放大，再送 OCR
OCR_SERVICE_READY_TIMEOUT = 60  # OCR 常驻服务初始化（模型加载）超时，秒
OCR_REQUEST_TIMEOUT = 30       # OCR 单次识别请求超时，秒
ADDRESS_MERGE_Y_THRESHOLD = 50  # 地址合并垂直距离阈值（像素）
ADDRESS_TYPE = 'receiver'      # 'receiver' 收件地址，'sender' 寄件地址

# OCR 字段识别关键词
NAME_KEYWORDS = ['收件人', '收货人', '姓名', '联系人', '收方', '收件', '客户姓名', '名称']
PHONE_KEYWORDS = ['电话', '手机', '联系方式', '联系', '手机号', '号码', '电话号', '移动电话']
TRACKING_KEYWORDS = ['运单号', '单号', '追踪号', '订单号', '快递单号', '运单编号', '编号', 'Tracking', 'No.']

# 忽略无关文本关键词（含包裹分类类别）
IGNORE_KEYWORDS = [
    '捆码', '补贴', '热线', '零食', '怀旧', '批发', '盒装', '馋香友', '猫耳朵', '进入', '谨防',
    '已验视', '买低价', '大牌正品', 'zhixiang', '纸箱', '文件封', '塑料袋', '申通', '中通',
    '圆通', '韵达', '顺丰', '邮政', '极兔', '京东', '百世', '德邦', 'STO'
]


def ensure_dirs():
    """创建运行期需要的目录（幂等）"""
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(DATABASE_DIR, exist_ok=True)


def check_paths():
    """启动时校验关键文件是否存在，缺失时打印告警（不中断运行）"""
    checks = [
        ('包裹分类模型权重', PACKAGE_WEIGHTS),
        ('面单检测模型权重', WAYBILL_WEIGHTS),
        ('PaddleOCR 检测模型', PADDLE_DET_DIR),
        ('PaddleOCR 识别模型', PADDLE_REC_DIR),
        ('PaddleOCR 方向分类模型', PADDLE_CLS_DIR),
        ('窗口图标', UI_ICON),
    ]
    missing = []
    for name, path in checks:
        if not os.path.exists(path):
            missing.append(f'[配置告警] {name} 不存在: {path}')
    for msg in missing:
        print(msg)
    return missing
