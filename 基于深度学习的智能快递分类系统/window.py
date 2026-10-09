# -*- coding: utf-8 -*-
import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import torch
import shutil
import subprocess
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal
import threading
import queue
import atexit
import sys
from pathlib import Path
import cv2
import torch.backends.cudnn as cudnn
import os.path as osp
import config
import db
import ui_style
import numpy as np
import json

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.datasets import LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, check_imshow, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


class LoginDialog(QDialog):
    """登录 / 注册窗口：左侧品牌区 + 右侧表单区（仅外观，验证逻辑不变）"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('用户登录')
        self.setFixedSize(880, 560)
        self.setWindowIcon(QIcon(config.UI_ICON))
        self.user_id = None
        self.is_login_mode = True
        self.initUI()
        self.toggle_mode()

    def initUI(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------------- 左：品牌区 ----------------
        brand_panel = QWidget()
        brand_panel.setObjectName('brandPanel')
        brand_panel.setFixedWidth(320)
        brand_layout = QVBoxLayout(brand_panel)
        brand_layout.setContentsMargins(30, 28, 26, 26)
        brand_layout.setSpacing(0)

        brand_top = QHBoxLayout()
        brand_top.setSpacing(10)
        logo = QLabel('快')
        logo.setFixedSize(34, 34)
        logo.setAlignment(Qt.AlignCenter)
        logo.setObjectName('brandLogo')
        app_name = QLabel('智能快递分类系统')
        app_name.setObjectName('brandName')
        brand_top.addWidget(logo)
        brand_top.addWidget(app_name)
        brand_top.addStretch(1)
        brand_layout.addLayout(brand_top)
        brand_layout.addStretch(1)

        slogan = QLabel('基于深度学习的\n快递包裹识别与面单提取')
        slogan.setObjectName('brandSlogan')
        slogan.setWordWrap(True)
        brand_layout.addWidget(slogan)
        brand_layout.addSpacing(12)

        intro = QLabel('包裹分类 · 面单检测 · OCR 字段提取\n支持图片与视频两种识别模式')
        intro.setObjectName('brandInfo')
        intro.setWordWrap(True)
        brand_layout.addWidget(intro)
        brand_layout.addSpacing(28)

        version = QLabel('v1.0 · 2026')
        version.setObjectName('brandVersion')
        brand_layout.addWidget(version)
        root.addWidget(brand_panel)

        # ---------------- 右：表单区 ----------------
        form_panel = QWidget()
        form_layout = QVBoxLayout(form_panel)
        form_layout.setContentsMargins(46, 46, 46, 38)
        form_layout.setSpacing(12)

        self.form_title = QLabel()
        self.form_title.setObjectName('loginFormTitle')
        self.form_sub = QLabel()
        self.form_sub.setObjectName('loginFormSub')
        form_layout.addWidget(self.form_title)
        form_layout.addWidget(self.form_sub)
        form_layout.addSpacing(18)

        def field_block(label_text, edit_widget, placeholder):
            block = QWidget()
            block_layout = QVBoxLayout(block)
            block_layout.setContentsMargins(0, 0, 0, 0)
            block_layout.setSpacing(5)
            label = QLabel(label_text)
            label.setObjectName('dlgFieldLabel')
            edit_widget.setObjectName('loginField')
            edit_widget.setPlaceholderText(placeholder)
            block_layout.addWidget(label)
            block_layout.addWidget(edit_widget)
            return block

        self.username_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.confirm_edit = QLineEdit()
        self.confirm_edit.setEchoMode(QLineEdit.Password)

        self.confirm_block = field_block('确认密码', self.confirm_edit, '确认密码（仅注册时需要）')
        self.password_block = field_block('密码', self.password_edit, '请输入密码')
        self.username_block = field_block('用户名', self.username_edit, '请输入用户名')

        # 兼容旧引用
        self.confirm_label = self.confirm_block

        form_layout.addWidget(self.username_block)
        form_layout.addWidget(self.password_block)
        form_layout.addWidget(self.confirm_block)
        form_layout.addSpacing(8)

        self.login_btn = QPushButton('登 录')
        self.login_btn.setObjectName('loginPrimary')
        self.login_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.login_btn.clicked.connect(self.handle_login)
        self.register_btn = QPushButton('注 册')
        self.register_btn.setObjectName('loginPrimary')
        self.register_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.register_btn.clicked.connect(self.handle_register)
        self.register_btn.hide()
        form_layout.addWidget(self.login_btn)
        form_layout.addWidget(self.register_btn)

        self.mode_label = QLabel('<a href="#">没有账号？点击注册</a>')
        self.mode_label.setObjectName('loginSwitch')
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setOpenExternalLinks(False)
        self.mode_label.linkActivated.connect(self.toggle_mode)
        form_layout.addWidget(self.mode_label)
        form_layout.addStretch(1)
        root.addWidget(form_panel, 1)

    def toggle_mode(self):
        self.is_login_mode = not self.is_login_mode
        if self.is_login_mode:
            self.setWindowTitle('用户登录')
            self.form_title.setText('欢迎回来')
            self.form_sub.setText('登录后即可上传包裹图片开始识别')
            self.confirm_block.hide()
            self.login_btn.show()
            self.register_btn.hide()
            self.mode_label.setText('<a href="#">没有账号？点击注册</a>')
        else:
            self.setWindowTitle('用户注册')
            self.form_title.setText('创建账号')
            self.form_sub.setText('注册后即可登录使用系统')
            self.confirm_block.show()
            self.login_btn.hide()
            self.register_btn.show()
            self.mode_label.setText('<a href="#">已有账号？去登录</a>')
        self.username_edit.clear()
        self.password_edit.clear()
        self.confirm_edit.clear()

    def handle_login(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()
        if not username or not password:
            ui_style.msg_warning(self, '提示', '用户名和密码不能为空')
            return
        user_id = db.login_user(username, password)
        if user_id:
            self.user_id = user_id
            self.accept()
        else:
            ui_style.msg_warning(self, '错误', '用户名或密码错误')

    def handle_register(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()
        confirm = self.confirm_edit.text().strip()
        if not username or not password:
            ui_style.msg_warning(self, '提示', '用户名和密码不能为空')
            return
        if len(password) < 6:
            ui_style.msg_warning(self, '提示', '密码长度不能少于6位')
            return
        if password != confirm:
            ui_style.msg_warning(self, '提示', '两次输入的密码不一致')
            return
        if db.register_user(username, password):
            ui_style.msg_info(self, '成功', '注册成功，请登录')
            self.toggle_mode()
        else:
            ui_style.msg_warning(self, '错误', '用户名已存在')


class EditUserDialog(QDialog):
    """管理员修改用户对话框：可修改用户名、重置密码（留空表示不修改）"""

    def __init__(self, original_username, parent=None):
        super().__init__(parent)
        self.setWindowTitle('修改用户信息')
        self.setFixedSize(460, 330)
        self.setWindowIcon(QIcon(config.UI_ICON))
        self._new_username = None
        self._new_password = None
        self.initUI(original_username)

    def initUI(self, original_username):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = QWidget()
        head_layout = QVBoxLayout(head)
        head_layout.setContentsMargins(22, 20, 22, 16)
        head_layout.setSpacing(3)
        title = QLabel('修改用户信息')
        title.setObjectName('dlgTitle')
        sub = QLabel(f'正在修改：{original_username}')
        sub.setObjectName('dlgSub')
        head_layout.addWidget(title)
        head_layout.addWidget(sub)
        root.addWidget(head)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(22, 4, 22, 10)
        body_layout.setSpacing(16)

        self.username_edit = QLineEdit(original_username)
        self.username_edit.setPlaceholderText('新用户名（留空则不修改）')
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setPlaceholderText('留空则不修改密码')

        username_label = QLabel('用户名')
        username_label.setObjectName('dlgFieldLabel')
        username_block = QWidget()
        username_box = QVBoxLayout(username_block)
        username_box.setContentsMargins(0, 0, 0, 0)
        username_box.setSpacing(5)
        username_box.addWidget(username_label)
        username_box.addWidget(self.username_edit)

        password_label = QLabel('新密码')
        password_label.setObjectName('dlgFieldLabel')
        password_block = QWidget()
        password_box = QVBoxLayout(password_block)
        password_box.setContentsMargins(0, 0, 0, 0)
        password_box.setSpacing(5)
        password_box.addWidget(password_label)
        password_box.addWidget(self.password_edit)

        hint = QLabel('用户名至少 2 个字符，密码至少 6 位')
        hint.setObjectName('dlgHint')

        body_layout.addWidget(username_block)
        body_layout.addWidget(password_block)
        body_layout.addWidget(hint)
        root.addWidget(body)
        root.addStretch(1)

        bottom = QWidget()
        bottom.setObjectName('dlgBottom')
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(16, 12, 16, 14)
        bottom_layout.setSpacing(8)
        cancel_btn = QPushButton('取消')
        ui_style.set_variant(cancel_btn, 'ghost')
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton('保存修改')
        ok_btn.clicked.connect(self._on_ok)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(cancel_btn)
        bottom_layout.addWidget(ok_btn)
        root.addWidget(bottom)

    def _on_ok(self):
        new_username = self.username_edit.text().strip()
        new_password = self.password_edit.text()
        if new_username and len(new_username) < 2:
            ui_style.msg_warning(self, '提示', '用户名至少2个字符')
            return
        if new_password and len(new_password) < 6:
            ui_style.msg_warning(self, '提示', '密码长度不能少于6位')
            return
        if not new_username and not new_password:
            ui_style.msg_warning(self, '提示', '请至少填写一项要修改的内容')
            return
        self._new_username = new_username or None
        self._new_password = new_password or None
        self.accept()

    def get_result(self):
        return self._new_username, self._new_password


class OCRServiceClient:
    """PaddleOCR 常驻服务客户端（单例）。

    paddle 与 torch 无法在同一进程共存，OCR 引擎运行在独立的常驻子进程中：
    服务启动时加载一次模型，之后通过 stdin/stdout 以「一行路径 -> 一行 JSON」
    协议复用，免去每次识别都新建进程 + 重复加载模型的开销。
    支持崩溃自动重启；应用退出时通过 atexit/closeEvent 清理子进程。"""
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self.proc = None
        self.result_queue = queue.Queue()
        self.io_lock = threading.Lock()
        atexit.register(self.shutdown)

    # ---------- 服务进程管理 ----------
    def _start(self):
        env = os.environ.copy()
        env['CUDA_MODULE_LOADING'] = 'LAZY'
        env['PYTHONIOENCODING'] = 'utf-8'
        self.proc = subprocess.Popen(
            [sys.executable, config.OCR_WORKER_SCRIPT, '--serve'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='ignore',
            cwd=str(config.ROOT), env=env,
        )
        # 持续排空 stdout/stderr，防止管道缓冲区写满导致子进程阻塞
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        # 等待模型加载完成的就绪信号
        ready = json.loads(self.result_queue.get(timeout=config.OCR_SERVICE_READY_TIMEOUT))
        if not ready.get('ready'):
            raise RuntimeError(f"OCR服务初始化失败: {ready.get('error')}")
        print('OCR常驻服务已就绪 (PID %d)' % self.proc.pid)

    def _pump_stdout(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if line:
                    self.result_queue.put(line)
        except Exception:
            pass

    def _pump_stderr(self):
        try:
            for line in self.proc.stderr:
                line = line.strip()
                if line:
                    print('[OCR服务]', line)
        except Exception:
            pass

    def _ensure_running(self):
        if self.proc is None or self.proc.poll() is not None:
            # 服务不存在或已崩溃：清空残留结果后重启
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
            self._start()

    # ---------- 对外接口 ----------
    def recognize(self, image_path):
        """识别一张面单图片，返回含 name/phone/address/tracking_no 的 dict；
        失败时返回带 error 键的 dict（服务将在下次调用时自动重启）。"""
        with self.io_lock:
            try:
                self._ensure_running()
                self.proc.stdin.write(image_path + '\n')
                self.proc.stdin.flush()
                line = self.result_queue.get(timeout=config.OCR_REQUEST_TIMEOUT)
                return json.loads(line)
            except Exception as e:
                print(f'OCR服务调用失败: {e}')
                self.shutdown()
                return {'name': '', 'phone': '', 'address': '', 'tracking_no': '', 'error': str(e)}

    def shutdown(self):
        """优雅停止服务进程：先发 exit，超时则 terminate/kill。可重复调用。"""
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write('exit\n')
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


class MainWindow(QWidget):
    """主窗口：左侧导航 + 内容栈"""
    frame_ready = pyqtSignal(QPixmap)

    def __init__(self, user_id):
        super().__init__()
        self.current_user_id = user_id
        self.setWindowTitle('智能快递分类系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon(config.UI_ICON))

        config.ensure_dirs()
        config.check_paths()
        db.init_db()
        self.user_role = db.get_user_role(user_id)  # 'admin' / 'user'
        self.ocr_client = OCRServiceClient()  # 常驻OCR服务（首次识别时才真正拉起进程）

        self.output_size = config.OUTPUT_SIZE
        self.img2predict = ""
        self.device = '0' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'

        self._build_chrome()

        self.model_package = self.model_load(weights=config.PACKAGE_WEIGHTS, device=self.device)
        self.model_waybill = self.model_load(weights=config.WAYBILL_WEIGHTS, device=self.device)
        self.model = self.model_package

        self.initUI()
        self._nav_list.setCurrentRow(0)
        self.reset_vid()
        self.stopEvent = threading.Event()
        self.frame_ready.connect(self.update_video_frame)

    def _build_chrome(self):
        """构建左侧导航 + 内容栈"""
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        side = QWidget()
        side.setObjectName('side')
        side.setFixedWidth(208)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 16, 12, 14)
        side_layout.setSpacing(2)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(2, 0, 2, 12)
        brand_row.setSpacing(8)
        logo = QLabel('快')
        logo.setFixedSize(26, 26)
        logo.setAlignment(Qt.AlignCenter)
        logo.setObjectName('brandLogo')
        app_name = QLabel('智能快递分类系统')
        app_name.setObjectName('brandName')
        app_name.setWordWrap(True)
        brand_row.addWidget(logo)
        brand_row.addWidget(app_name, 1)
        side_layout.addLayout(brand_row)

        self._nav_list = QListWidget()
        self._nav_list.setObjectName('nav')
        self._nav_list.setFocusPolicy(Qt.NoFocus)
        self._nav_list.setSpacing(2)
        self._nav_list.currentRowChanged.connect(self._on_nav_changed)
        side_layout.addWidget(self._nav_list, 1)

        user_row = QHBoxLayout()
        user_row.setContentsMargins(2, 12, 2, 0)
        user_row.setSpacing(8)
        ava = QLabel('管' if self.user_role == 'admin' else '用')
        ava.setFixedSize(28, 28)
        ava.setAlignment(Qt.AlignCenter)
        ava.setObjectName('userAva')
        user_col = QWidget()
        user_col_layout = QVBoxLayout(user_col)
        user_col_layout.setContentsMargins(0, 0, 0, 0)
        user_col_layout.setSpacing(0)
        role_label = QLabel('管理员' if self.user_role == 'admin' else '普通用户')
        role_label.setObjectName('userRole')
        account_label = QLabel('当前账号')
        account_label.setObjectName('userName')
        user_col_layout.addWidget(role_label)
        user_col_layout.addWidget(account_label)
        user_row.addWidget(ava)
        user_row.addWidget(user_col, 1)
        side_layout.addLayout(user_row)

        root.addWidget(side)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        self._nav_icons = [
            ui_style.nav_icon('image'), ui_style.nav_icon('video'),
            ui_style.nav_icon('clock'), ui_style.nav_icon('users'),
            ui_style.nav_icon('log'),
        ]

    def addTab(self, widget, title):
        """将页面注册到内容栈，并添加对应导航项到侧栏"""
        index = self.stack.count()
        self.stack.addWidget(widget)
        item = QListWidgetItem(title)
        if index < len(self._nav_icons):
            item.setIcon(self._nav_icons[index])
        self._nav_list.addItem(item)
        return index

    def setTabIcon(self, index, icon):
        """兼容旧调用，侧边导航图标由侧栏统一绘制"""
        return None

    def _on_nav_changed(self, row):
        if 0 <= row < self.stack.count():
            self.stack.setCurrentIndex(row)

    def crop_waybill(self, image_bgr, detections, margin=config.WAYBILL_CROP_MARGIN):
        """选择面积最大的面单检测框，并向外扩展margin像素"""
        if detections is None or len(detections) == 0:
            return None
        areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        best_idx = np.argmax(areas)
        best_det = detections[best_idx]
        x1, y1, x2, y2 = map(int, best_det[:4])
        h, w = image_bgr.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        if x2 <= x1 or y2 <= y1:
            return None
        return image_bgr[y1:y2, x1:x2]

    @torch.no_grad()
    def model_load(self, weights="", device='', half=False, dnn=False):
        device = select_device(device)
        half &= device.type != 'cpu'
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt = model.stride, model.names, model.pt
        half &= pt and device.type != 'cpu'
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model

    def initUI(self):
        """构建功能页面（图片检测 / 视频检测 / 历史记录 + 管理员页）"""

        # ---------------- 图片检测 ----------------
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout(img_detection_widget)
        img_detection_layout.setContentsMargins(0, 0, 0, 0)
        img_detection_layout.setSpacing(14)

        img_header = QWidget()
        img_header_layout = QHBoxLayout(img_header)
        img_header_layout.setContentsMargins(0, 0, 0, 0)
        img_title_box = QVBoxLayout()
        img_title_box.setSpacing(2)
        img_title_box.setContentsMargins(0, 0, 0, 0)
        img_title = QLabel('图片识别')
        img_title.setObjectName('pageTitle')
        img_sub = QLabel('上传包裹照片，自动识别包裹类型并提取面单信息')
        img_sub.setObjectName('pageSub')
        img_title_box.addWidget(img_title)
        img_title_box.addWidget(img_sub)
        img_header_layout.addLayout(img_title_box)
        img_header_layout.addStretch(1)
        up_img_button = QPushButton('上传图片')
        ui_style.set_variant(up_img_button, 'ghost')
        det_img_button = QPushButton('开始检测')
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        img_header_layout.addWidget(up_img_button)
        img_header_layout.addWidget(det_img_button)
        img_detection_layout.addWidget(img_header)

        # 原图 / 识别结果 双卡片
        duo_widget = QWidget()
        duo_layout = QHBoxLayout(duo_widget)
        duo_layout.setContentsMargins(0, 0, 0, 0)
        duo_layout.setSpacing(14)

        def image_card(header_text, side_text):
            card = QFrame()
            card.setObjectName('card')
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 12, 12, 12)
            card_layout.setSpacing(10)
            card_header = QHBoxLayout()
            card_header.setContentsMargins(0, 0, 0, 0)
            head_label = QLabel(header_text)
            head_label.setObjectName('cardHeader')
            side_label = QLabel(side_text)
            side_label.setObjectName('cardSide')
            card_header.addWidget(head_label)
            card_header.addStretch(1)
            card_header.addWidget(side_label)
            card_layout.addLayout(card_header)
            return card, card_layout

        left_card, left_layout = image_card('原图', '等待上传')
        self.left_img = QLabel()
        self.left_img.setObjectName('imageFrame')
        self.left_img.setAlignment(Qt.AlignCenter)
        self.left_img.setPixmap(QPixmap(config.UI_IMG_LEFT))
        left_layout.addWidget(self.left_img, 1)

        right_card, right_layout = image_card('识别结果', '—')
        self.right_img = QLabel()
        self.right_img.setObjectName('imageFrame')
        self.right_img.setAlignment(Qt.AlignCenter)
        self.right_img.setPixmap(QPixmap(config.UI_IMG_RIGHT))
        right_layout.addWidget(self.right_img, 1)

        duo_layout.addWidget(left_card)
        duo_layout.addWidget(right_card)
        img_detection_layout.addWidget(duo_widget, 1)

        # 识别信息卡片
        result_card = QFrame()
        result_card.setObjectName('card')
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(12, 12, 12, 12)
        result_layout.setSpacing(10)
        result_header = QHBoxLayout()
        result_header.setContentsMargins(0, 0, 0, 0)
        result_title = QLabel('识别信息（OCR 结果）')
        result_title.setObjectName('cardHeader')
        result_side = QLabel('包裹类型 · 面单字段')
        result_side.setObjectName('cardSide')
        result_header.addWidget(result_title)
        result_header.addStretch(1)
        result_header.addWidget(result_side)
        result_layout.addLayout(result_header)
        self.ocr_result_text = QTextEdit()
        self.ocr_result_text.setObjectName('ocrText')
        self.ocr_result_text.setReadOnly(True)
        self.ocr_result_text.setMaximumHeight(150)
        self.ocr_result_text.setPlaceholderText('识别完成后显示包裹类型与面单字段')
        result_layout.addWidget(self.ocr_result_text)
        img_detection_layout.addWidget(result_card)

        self.addTab(img_detection_widget, '图片检测')

        # ---------------- 视频检测 ----------------
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout(vid_detection_widget)
        vid_detection_layout.setContentsMargins(0, 0, 0, 0)
        vid_detection_layout.setSpacing(14)

        vid_header = QWidget()
        vid_header_layout = QHBoxLayout(vid_header)
        vid_header_layout.setContentsMargins(0, 0, 0, 0)
        vid_title_box = QVBoxLayout()
        vid_title_box.setSpacing(2)
        vid_title_box.setContentsMargins(0, 0, 0, 0)
        vid_title = QLabel('视频检测')
        vid_title.setObjectName('pageTitle')
        vid_sub = QLabel('支持摄像头实时监测与本地视频文件逐帧检测')
        vid_sub.setObjectName('pageSub')
        vid_title_box.addWidget(vid_title)
        vid_title_box.addWidget(vid_sub)
        vid_header_layout.addLayout(vid_title_box)
        vid_header_layout.addStretch(1)
        self.webcam_detection_btn = QPushButton('摄像头实时监测')
        self.mp4_detection_btn = QPushButton('视频文件检测')
        ui_style.set_variant(self.mp4_detection_btn, 'ghost')
        self.vid_stop_btn = QPushButton('停止检测')
        ui_style.set_variant(self.vid_stop_btn, 'danger')
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        vid_header_layout.addWidget(self.webcam_detection_btn)
        vid_header_layout.addWidget(self.mp4_detection_btn)
        vid_header_layout.addWidget(self.vid_stop_btn)
        vid_detection_layout.addWidget(vid_header)

        vid_card = QFrame()
        vid_card.setObjectName('card')
        vid_card_layout = QVBoxLayout(vid_card)
        vid_card_layout.setContentsMargins(12, 12, 12, 12)
        vid_card_layout.setSpacing(10)
        vid_card_header = QHBoxLayout()
        vid_card_header.setContentsMargins(0, 0, 0, 0)
        vid_head_label = QLabel('检测画面')
        vid_head_label.setObjectName('cardHeader')
        self.vid_status_chip = ui_style.chip('待机')
        vid_card_header.addWidget(vid_head_label)
        vid_card_header.addStretch(1)
        vid_card_header.addWidget(self.vid_status_chip)
        vid_card_layout.addLayout(vid_card_header)
        self.vid_img = QLabel()
        self.vid_img.setObjectName('imageFrame')
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.vid_img.setScaledContents(False)
        self.vid_img.setMinimumHeight(340)
        self.vid_img.setPixmap(QPixmap(config.UI_IMG_VIDEO))
        vid_card_layout.addWidget(self.vid_img, 1)
        self.vid_counter = QLabel('未开始检测')
        self.vid_counter.setObjectName('statusText')
        vid_card_layout.addWidget(self.vid_counter)
        vid_detection_layout.addWidget(vid_card)

        self.addTab(vid_detection_widget, '视频检测')

        # ---------------- 历史记录 ----------------
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(14)

        history_header = QWidget()
        history_header_layout = QHBoxLayout(history_header)
        history_header_layout.setContentsMargins(0, 0, 0, 0)
        history_title_box = QVBoxLayout()
        history_title_box.setSpacing(2)
        history_title_box.setContentsMargins(0, 0, 0, 0)
        history_title = QLabel('识别历史记录')
        history_title.setObjectName('pageTitle')
        history_sub = QLabel('仅显示当前账号的识别记录，按时间倒序')
        history_sub.setObjectName('pageSub')
        history_title_box.addWidget(history_title)
        history_title_box.addWidget(history_sub)
        history_header_layout.addLayout(history_title_box)
        history_header_layout.addStretch(1)
        refresh_btn = QPushButton('刷新记录')
        ui_style.set_variant(refresh_btn, 'ghost')
        refresh_btn.clicked.connect(self.load_history)
        history_header_layout.addWidget(refresh_btn)
        history_layout.addWidget(history_header)

        history_card = QFrame()
        history_card.setObjectName('card')
        history_card_layout = QVBoxLayout(history_card)
        history_card_layout.setContentsMargins(10, 10, 10, 12)
        history_card_layout.setSpacing(8)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(8)
        self.history_table.setHorizontalHeaderLabels(["ID", "识别时间", "类型", "包裹类型", "姓名", "电话", "地址摘要", "操作"])
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 识别时间
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 类型
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 包裹类型
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 姓名
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # 电话
        header.setSectionResizeMode(6, QHeaderView.Stretch)  # 地址摘要
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # 操作
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setMinimumSectionSize(70)  # 操作列按钮最小宽度
        history_card_layout.addWidget(self.history_table)
        self.history_count = QLabel('')
        self.history_count.setObjectName('statusText')
        history_card_layout.addWidget(self.history_count)
        history_layout.addWidget(history_card)

        self.addTab(history_widget, '历史记录')
        self._hist_nav_index = self._nav_list.count() - 1

        # 管理员专属标签页
        if self.user_role == 'admin':
            self.init_admin_tabs()

    def init_admin_tabs(self):
        """管理员标签页：用户管理 + 登录日志"""
        users = db.get_all_users()
        total = len(users)
        admins = sum(1 for u in users if u[2] == 'admin')

        # ---------------- 用户管理 ----------------
        user_mgmt_widget = QWidget()
        user_mgmt_layout = QVBoxLayout(user_mgmt_widget)
        user_mgmt_layout.setContentsMargins(0, 0, 0, 0)
        user_mgmt_layout.setSpacing(14)

        user_header = QWidget()
        user_header_layout = QHBoxLayout(user_header)
        user_header_layout.setContentsMargins(0, 0, 0, 0)
        user_title_box = QVBoxLayout()
        user_title_box.setSpacing(2)
        user_title_box.setContentsMargins(0, 0, 0, 0)
        user_title = QLabel('用户管理')
        user_title.setObjectName('pageTitle')
        user_sub = QLabel(f'系统内共 {total} 个账号，其中管理员 {admins} 个')
        user_sub.setObjectName('pageSub')
        user_title_box.addWidget(user_title)
        user_title_box.addWidget(user_sub)
        user_header_layout.addLayout(user_title_box)
        user_header_layout.addStretch(1)
        refresh_user_btn = QPushButton('刷新列表')
        ui_style.set_variant(refresh_user_btn, 'ghost')
        refresh_user_btn.clicked.connect(self.load_users)
        user_header_layout.addWidget(refresh_user_btn)
        user_mgmt_layout.addWidget(user_header)

        # 概览数字
        stats_row = QWidget()
        stats_layout = QHBoxLayout(stats_row)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(12)
        stats_layout.addWidget(ui_style.stat_tile(total, '用户总数'))
        stats_layout.addWidget(ui_style.stat_tile(admins, '管理员'))
        stats_layout.addWidget(ui_style.stat_tile(total - admins, '普通用户'))
        stats_layout.addStretch(1)
        user_mgmt_layout.addWidget(stats_row)

        user_card = QFrame()
        user_card.setObjectName('card')
        user_card_layout = QVBoxLayout(user_card)
        user_card_layout.setContentsMargins(10, 10, 10, 12)
        user_card_layout.setSpacing(8)
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(6)
        self.user_table.setHorizontalHeaderLabels(["ID", "用户名", "角色", "注册时间", "最后登录", "操作"])
        header = self.user_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # 用户名
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 角色
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 注册时间
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 最后登录
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # 操作
        self.user_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.user_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.user_table.verticalHeader().setVisible(False)
        self.user_table.horizontalHeader().setMinimumSectionSize(70)  # 操作列按钮最小宽度
        user_card_layout.addWidget(self.user_table)
        user_mgmt_layout.addWidget(user_card)

        self.addTab(user_mgmt_widget, '用户管理')

        # ---------------- 登录日志 ----------------
        login_log_widget = QWidget()
        login_log_layout = QVBoxLayout(login_log_widget)
        login_log_layout.setContentsMargins(0, 0, 0, 0)
        login_log_layout.setSpacing(14)

        log_header = QWidget()
        log_header_layout = QHBoxLayout(log_header)
        log_header_layout.setContentsMargins(0, 0, 0, 0)
        log_title_box = QVBoxLayout()
        log_title_box.setSpacing(2)
        log_title_box.setContentsMargins(0, 0, 0, 0)
        log_title = QLabel('用户登录日志')
        log_title.setObjectName('pageTitle')
        log_sub = QLabel('记录全部登录尝试（含失败），按时间倒序')
        log_sub.setObjectName('pageSub')
        log_title_box.addWidget(log_title)
        log_title_box.addWidget(log_sub)
        log_header_layout.addLayout(log_title_box)
        log_header_layout.addStretch(1)
        refresh_log_btn = QPushButton('刷新日志')
        ui_style.set_variant(refresh_log_btn, 'ghost')
        refresh_log_btn.clicked.connect(self.load_login_logs)
        log_header_layout.addWidget(refresh_log_btn)
        login_log_layout.addWidget(log_header)

        log_card = QFrame()
        log_card.setObjectName('card')
        log_card_layout = QVBoxLayout(log_card)
        log_card_layout.setContentsMargins(10, 10, 10, 12)
        log_card_layout.setSpacing(8)

        # 筛选栏（全部 / 成功 / 失败）
        self._log_filter = 'all'
        self._log_filter_btns = {}
        filter_row = QWidget()
        filter_layout = QHBoxLayout(filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(8)
        for key in ('all', 'ok', 'fail'):
            btn = QPushButton('全部')
            btn.setObjectName('filterChip')
            btn.setProperty('on', str(key == 'all').lower())
            btn.clicked.connect(lambda checked, k=key: self._set_log_filter(k))
            filter_layout.addWidget(btn)
            self._log_filter_btns[key] = btn
        filter_layout.addStretch(1)
        log_card_layout.addWidget(filter_row)

        self.login_log_table = QTableWidget()
        self.login_log_table.setColumnCount(4)
        self.login_log_table.setHorizontalHeaderLabels(["ID", "用户名", "登录时间", "结果"])
        header = self.login_log_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # ID
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # 用户名
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 登录时间
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 结果
        self.login_log_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.login_log_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.login_log_table.verticalHeader().setVisible(False)
        self.login_log_table.horizontalHeader().setMinimumSectionSize(80)  # 列宽兜底
        log_card_layout.addWidget(self.login_log_table)
        login_log_layout.addWidget(log_card)

        self.addTab(login_log_widget, '登录日志')

        self.load_users()
        self.load_login_logs()

    def _set_log_filter(self, key):
        """登录日志结果筛选（仅过滤当前展示，不改变记录）"""
        self._log_filter = key
        for k, btn in self._log_filter_btns.items():
            btn.setProperty('on', 'true' if k == key else 'false')
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._render_login_logs()

    def load_users(self):
        """加载全部用户列表"""
        users = db.get_all_users()
        self.user_table.setRowCount(len(users))
        self.user_table.verticalHeader().setDefaultSectionSize(36)  # 行高确保按钮显示完整
        for row_idx, u in enumerate(users):
            uid, username, role, reg_time, last_login = u
            role_text = '管理员' if role == 'admin' else '普通用户'
            for col, val in zip((0, 1, 3, 4), (str(uid), username, reg_time or '', last_login or '')):
                self.user_table.setItem(row_idx, col, QTableWidgetItem(val))
            tone = 'blue' if role == 'admin' else ''
            self.user_table.setCellWidget(row_idx, 2, ui_style.wrapped_chip(role_text, tone))
            delete_slot = (None if uid == self.current_user_id
                           else lambda checked, u=uid, n=username: self.delete_user_row(u, n))
            ops = ui_style.cell_buttons([
                ('修改', '', lambda checked, u=uid, n=username: self.edit_user(u, n)),
                ('删除', 'danger', delete_slot),
            ])
            self.user_table.setCellWidget(row_idx, 5, ops)

    def edit_user(self, uid, username):
        """打开修改用户对话框"""
        dialog = EditUserDialog(username, self)
        if dialog.exec_() == QDialog.Accepted:
            new_username, new_password = dialog.get_result()
            if db.update_user(uid, new_username, new_password):
                ui_style.msg_info(self, "成功", f"用户 {username} 的信息已更新")
                self.load_users()
            else:
                ui_style.msg_warning(self, "错误", "修改失败：用户名可能已存在或内容未填写")

    def delete_user_row(self, uid, username):
        """删除指定用户（不能删除当前登录账户）"""
        if uid == self.current_user_id:
            ui_style.msg_warning(self, "提示", "不能删除当前登录的账户")
            return
        if ui_style.msg_confirm(self, "确认删除",
                                f"确定要删除用户 {username} 吗？\n该用户的识别记录将一并删除，登录日志保留。"):
            if db.delete_user(uid):
                ui_style.msg_info(self, "成功", f"用户 {username} 已删除")
                self.load_users()
            else:
                ui_style.msg_warning(self, "错误", "删除失败，请稍后重试")

    def load_login_logs(self):
        """加载登录日志并缓存，按当前筛选条件渲染（不改变记录）"""
        self._login_logs = db.get_login_logs()
        self._render_login_logs()

    def _render_login_logs(self):
        logs = self._login_logs
        if self._log_filter == 'ok':
            shown = [log for log in logs if log[3]]
        elif self._log_filter == 'fail':
            shown = [log for log in logs if not log[3]]
        else:
            shown = logs
        ok_count = sum(1 for log in logs if log[3])
        fail_count = len(logs) - ok_count
        labels = {'all': '全部', 'ok': '成功', 'fail': '失败'}
        counts = {'all': len(logs), 'ok': ok_count, 'fail': fail_count}
        for key, label in labels.items():
            self._log_filter_btns[key].setText(f'{label} {counts[key]}')

        self.login_log_table.setRowCount(len(shown))
        self.login_log_table.verticalHeader().setDefaultSectionSize(34)  # 行高
        for row_idx, log in enumerate(shown):
            log_id, username, login_time, success = log
            self.login_log_table.setItem(row_idx, 0, QTableWidgetItem(str(log_id)))
            self.login_log_table.setItem(row_idx, 1, QTableWidgetItem(username))
            self.login_log_table.setItem(row_idx, 2, QTableWidgetItem(login_time))
            tone = 'green' if success else 'red'
            self.login_log_table.setCellWidget(
                row_idx, 3, ui_style.wrapped_chip('成功' if success else '失败', tone))

    def upload_img(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = config.TMP_UPLOAD + suffix
            os.makedirs(config.TMP_DIR, exist_ok=True)
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite(config.TMP_UPLOAD_SHOW, im0)
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap(config.TMP_UPLOAD_SHOW))
            self.right_img.setPixmap(QPixmap(config.UI_IMG_RIGHT))

    def detect_img(self):
        source = self.img2predict
        if source == "":
            ui_style.msg_warning(self, "请上传", "请先上传图片再进行检测")
            return

        device = select_device(self.device)
        imgsz = check_img_size(config.IMG_SIZE, s=self.model_package.stride)

        # 第一阶段：包裹分类
        model_pkg = self.model_package
        dataset = LoadImages(source, img_size=imgsz, stride=model_pkg.stride, auto=model_pkg.pt and not model_pkg.jit)
        pkg_type = "未知"
        pkg_detections = []
        im0_pkg = None
        for path, im, im0s, vid_cap, s in dataset:
            im0_pkg = im0s.copy()
            im = torch.from_numpy(im).to(device).float() / 255.0
            if len(im.shape) == 3:
                im = im[None]
            pred = model_pkg(im)
            pred = non_max_suppression(pred, config.CONF_THRES_PACKAGE, config.IOU_THRES_PACKAGE, None, False, max_det=config.MAX_DET)
            for det in pred:
                if len(det):
                    pkg_detections = det.clone()
                    if det.shape[1] > 6:
                        scores = det[:, 4] * det[:, 5]
                    else:
                        scores = det[:, 4]
                    best_idx = torch.argmax(scores).item()
                    pkg_type = model_pkg.names[int(det[best_idx, -1].item())]
            break

        # 第二阶段：面单检测
        model_wb = self.model_waybill
        dataset = LoadImages(source, img_size=imgsz, stride=model_wb.stride, auto=model_wb.pt and not model_wb.jit)
        waybill_img = None
        waybill_detections = []
        im0_final = None
        im_wb = None
        for path, im, im0s, vid_cap, s in dataset:
            im0_final = im0s.copy()
            im = torch.from_numpy(im).to(device).float() / 255.0
            if len(im.shape) == 3:
                im = im[None]
            pred = model_wb(im)
            pred = non_max_suppression(pred, config.CONF_THRES_WAYBILL, config.IOU_THRES_WAYBILL, None, False, max_det=config.MAX_DET)
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0_final.shape).round()
                    waybill_detections = det.clone()
                    waybill_img = self.crop_waybill(im0_final, waybill_detections.cpu().numpy(), margin=config.WAYBILL_CROP_MARGIN)
            break

        im0 = im0_final if im0_final is not None else im0_pkg
        annotator = Annotator(im0, line_width=3)

        if len(pkg_detections):
            pkg_boxes = pkg_detections.clone()
            pkg_boxes[:, :4] = scale_coords(im.shape[2:], pkg_boxes[:, :4], im0.shape).round()
            for *xyxy, conf, cls in pkg_boxes:
                c = int(cls)
                label = f'{model_pkg.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        if len(waybill_detections):
            # waybill_detections 中的坐标已经是原图尺寸，直接使用
            for *xyxy, conf, cls in waybill_detections:
                annotator.box_label(xyxy, f'waybill {conf:.2f}', color=(0, 255, 0))

        im0_annotated = annotator.result()
        resize_scale = self.output_size / im0_annotated.shape[0]
        im0_resized = cv2.resize(im0_annotated, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite(config.TMP_SINGLE_RESULT, im0_resized)
        self.right_img.setPixmap(QPixmap(config.TMP_SINGLE_RESULT))

        name = phone = address = tracking_no = ""
        if waybill_img is not None:
            waybill_path = config.TMP_WAYBILL_CROP
            # 如果裁剪图像尺寸过小，放大至短边 config.OCR_MIN_SIDE 像素
            h, w = waybill_img.shape[:2]
            if min(h, w) < config.OCR_MIN_SIDE:
                scale = config.OCR_MIN_SIDE / min(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                waybill_img = cv2.resize(waybill_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(waybill_path, waybill_img)

            # 通过常驻OCR服务识别（模型只加载一次，进程复用）
            info = self.ocr_client.recognize(waybill_path)
            if info.get('error'):
                print('OCR识别异常:', info['error'])
            name = info.get('name', '')
            phone = info.get('phone', '')
            address = info.get('address', '')
            tracking_no = info.get('tracking_no', '')

        try:
            db.insert_log(self.current_user_id, "图片", pkg_type, name, phone, address, tracking_no)
        except Exception as e:
            print(f"数据库写入失败: {e}")

        if waybill_img is None:
            display_text = f"包裹类型: {pkg_type}\n未检测到面单区域"
        else:
            display_text = f"包裹类型: {pkg_type}\n姓名: {name}\n电话: {phone}\n地址: {address}\n运单号: {tracking_no}"
        self.ocr_result_text.setText(display_text)

    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        self.stopEvent.clear()
        th = threading.Thread(target=self.detect_vid)
        th.daemon = True
        th.start()

    def open_mp4(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            self.stopEvent.clear()
            th = threading.Thread(target=self.detect_vid)
            th.daemon = True
            th.start()

    def close_vid(self):
        self.stopEvent.set()
        self.vid_status_chip.setText('已停止')

    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_stop_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap(config.UI_IMG_VIDEO))
        self.vid_status_chip.setText('待机')
        self.vid_source = '0'
        self.webcam = True

    def update_video_frame(self, pixmap):
        if pixmap.isNull():
            return
        self._video_frames = getattr(self, '_video_frames', 0) + 1
        self.vid_status_chip.setText('检测中')
        self.vid_counter.setText(f'运行中 · 已处理 {self._video_frames} 帧')
        label_size = self.vid_img.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.vid_img.setPixmap(scaled_pixmap)

    def detect_vid(self):
        model = self.model
        output_size = self.output_size
        imgsz = config.IMG_SIZE
        conf_thres = config.CONF_THRES_PACKAGE
        iou_thres = config.IOU_THRES_PACKAGE
        source = str(self.vid_source)
        webcam = self.webcam
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)

        try:
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = len(dataset)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                bs = 1
        except Exception as e:
            LOGGER.error(f'视频流/文件打开失败: {e}')
            self.reset_vid()
            return

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))

        for path, im, im0s, vid_cap, s in dataset:
            if self.stopEvent.is_set():
                break

            im = torch.from_numpy(im).to(device).float() / 255.0
            if len(im.shape) == 3:
                im = im[None]
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            for i, det in enumerate(pred):
                if webcam:
                    im0 = im0s[i].copy()
                else:
                    im0 = im0s.copy()

                annotator = Annotator(im0, line_width=3, example=str(names))
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))

                im0 = annotator.result()
                resize_scale = output_size / im0.shape[0]
                frame_resized = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite(config.TMP_VID_RESULT, frame_resized)
                pixmap = QPixmap(config.TMP_VID_RESULT)
                self.frame_ready.emit(pixmap)

            QApplication.processEvents()

        self.reset_vid()

    def load_history(self):
        """加载当前账号识别记录"""
        logs = db.get_logs_by_user(self.current_user_id)
        self.history_table.setRowCount(len(logs))
        self.history_table.verticalHeader().setDefaultSectionSize(36)  # 行高确保按钮显示完整
        for row_idx, log in enumerate(logs):
            log_id, rec_time, rec_type, pkg_type, name, phone, address, tracking_no = log
            addr_short = (address[:20] + '...') if len(address) > 20 else address
            self.history_table.setItem(row_idx, 0, QTableWidgetItem(str(log_id)))
            self.history_table.setItem(row_idx, 1, QTableWidgetItem(rec_time))
            self.history_table.setItem(row_idx, 4, QTableWidgetItem(name or ''))
            self.history_table.setItem(row_idx, 5, QTableWidgetItem(phone or ''))
            self.history_table.setItem(row_idx, 6, QTableWidgetItem(addr_short or ''))
            self.history_table.setCellWidget(row_idx, 2, ui_style.wrapped_chip(rec_type or '图片'))
            pkg_tone = 'amber' if pkg_type in ('纸箱', '文件封', '塑料袋') else ''
            self.history_table.setCellWidget(row_idx, 3, ui_style.wrapped_chip(pkg_type or '未知', pkg_tone))
            ops = ui_style.cell_buttons([
                ('详情', '', lambda checked, l=log: self.show_log_detail(l)),
                ('删除', 'danger', lambda checked, lid=log_id: self.delete_log_by_id(lid)),
            ])
            self.history_table.setCellWidget(row_idx, 7, ops)

        self.history_count.setText(f'共 {len(logs)} 条 · 显示最近 100 条')
        if getattr(self, '_hist_nav_index', None) is not None:
            self._nav_list.item(self._hist_nav_index).setText(f'历史记录 {len(logs)}')

    def show_log_detail(self, log):
        log_id, rec_time, rec_type, pkg_type, name, phone, address, tracking_no = log
        detail_text = f"""
        识别时间: {rec_time}
        识别类型: {rec_type}
        包裹类型: {pkg_type}
        姓名: {name}
        电话: {phone}
        地址: {address}
        运单号: {tracking_no}
        """
        ui_style.msg_info(self, "识别详情", detail_text)

    def delete_log_by_id(self, log_id):
        if ui_style.msg_confirm(self, "确认删除", "确定要删除这条记录吗？"):
            if db.delete_log(log_id, self.current_user_id):
                self.load_history()
            else:
                ui_style.msg_warning(self, "错误", "删除失败，请稍后重试")

    def closeEvent(self, event):
        if ui_style.msg_confirm(self, '退出', '确定要退出吗？'):
            self.ocr_client.shutdown()  # 停止常驻OCR服务进程
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(ui_style.GLOBAL_QSS)
    db.init_db()  # 登录前初始化表结构并确保默认管理员存在（幂等）
    login_dialog = LoginDialog()
    if login_dialog.exec_() != QDialog.Accepted:
        sys.exit(0)
    user_id = login_dialog.user_id
    mainWindow = MainWindow(user_id=user_id)
    mainWindow.show()
    sys.exit(app.exec_())