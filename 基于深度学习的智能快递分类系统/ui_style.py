# -*- coding: utf-8 -*-
"""
界面主题与通用 UI 组件（纯外观层，不含任何业务逻辑）

集中管理配色、字体、QSS 与通用控件工厂（色标签、单元格按钮、筛选 chip、
中文消息框等），window.py 只负责布局结构，具体视觉效果全部来自这里。
所有风格参数与界面美化参考稿保持一致。
"""
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap, QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import (QLabel, QPushButton, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem, QFrame, QSizePolicy)

# ---------------- 主题常量（与参考稿配色一致） ----------------
PAGE_BG = '#F4F6FA'
SURFACE = '#FFFFFF'
INK = '#101828'
MUTED = '#667085'
LINE = '#E4E7EC'
BRAND = '#2B5CE6'
BRAND_DARK = '#1E47B8'
NAVY = '#0E1B3D'
AMBER = '#F59E0B'
GREEN = '#12915A'
RED = '#E5484D'

FONT_FAMILY = 'Microsoft YaHei UI'
FONT_MONO = 'Consolas'

FONT_DEFAULT = QFont(FONT_FAMILY, 10)        # 14px
FONT_SMALL = QFont(FONT_FAMILY, 9)           # 12px
FONT_TITLE = QFont(FONT_FAMILY, 11, QFont.Bold)  # 16px
FONT_BIG = QFont(FONT_FAMILY, 12, QFont.Bold)    # 18px


# ---------------- 全局样式表 ----------------
GLOBAL_QSS = """
* { outline: none; }
QWidget {
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "SimHei", sans-serif;
    font-size: 14px;
    color: #101828;
}
QMainWindow, QDialog { background: #F4F6FA; }

/* ---------- 页面标题与卡片头 ---------- */
QLabel#pageTitle { font-size: 16px; font-weight: 600; color: #101828; }
QLabel#pageSub { font-size: 12px; color: #667085; }
QLabel#cardHeader { font-size: 13px; font-weight: 500; color: #101828; }
QLabel#cardSide {
    font-size: 11px; color: #667085;
    font-family: "Consolas", "Microsoft YaHei UI";
}

/* ---------- 左侧导航 ---------- */
QWidget#side { background-color: #0E1B3D; }
QLabel#brandLogo {
    background-color: #F59E0B; color: #3A2400;
    border-radius: 8px; font-size: 13px; font-weight: 700;
}
QLabel#brandName { color: #FFFFFF; font-size: 13px; font-weight: 500; }
QLabel#userAva {
    background-color: #2E3E66; color: #FFFFFF;
    border-radius: 14px; font-size: 12px; font-weight: 600;
}
QLabel#userName { color: #8E9CBB; font-size: 11px; }
QLabel#userRole { color: #FFFFFF; font-size: 12px; font-weight: 500; }
QListWidget#nav { background: transparent; border: none; padding: 6px 4px; }
QListWidget#nav::item {
    color: #C3CDE0; background: transparent;
    border-radius: 8px; padding: 8px 12px; font-size: 13px;
}
QListWidget#nav::item:hover { background: rgba(255,255,255,0.08); color: #FFFFFF; }
QListWidget#nav::item:selected { background: #2B5CE6; color: #FFFFFF; }

/* ---------- 卡片 / 图像框 / 统计块 ---------- */
QFrame#card { background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 12px; }
QWidget#statTile { background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 10px; }
QLabel#statNum { font-size: 18px; font-weight: 700; color: #101828; }
QLabel#statLabel { font-size: 11px; color: #667085; }
QLabel#imageFrame {
    background: #EAEEF4; border: 1px solid #E4E7EC; border-radius: 9px;
}

/* ---------- 按钮：主 / 次 / 危险（用动态属性 variant 区分） ---------- */
QPushButton {
    background: #2B5CE6; color: #FFFFFF; border: 1px solid #2B5CE6;
    border-radius: 8px; padding: 7px 16px; font-size: 13px;
}
QPushButton:hover { background: #1E47B8; border-color: #1E47B8; }
QPushButton:pressed { background: #173AA0; border-color: #173AA0; }
QPushButton:disabled { background: #EEF0F4; color: #98A2B3; border-color: #EEF0F4; }

QPushButton[variant="ghost"] { background: #FFFFFF; color: #101828; border: 1px solid #E4E7EC; }
QPushButton[variant="ghost"]:hover { border-color: #2B5CE6; color: #2B5CE6; background: #FFFFFF; }

QPushButton[variant="danger"] { background: #FFFFFF; color: #E5484D; border: 1px solid #E5484D; }
QPushButton[variant="danger"]:hover { background: #FDECEC; }
QPushButton[variant="danger"]:disabled { background: #EEF0F4; color: #98A2B3; border-color: #EEF0F4; }

QPushButton#loginPrimary { padding: 10px 16px; font-size: 14px; border-radius: 9px; }

/* 表格单元格内的小按钮 */
QPushButton#cellBtn {
    background: transparent; color: #344054;
    border: 1px solid #E4E7EC; border-radius: 6px;
    padding: 3px 9px; font-size: 12px;
}
QPushButton#cellBtn:hover { border-color: #2B5CE6; color: #2B5CE6; }
QPushButton#cellBtn[variant="danger"] { color: #E5484D; border-color: #F6C9CB; }
QPushButton#cellBtn[variant="danger"]:hover { background: #FDECEC; }

/* ---------- 输入框与多行文本 ---------- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox {
    background: #FFFFFF; border: 1px solid #E4E7EC;
    border-radius: 8px; padding: 8px 10px;
    selection-background-color: #BFD0FC;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus { border: 1px solid #2B5CE6; }
QLineEdit#loginField { padding: 9px 12px; border-radius: 9px; }

QTextEdit#ocrText {
    background: #F7F9FC; color: #475467;
    border: 1px dashed #E4E7EC; border-radius: 8px;
    font-family: "Consolas", "Microsoft YaHei UI"; font-size: 12px;
}

/* ---------- 表格 ---------- */
QTableWidget {
    background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 10px;
    gridline-color: transparent; outline: none;
}
QTableWidget::item { border-bottom: 1px solid #F1F3F7; padding: 6px 8px; }
QTableWidget::item:selected { background: #EAF0FF; color: #101828; }
QHeaderView::section {
    background: #FFFFFF; color: #667085; font-weight: 600; font-size: 12px;
    border: none; border-bottom: 1px solid #E4E7EC; padding: 9px;
}
QTableCornerButton::section { background: #FFFFFF; border: none; }

/* ---------- 色标签 chip ---------- */
QLabel#chip {
    background: #F1F3F7; color: #475467;
    border-radius: 9px; padding: 2px 8px; font-size: 11px;
}
QLabel#chip[tone="amber"] { background: #FEF3E2; color: #B45309; }
QLabel#chip[tone="blue"] { background: #EAF0FF; color: #1D4ED8; }
QLabel#chip[tone="green"] { background: #E7F6EE; color: #12915A; }
QLabel#chip[tone="red"] { background: #FDECEC; color: #E5484D; }

/* ---------- 筛选 chip 按钮（全部 / 成功 / 失败） ---------- */
QPushButton#filterChip {
    background: #FFFFFF; color: #475467;
    border: 1px solid #E4E7EC; border-radius: 999px;
    padding: 4px 12px; font-size: 12px;
}
QPushButton#filterChip:hover { border-color: #2B5CE6; color: #2B5CE6; }
QPushButton#filterChip[on="true"] {
    background: #EAF0FF; color: #1D4ED8; border-color: #C7D7FA; font-weight: 500;
}

/* ---------- 状态栏文本 ---------- */
QLabel#statusText {
    font-size: 11px; color: #667085;
    font-family: "Consolas", "Microsoft YaHei UI";
}
QLabel#statusChip { font-size: 11px; color: #475467; }

/* ---------- 对话框通用 ---------- */
QWidget#dlgBottom { background: #FAFBFC; }
QWidget#brandPanel { background-color: #0E1B3D; }
QLabel#brandSlogan { font-size: 20px; font-weight: 700; color: #FFFFFF; }
QLabel#brandInfo { font-size: 12px; color: #8E9CBB; }
QLabel#brandVersion {
    font-size: 11px; color: #6C7B99;
    font-family: "Consolas", "Microsoft YaHei UI";
}
QLabel#loginFormTitle { font-size: 18px; font-weight: 700; color: #101828; }
QLabel#loginFormSub { font-size: 12px; color: #667085; }
QLabel#loginSwitch { font-size: 12px; color: #667085; }
QLabel#dlgTitle { font-size: 14px; font-weight: 700; color: #101828; }
QLabel#dlgSub { font-size: 11px; color: #667085; }
QLabel#dlgFieldLabel { font-size: 12px; color: #344054; font-weight: 500; }
QLabel#dlgHint { font-size: 11px; color: #667085; }
QMessageBox { background: #FFFFFF; }
QMessageBox QLabel { font-size: 13px; }

/* ---------- 滚动条 ---------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #C9CFDA; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #AEB6C4; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #C9CFDA; border-radius: 5px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #AEB6C4; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
"""


# ---------------- 通用控件工厂 ----------------

def set_variant(widget, variant):
    """切换控件动态属性并重新应用样式（如按钮的主/次/危险分级）"""
    if variant:
        widget.setProperty('variant', variant)
    else:
        widget.setProperty('variant', None)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


def chip(text, tone=''):
    """色标签 QLabel；tone 取 '' / 'amber' / 'blue' / 'green' / 'red'"""
    label = QLabel(text)
    label.setObjectName('chip')
    label.setAlignment(Qt.AlignCenter)
    if tone:
        label.setProperty('tone', tone)
    return label


def wrapped_chip(text, tone=''):
    """表格单元格用的色标签（带上下留白，随行高垂直居中）"""
    cell = QWidget()
    layout = QHBoxLayout(cell)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(0)
    layout.addWidget(chip(text, tone))
    layout.addStretch(1)
    return cell


def cell_buttons(buttons):
    """表格行内操作按钮组。buttons: [(text, variant, slot 或 None)]
    variant 取 '' / 'danger'；slot 为 None 时按钮置灰不可用。"""
    cell = QWidget()
    layout = QHBoxLayout(cell)
    layout.setContentsMargins(0, 2, 0, 2)
    layout.setSpacing(6)
    for text, variant, slot in buttons:
        btn = QPushButton(text)
        btn.setObjectName('cellBtn')
        btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        if variant:
            btn.setProperty('variant', variant)
        if slot is None:
            btn.setEnabled(False)
        else:
            btn.clicked.connect(slot)
        layout.addWidget(btn)
    layout.addStretch(0)
    return cell


def stat_tile(num, label_text):
    """概览数字块"""
    frame = QWidget()
    frame.setObjectName('statTile')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 10, 14, 10)
    layout.setSpacing(2)
    num_label = QLabel(str(num))
    num_label.setObjectName('statNum')
    text_label = QLabel(label_text)
    text_label.setObjectName('statLabel')
    layout.addWidget(num_label)
    layout.addWidget(text_label)
    return frame


def nav_icon(kind):
    """绘制 18x18 的侧边导航图标（image / video / clock / users / log）"""
    pm = QPixmap(18, 18)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor('#FFFFFF'))
    pen.setWidthF(1.5)
    pen.setStyle(Qt.SolidLine)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

    if kind == 'image':
        p.drawRoundedRect(QRectF(2, 3, 14, 11), 2, 2)
        p.drawEllipse(QRectF(5.5, 6.3, 2.6, 2.6))
        p.drawLine(QPointF(3, 13), QPointF(6.3, 9.4))
        p.drawLine(QPointF(6.3, 9.4), QPointF(8.8, 11.4))
        p.drawLine(QPointF(8.8, 11.4), QPointF(11, 9.4))
        p.drawLine(QPointF(11, 9.4), QPointF(15, 13))
    elif kind == 'video':
        p.drawRoundedRect(QRectF(2, 3.5, 9, 9), 2, 2)
        pp = QPen(QColor('#FFFFFF'))
        pp.setWidthF(1.5)
        pp.setJoinStyle(Qt.RoundJoin)
        p.setPen(pp)
        p.drawPolygon([QPointF(11, 7.2), QPointF(15.2, 5), QPointF(15.2, 11), QPointF(11, 8.8)])
    elif kind == 'clock':
        p.drawEllipse(QRectF(2.4, 2.4, 11.2, 11.2))
        p.drawLine(QPointF(8, 5.4), QPointF(8, 8))
        p.drawLine(QPointF(8, 8), QPointF(10.3, 9.2))
    elif kind == 'users':
        p.drawEllipse(QRectF(4.2, 3.6, 3.6, 3.6))
        p.drawEllipse(QRectF(11, 4.8, 2.2, 2.2))
        p.drawPath(_arch_path(QRectF(4.6, 7.2, 4.4, 4.2)))
        p.drawLine(QPointF(4.8, 11.4), QPointF(9.4, 11.4))
        p.drawPath(_arch_path(QRectF(9.6, 8.4, 4.2, 4.2)))
        p.drawLine(QPointF(9.8, 12), QPointF(14.2, 12))
    elif kind == 'log':
        p.drawRoundedRect(QRectF(3, 3, 10, 10), 1.6, 1.6)
        p.drawLine(QPointF(5.6, 6.6), QPointF(11, 6.6))
        p.drawLine(QPointF(5.6, 9.2), QPointF(8.6, 9.2))
    p.end()
    return QIcon(pm)


def _arch_path(rect):
    """用 QPainterPath 绘制一段拱形，用于 user 图标"""
    from PyQt5.QtGui import QPainterPath
    path = QPainterPath()
    path.moveTo(rect.left(), rect.top())
    path.cubicTo(rect.left() - rect.width() * 0.2, rect.bottom(),
                 rect.right() + rect.width() * 0.2, rect.bottom(),
                 rect.right(), rect.top())
    return path


# ---------------- 中文消息框（外观层，不改判断逻辑） ----------------

def msg_info(parent, title, text):
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Information)
    box.setText(text)
    ok = box.addButton('确定', QMessageBox.AcceptRole)
    box.setDefaultButton(ok)
    box.exec_()


def msg_warning(parent, title, text):
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Warning)
    box.setText(text)
    ok = box.addButton('确定', QMessageBox.AcceptRole)
    box.setDefaultButton(ok)
    box.exec_()


def msg_confirm(parent, title, text):
    """确定的操作弹窗：返回 True 表示用户确认，取消/关闭返回 False"""
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Question)
    box.setText(text)
    ok = box.addButton('确定', QMessageBox.YesRole)
    cancel = box.addButton('取消', QMessageBox.NoRole)
    set_variant(cancel, 'ghost')
    box.setDefaultButton(cancel)
    box.exec_()
    return box.clickedButton() is ok