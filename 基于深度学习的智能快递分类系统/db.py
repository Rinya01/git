# -*- coding: utf-8 -*-
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import config

DB_PATH = config.DB_PATH
config.ensure_dirs()

# 默认管理员账户（首次启动自动创建，建议登录后修改密码）
DEFAULT_ADMIN_USER = 'admin'
DEFAULT_ADMIN_PWD = 'admin123'

def init_db():
    """初始化数据库，创建用户表、识别日志表和登录日志表"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 用户表（role: 'admin' 管理员 / 'user' 普通用户）
    c.execute('''
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            reg_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    # 旧库迁移：为已存在的用户表补充 role 列
    c.execute('PRAGMA table_info(user)')
    if 'role' not in [row[1] for row in c.fetchall()]:
        c.execute("ALTER TABLE user ADD COLUMN role TEXT DEFAULT 'user'")
    # 识别日志表
    c.execute('''
        CREATE TABLE IF NOT EXISTS recognition_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            rec_type TEXT NOT NULL,
            package_type TEXT,
            name TEXT,
            phone TEXT,
            address TEXT,
            tracking_no TEXT,
            rec_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES user(id)
        )
    ''')
    # 登录日志表（user_id 允许为空：登录失败时可能对应不存在的用户名）
    c.execute('''
        CREATE TABLE IF NOT EXISTS login_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            success INTEGER NOT NULL DEFAULT 1,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES user(id)
        )
    ''')
    # 创建组合索引
    c.execute('CREATE INDEX IF NOT EXISTS idx_user_time ON recognition_log(user_id, rec_time)')
    # 迁移已有的明文密码为加盐哈希
    _migrate_plaintext_passwords(conn, c)
    conn.commit()
    conn.close()
    # 确保存在管理员账户
    _ensure_admin()

def _ensure_admin():
    """确保系统内至少存在一个管理员账户：
    1) 已有管理员则不做处理；
    2) 存在名为 admin 的普通用户，将其提升为管理员；
    3) 否则创建默认管理员 admin/admin123（建议登录后修改密码）。"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM user WHERE role='admin'")
    if c.fetchone()[0] == 0:
        c.execute("SELECT id FROM user WHERE username=?", (DEFAULT_ADMIN_USER,))
        row = c.fetchone()
        if row:
            c.execute("UPDATE user SET role='admin' WHERE id=?", (row[0],))
            print(f'[系统提示] 已将用户 {DEFAULT_ADMIN_USER} 提升为管理员')
        else:
            c.execute("INSERT INTO user (username, password, role) VALUES (?, ?, 'admin')",
                      (DEFAULT_ADMIN_USER, generate_password_hash(DEFAULT_ADMIN_PWD)))
            print(f'[系统提示] 已创建默认管理员账户 {DEFAULT_ADMIN_USER}/{DEFAULT_ADMIN_PWD}（建议登录后修改密码）')
        conn.commit()
    conn.close()

# 已哈希密码的已知前缀：以此类前缀开头的视为已加密，跳过迁移。
# 注意：新版 werkzeug（>=2.4）默认生成 'scrypt:' 前缀的哈希，必须包含在内，
# 否则每次启动都会把已有哈希误判为明文并二次加密，导致所有账户无法登录。
_HASHED_PREFIXES = ('pbkdf2:', 'scrypt:', '$2b$', '$2a$', '$2y$', '$argon2', 'md5$', 'sha1$', 'sha256$')


def _migrate_plaintext_passwords(conn, c):
    """将数据库中已有的明文密码迁移为werkzeug加盐哈希值"""
    try:
        rows = c.execute('SELECT id, password FROM user').fetchall()
        for uid, pwd in rows:
            if not pwd.startswith(_HASHED_PREFIXES):
                c.execute('UPDATE user SET password=? WHERE id=?',
                          (generate_password_hash(pwd), uid))
        conn.commit()
    except Exception:
        # 迁移失败不影响应用启动
        pass
# db.py 新增以下函数

def register_user(username, password):
    """注册新用户，密码加盐哈希后存储；成功返回True，用户名已存在返回False"""
    password_hash = generate_password_hash(password)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO user (username, password) VALUES (?, ?)', (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    """验证登录并写入登录日志（成功/失败均记录）；成功返回用户ID，失败返回None"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, password FROM user WHERE username=?', (username,))
    row = c.fetchone()
    success = bool(row and check_password_hash(row[1], password))
    # 记录登录日志：失败登录时 user_id 为空，仅保留用户名快照
    c.execute('INSERT INTO login_log (user_id, username, success) VALUES (?, ?, ?)',
              (row[0] if row else None, username, 1 if success else 0))
    if success:
        user_id = row[0]
        # 更新最后登录时间
        c.execute('UPDATE user SET last_login=CURRENT_TIMESTAMP WHERE id=?', (user_id,))
    conn.commit()
    conn.close()
    return row[0] if success else None
def insert_log(user_id, rec_type, package_type, name, phone, address, tracking_no):
    """插入一条识别日志"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO recognition_log 
        (user_id, rec_type, package_type, name, phone, address, tracking_no)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, rec_type, package_type, name, phone, address, tracking_no))
    conn.commit()
    conn.close()

def get_logs_by_user(user_id, limit=100):
    """获取指定用户的识别日志，按时间倒序"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, rec_time, rec_type, package_type, name, phone, address, tracking_no
        FROM recognition_log
        WHERE user_id = ?
        ORDER BY rec_time DESC
        LIMIT ?
    ''', (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return rows

def delete_log(log_id, user_id):
    """删除指定日志（同时校验用户ID防止误删）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM recognition_log WHERE id = ? AND user_id = ?', (log_id, user_id))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

# ================= 管理员功能 =================

def get_user_role(user_id):
    """获取用户角色：'admin' 或 'user'（用户不存在时返回 'user'）"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT role FROM user WHERE id=?', (user_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row and row[0] else 'user'

def get_all_users():
    """获取全部用户：(id, username, role, reg_time, last_login)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, username, role, reg_time, last_login FROM user ORDER BY id')
    rows = c.fetchall()
    conn.close()
    return rows

def update_user(user_id, new_username=None, new_password=None):
    """修改用户信息（用户名 / 密码，None 表示不修改）；成功返回 True。
    新用户名与其他用户冲突或两项均未填写时返回 False。"""
    if not new_username and not new_password:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if new_username:
        c.execute('SELECT COUNT(*) FROM user WHERE username=? AND id!=?', (new_username, user_id))
        if c.fetchone()[0] > 0:
            conn.close()
            return False
        c.execute('UPDATE user SET username=? WHERE id=?', (new_username, user_id))
    if new_password:
        c.execute('UPDATE user SET password=? WHERE id=?',
                  (generate_password_hash(new_password), user_id))
    conn.commit()
    conn.close()
    return True

def delete_user(user_id):
    """删除用户及其识别日志（登录日志保留，含用户名快照）；成功返回 True"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM recognition_log WHERE user_id=?', (user_id,))
    c.execute('DELETE FROM user WHERE id=?', (user_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def get_login_logs(limit=200):
    """获取登录日志（全部用户，时间倒序）：(id, username, login_time, success)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, username, login_time, success
        FROM login_log
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
