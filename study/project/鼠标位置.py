import ctypes
import time
from ctypes import wintypes
# 获取鼠标位置
def get_mouse_position():
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

# 锁定鼠标到屏幕中心
def lock_mouse_to_center():
    ctypes.windll.user32.SetCursorPos(1023, 768)

# 检测鼠标移动
def detect_mouse_movement():
    center_x, center_y = 1023, 768
    while True:
        x, y = get_mouse_position()
        if x != center_x or y != center_y:
            print('鼠标移动到 ({}, {})'.format(x, y))
            lock_mouse_to_center()  # 锁定鼠标到屏幕中心

        time.sleep(0.1)

if __name__ == "__main__":
    lock_mouse_to_center()
    detect_mouse_movement()
