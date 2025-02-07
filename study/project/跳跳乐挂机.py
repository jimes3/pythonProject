import pyautogui
import pydirectinput
import time
import os
import sys
pyautogui.PAUSE = 0

# 获取图片的绝对路径
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def repeat_keys(keys):
    try:
        a = time.time()
        bb = time.time()
        while a-bb+660>0:
            for key in keys:
                pyautogui.keyDown(key)
                time.sleep(1)
                pyautogui.keyUp(key)
                time.sleep(9)
                bb = time.time()
    except KeyboardInterrupt:
        print("Stopped by user")

# 4：3比例 + 无边框窗口
while True:
    try:
        left, top, width, height = pyautogui.locateOnScreen(get_resource_path('./source/sta1.png'),confidence=0.9,grayscale=True)   # 寻找图片
        center = pyautogui.center((left, top, width, height))    # 寻找图片的中心
        print('开始',center)
        pydirectinput.moveTo(center[0],center[1])
        pydirectinput.click()
        repeat_keys(['w', 'a', 's', 'd'])
    except TypeError:
        try:
            left1, top1, width1, height1 = pyautogui.locateOnScreen(get_resource_path('./source/bac.png'),confidence=0.8,grayscale=True)   # 寻找图片
            center1 = pyautogui.center((left1, top1, width1, height1))    # 寻找图片的中心
            print('返回',center1)
            pydirectinput.moveTo(center1[0],center1[1])
            pydirectinput.click()
        except TypeError:
            print('无')
            time.sleep(2)