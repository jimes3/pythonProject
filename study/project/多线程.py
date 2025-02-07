import threading
import pyautogui
import time
import keyboard
import cv2
import numpy as np
pyautogui.PAUSE = 0
# 定义线程函数
def print_numbers():
    for i in range(1, 100):
        print("Number:", i)
def print_letters():
    for letter in 'abcde':
        print("Letter:", letter)
# 创建线程对象
numbers_thread = threading.Thread(target=print_numbers)
letters_thread = threading.Thread(target=print_letters)
# 启动线程
numbers_thread.start()
letters_thread.start()
# 等待线程完成（可选）
numbers_thread.join()
letters_thread.join()
print("主线程结束")

def key_press(event):
    if event.name == 'x':
        prev_gray = None
        while True:
            a = time.time()
            screenshot = pyautogui.screenshot(region=(1269, 770, 21, 61))
            b = time.time()
            # 将截屏转换为灰度
            gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            # 第一次循环不计算MSE
            if prev_gray is None:
                prev_gray = gray
                continue
            c = time.time()
            # 计算均方误差
            mse = np.mean((gray - prev_gray) ** 2)
            d = time.time()
            print(d-a)
            if mse <= 1:
                continue
            else:
                pyautogui.click()  # 点击
                pyautogui.mouseUp()  # 释放
                break
            prev_gray = gray
keyboard.on_press(key_press)
# 设置"Enter"键为按键
keyboard.add_hotkey('x',lambda:None, suppress=True)
# 进入事件循环
keyboard.wait()