import time
import pydirectinput
import threading

# 定义一个全局标志
stop_flag = False
def thread_function():
    global stop_flag
    while not stop_flag:
        # 执行一些操作
        pydirectinput.click()
    print("Thread is terminating.")
# 创建线程
one_thread = threading.Thread(target=thread_function)
two_thread = threading.Thread(target=thread_function)
# 启动线程
one_thread.start()
two_thread.start()
# 主线程等待一段时间
time.sleep(60)
# 设置停止标志，以终止while循环
stop_flag = True
# 等待线程结束
one_thread.join()
two_thread.join()
print("Main thread exiting.")

