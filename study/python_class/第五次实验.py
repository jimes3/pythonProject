import pandas as pd
import tushare as ts
import os

##############################################################
# 文件路径
file_path1 = 'open_count1.csv'
try:
    # 尝试读取已有文件
    df = pd.read_csv(file_path1)
except FileNotFoundError:
    # 如果文件不存在，创建一个新的DataFrame
    df = pd.DataFrame({'Open_Count': [0]})
# 获取当前打开次数
if 'Open_Count' not in df.columns:
    df['Open_Count'] = 0
open_count = df['Open_Count'][0]
# 增加打开次数
open_count += 1
# 更新DataFrame中的数据
df['Open_Count'] = open_count
# 保存新的打开次数到文件中
df.to_csv(file_path1,index=False)
# 显示新的打开次数
print(f"这是第 {open_count} 次打开文件。")

##############################################################
# 文件路径
file_path2 = 'open_count2.csv'
# 检查文件是否存在
if os.path.exists(file_path2):
    # 如果文件存在，读取文件中的打开次数
    df = pd.read_csv(file_path2)
    if 'Open_Count' not in df.columns:
        df['Open_Count'] = 0
    open_count = df['Open_Count'][0]
else:
    # 如果文件不存在，则初始化计数为0
    df = pd.DataFrame({'Open_Count': [0]})
# 获取当前打开次数
open_count = df['Open_Count'][0]
# 增加打开次数
open_count += 1
# 更新DataFrame中的数据
df['Open_Count'] = open_count
# 保存新的打开次数到文件中
df.to_csv(file_path2,index=False)
# 显示新的打开次数
print(f"这是第 {open_count} 次打开文件。")
#########################################################
# 学生学号列表
student_ids = ["2021001", "2021002", "2021003", "2021004"]
# 指定父文件夹路径
parent_folder = "学生文件夹"
# 使用循环一次性创建所有文件夹
for student_id in student_ids:
    folder_path = os.path.join(parent_folder, student_id)
    os.makedirs(folder_path, exist_ok=True)
print(f"一次性创建了{len(student_ids)}个文件夹。")

################################################
# 学生学号列表
student_ids = ["2021001", "2021002", "2021003", "2021004"]
# 指定父文件夹路径
parent_folder = "学生文件夹"
# 创建文件夹的函数
def create_student_folder(student_id):
    folder_path = os.path.join(parent_folder, student_id)
    os.makedirs(folder_path, exist_ok=True)
    print(f"创建文件夹 {folder_path}")
# 逐个创建文件夹
for student_id in student_ids:
    create_student_folder(student_id)

#####################################################
# 设置你的tushare API token
ts.set_token('7cbf4beb72d223898ffafdb826d7cb0879a3f372689e214461274361')
# 初始化tushare pro接口
pro = ts.pro_api()
# 指定股票代码和查询日期范围
stock_code = '600000.SH'  # 例如，这里使用中国平安的股票代码
start_date = '20230101'
end_date = '20231010'
# 获取股票交易数据
df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
# 计算盈利额（以收盘价和开盘价为基础）
df['Profit'] = (df['close'] - df['open']) * 1000  # 1000股为一手
# 打印结果
print(df[['trade_date', 'open', 'close', 'Profit']])
