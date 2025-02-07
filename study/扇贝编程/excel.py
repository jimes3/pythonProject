import yagmail
import csv

filenames = ['2019-12-%02d-销售数据.csv' % (i + 1) for i in range(31)]

with open('12月销售数据汇总.csv', 'w', newline='') as file:
  csv_writer = csv.writer(file)

  for filename in filenames:
    with open(filename, newline='') as file:
      csv_reader = csv.reader(file)

      if filename == filenames[0]:
        rows = csv_reader
      else:
        rows = list(csv_reader)[1:]
      csv_writer.writerows(rows)


with open('12月销售计算数据汇总.csv', 'w', newline='') as file:
  csv_writer = csv.writer(file)

  with open('12月销售数据汇总.csv', newline='') as file:
    csv_reader = csv.reader(file)

    for index, row in enumerate(csv_reader):
      if index == 0:  # 第一个是表头
        csv_writer.writerow(row + ['购买转化率', '客单价'])  # 添加两个新表头
      else:
        visitors = int(row[2])  # 访客数
        buyers = int(row[3])  # 买家数
        gmv = int(row[4])  # 交易额
        sale_rate = buyers / visitors if visitors else 0  # 购买转化率
        pct = gmv / buyers if buyers else 0  # 客单价
        csv_writer.writerow(row + [sale_rate, pct])  # 添加购买转化率和客单价

user = 'xxxxxx@qq.com'  # 发件人邮箱
password = 'xxxxxxxxxxxxxxxx'  # 授权码
host = 'smtp.qq.com'  # smtp 服务器地址
to = ['xxxxxx@qq.com']  # 收件人邮箱列表
subject = '12月销售计算数据汇总'  # 邮件主题
contents = ['统计了 12 月的销售数据，请查收~', 'D:\\excel\\12月销售计算数据汇总.csv']  # 邮件正文

yag = yagmail.SMTP(user=user, password=password, host=host)
yag.send(to=to, subject=subject, contents=contents)