import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

wb = Workbook()
sheet = wb.active
sheet.title = '豆瓣图书Top250'
# 写入表头
header = ['书名', '评分', '链接']
sheet.append(header)

headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'}
res = requests.get('https://book.douban.com/top250/', headers=headers)
soup = BeautifulSoup(res.text, 'html.parser')
# 通过 class 属性选中所有包含图书信息的整个 HTML 元素
items = soup.select('.item')
# 遍历每个元素，即遍历每本图书
for i in items:
  tag = i.select('div.pl2 a')[0] # 选中包含书名、书籍链接的 a 元素
  rating = i.select('.rating_nums')[0].text # 书籍评分
  name = tag['title'] # 书名
  link = tag['href'] # 书籍链接
  # 写入一行数据
  row = [name, rating, link]
  sheet.append(row)
  print(name, rating, link)

# 保存为 豆瓣.xlsx
wb.save('豆瓣.xlsx')