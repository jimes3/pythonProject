# 导入 requests 库
import requests
# 从 bs4 库导入 BeautifulSoup
from bs4 import BeautifulSoup
import time
# 将获取一页图书数据代码封装成函数 get_one_page_data()
def get_one_page_data(page):
    # 豆瓣读书 Top 250 首页 URL
    url = 'https://book.douban.com/top250/'
    # 定制消息头
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 SLBrowser/8.0.0.7062 SLBChan/103'}
    # 根据传入参数定制查询参数
    params = {'start': (page - 1) * 25}
    # 发送带消息头和查询参数的请求
    res = requests.get(url, headers=headers, params=params)
    # 解析成 BeautifulSoup 对象
    soup = BeautifulSoup(res.text, 'html.parser')
    # 提取出书名、作者、出版社信息并按行打印
    # 所有书名所在元素
    book_name = soup.select('div.pl2 a')
    # 所有书籍信息所在元素
    book_info = soup.select('p.pl')
    # 遍历每本图书
    for i in range(len(book_name)):
        # 通过元素title属性提取书名
        name = book_name[i]['title']
        # 获取书籍信息
        info = book_info[i].text
        # 按‘/’分割字符串
        info_list = info.split('/')
        # 结果列表中第一项为作者信息
        author = info_list[0]
        # 倒数第三项为出版社信息
        publisher = info_list[-3]
        # 打印
        print(name, author, publisher)
# 循环 10 次，分别获取第 1～10 页数据
for i in range(1, 11):
    get_one_page_data(i)
    time.sleep(1)