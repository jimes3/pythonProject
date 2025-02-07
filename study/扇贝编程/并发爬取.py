import requests
from concurrent import futures
from bs4 import BeautifulSoup

# 解析列表页内容，得到这一页的内容链接
def parse_list_page(text):
  soup = BeautifulSoup(text, 'html.parser')
  tbody = soup.select('table.board-list.tiz > tbody')[0]
  urls = []
  for tr in tbody:
    td = tr.select('td.title_9')[0]
    urls.append(td.a.attrs['href'])
  return urls

# 解析内容页，得到标题和所有帖子内容
def parse_content_page(text):
  soup = BeautifulSoup(text, 'html.parser')

  title = soup.select('span.n-left')[0].text.strip('文章主题:').strip()

  content_div = soup.select('div.b-content.corner')[0]
  contents = []
  for awrap in content_div.select('div.a-wrap.corner'):
    content = awrap.p.text
    contents.append(content)
  return title, contents

# 把列表页得到的链接转换成我们要抓取的链接
def convert_content_url(path):
  URL_PREFIX = 'http://www.mysmth.net'
  path += '?ajax'
  return URL_PREFIX + path

# 生成前十页的链接
list_urls = []
for i in range(1, 11):
  url = 'http://www.mysmth.net/nForum/board/Python?ajax&p='
  url += str(i)
  list_urls.append(url)

session = requests.Session()
executor = futures.ThreadPoolExecutor(max_workers=5)

# 这个函数获取列表页数据，解析出链接，并转换成真实链接
def get_content_urls(list_url):
  res = session.get(list_url)
  content_urls = parse_list_page(res.text)
  real_content_urls = []
  for url in content_urls:
    url = convert_content_url(url)
    real_content_urls.append(url)
  return real_content_urls

print('请等待，程序正在运行。')

# 根据刚刚生成的十个列表页链接，开始提交任务
fs = []
for list_url in list_urls:
  fs.append(executor.submit(get_content_urls, list_url))
futures.wait(fs)

content_urls = set()
for f in fs:
  for url in f.result():
    content_urls.add(url)

# 获取正文页内容，解析出标题和帖子
def get_content(url):
  r = session.get(url)
  title, contents = parse_content_page(r.text)
  return title, contents

# 提交解析正文的任务
fs = []
for url in content_urls:
  fs.append(executor.submit(get_content, url))
futures.wait(fs)

results = {}
for f in fs:
  title, contents = f.result()
  results[title] = contents

for i in results.keys():
  print(i)