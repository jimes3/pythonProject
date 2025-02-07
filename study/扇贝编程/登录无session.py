import requests
from bs4 import BeautifulSoup

headers = {
  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'
}
# 登录参数
login_data = {
  'log': 'codetime',
  'pwd': 'shanbay520',
  'wp-submit': '登录',
  'redirect_to': 'https://wpblog.x0y1.com',
  'testcookie': '1'
}

# 发请求登录
login_req = requests.post('https://wpblog.x0y1.com/wp-login.php', data=login_data, headers=headers)

# 获取登录后的 cookies
shared_cookies = login_req.cookies

# 将登录后的 cookies 传递给 cookies 参数用于获取文章页面内容
res = requests.get('https://wpblog.x0y1.com/?cat=2', cookies=shared_cookies, headers=headers)

# 解析页面
soup = BeautifulSoup(res.text, 'html.parser')

# 选择所有的代表标题的 a 标签
titles = soup.select('h2.entry-title a')

# 获取四篇文章的链接
links = [i.attrs['href'] for i in titles]

for link in links:
  # 获取文章页面内容
  res_psg = requests.get(link, cookies=shared_cookies, headers=headers)
  # 解析文章页面
  soup_psg = BeautifulSoup(res_psg.text, 'html.parser')
  # 获取文章内容的标签
  content = soup_psg.select('div.entry-content')[0]
  # 打印文章内容
  print(content.text)