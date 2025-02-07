import requests
import time

headers = {
  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36',
  'referer': 'http://movie.mtime.com/'
}

for num in range(1, 6):
  params = {
    "tt": "{}".format(int(time.time() * 1000)),
    "movieId": "209164",
    "pageIndex": "{}".format(num),
    "pageSize": "20",
    "orderType": "1"
  }

  res = requests.get(
    'http://front-gateway.mtime.com/library/movie/comment.api',
    params=params,
    headers=headers)

  comment_list = res.json()['data']['list']
  for i in comment_list:
    print("用户：", i['nickname'])
    print("打分：", i['rating'])
    print("评论：", i['content'])
  # 暂停一下，防止爬取太快被封
  time.sleep(1)