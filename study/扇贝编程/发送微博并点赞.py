import requests

class WeiboSpider:
  # 创建一个session
  def __init__(self):
    self.session = requests.Session()
    self.headers = {
      'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36',
      'mweibo-pwa': '1',
      'x-requested-with': 'XMLHttpRequest',
      'cookie': '_T_WM=29890277193; SUB=_2A25P2tVBDeRhGeFJ6lEU8ivOzDSIHXVtJPsJrDV6PUJbktCOLU6gkW1NfIIPrWMY7uQGVb7rlMrUqWaAEyzxomec; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5pkUrzXokqb7yJMb3AY9LJ5NHD95QNS020SKzfeoMRWs4DqcjnCJL_9PeLxKBLBonL1h.LxKqL1-eL1hnLxK-LB--LBKzt; SSOLoginState=1658758417; XSRF-TOKEN=458a1b; WEIBOCN_FROM=1110006030; mweibo_short_token=4dc29e8386; MLOGIN=1'
    }
    self.session.headers.update(self.headers)

  def get_st(self):
    # 获取新token所需请求头
    config_headers = {
      'origin': 'https://m.weibo.cn/',
      'referer': 'https://m.weibo.cn/'
    }
    # 更新请求头
    self.session.headers.update(config_headers)
    # 发送获取 token 请求
    config_req = self.session.get('https://m.weibo.cn/api/config')
    config = config_req.json()
    st = config['data']['st']
    return st
  # 编写微博
  def compose(self, content, st):
    compose_headers = {
      'origin': 'https://m.weibo.cn/',
      'referer': 'https://m.weibo.cn/compose/',
      'x-xsrf-token': st
    }
    # 更新默认请求头
    self.session.headers.update(compose_headers)
    # 发送微博所需请求头
    compose_data = {
      'content': content,
      'st': st
    }
    compose_req = self.session.post('https://m.weibo.cn/api/statuses/update', data=compose_data)
    print(compose_req.json())
  # 发送微博
  def send(self, content):
    st = self.get_st()
    self.compose(content, st)

  # 获取微博列表
  def get_weibo_list(self):
    params = {
      'sudaref':'security.weibo.com',
      'type':'uid',
      'value':'2139359753',  #扇贝id     ##################
      'containerid':'1076032139359753'
    }
    weibo_list_req = self.session.get('http://m.weibo.cn/api/container/getIndex',params=params)
    weibo_list_data = weibo_list_req.json()
    weibo_list = weibo_list_data['data']['cards']
    return weibo_list
  # 点赞微博
  def vote_up(self,id):
    vote_up_data = {
      'id': 4790716892451442,    #要点赞的微博id   ###################
      'attitude':'heart',
      'st':self.get_st()
    }
    vote_up_req = self.session.post('https://m.weibo.cn/api/attitudes/create',data = vote_up_data)
    json = vote_up_req.json()
    print(json['msg'])

    # 批量点赞微博
  def vote_up_all(self):
    st = self.get_st()
    vote_headers = {
      'x-xsrf-token':st
    }
    self.session.headers.update(vote_headers)
    weibo_list = self.get_weibo_list()
    for i in weibo_list:
      # card_type 为9是正常微博
      if i['card_type'] == 9:
        self.vote_up(i['mblog']['id'])

weibo = WeiboSpider()
weibo.send('本条微博由 Python 发送')
weibo.vote_up_all()
