import urllib.request
import re
   
# 获取主页源码
url = 'https://pvp.qq.com/web201605/herolist.shtml'
a = urllib.request.urlopen(url)  # 访问网页
a = a.read()  # 读取源码
a = a.decode('gbk')  # 转码为gbk
# 获取英雄名称和ID
a = (re.compile('输入英雄不存在，请重新输入' + "(.*?)" + '英雄介绍', re.S).findall(a))[0]  # 取主要源码
id = re.compile('<li><a href="herodetail/' + '(.*?)' + '.shtml"', re.S).findall(a)  # 取id
name = re.compile('height="91" alt="' + '(.*?)' + '">').findall(a)  # 取名字
print('本次共抓取到' + str(len(id)) + '个英雄数据')
# 询问保存路径
r = True
while r:
    f = input('请输入想要保存的路径：')
    if f == '':
        print('不能为空')
    if f != '':
        r = False
   
# i
i = 0
for i in range(0, len(id) + 1):  # 对列表位置进行循环
    print('正在下载' + name[i])
    url1 = 'https://pvp.qq.com/web201605/herodetail/' + id[i] + '.shtml'  # 英雄主页网址
    url1 = ((urllib.request.urlopen(url1)).read()).decode('gbk')  # 获取源码，并且转码为gbk
    pf = re.compile('data-imgname="' + '(.*?)' + '">', re.S).findall(url1)  # 取出皮肤名字
    pf = pf[0]
    """删除沉余字符"""
    pf = pf.replace('&', '', )
    pf = pf.replace('0', '', )
    pf = pf.replace('1', '', )
    pf = pf.replace('2', '', )
    pf = pf.replace('3', '', )
    pf = pf.replace('4', '', )
    pf = pf.replace('5', '', )
    pf = pf.replace('6', '', )
    pf = pf.replace('7', '', )
    pf = pf.replace('8', '', )
    pf = pf.replace('9', '', )
   
    pf = pf.split('|')#文本分割
    print(pf)
   
    for i1 in range(0, len(pf)):  # 对皮肤列表进行循环
        print(name[i] + '-' + pf[i1])
        url2 = 'https://game.gtimg.cn/images/yxzj/img201606/skin/hero-info/' + id[i] + '/' + id[i] + '-bigskin-' + str(
            i1 + 1) + '.jpg'  # 皮肤下载地址
        urllib.request.urlretrieve(url2, f + '/' + name[i] + '-' + pf[i1] + '.jpg')  # 下载
