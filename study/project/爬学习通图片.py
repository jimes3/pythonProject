from urllib.request import urlretrieve
import time
# img_url为图片链接,
# file_name为文件储存路径及文件名
for i in range(1,51):
    img_url= f"https://s3.ananas.chaoxing.com/sv-w7/doc/aa/ec/30/937438b56cdaac1fb487080421263bbd/thumb/{i}.png"
    file_name = f'D:\.b学习资料\图片\微分方程{i}.png'
    urlretrieve(img_url, file_name)
    time.sleep(0.1)
