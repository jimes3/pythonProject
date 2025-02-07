import base64

def to_py(other_names, py_name):
    # 用于存放所有图片和音频的base64
    write_data = []
    # 循环处理每个文件(图片or音频)
    for other_name in other_names:
        # 切割文件名，把文件名转为变量名
        filename = other_name.replace('.', '_')
        # 以二进制读取文件
        with open(other_name, 'rb') as r:
            # 将文件转为base64
            b64str = base64.b64encode(r.read())
        # 拼接变量，格式：变量名 = "base64编码"
        write_data.append(f'{filename} = "{b64str.decode()}"\n')

    # 循环把所有base64变量写入py文件
    with open(f'{py_name}.py', 'w+') as w:
        for data in write_data:
            w.write(data)
# 需要转码所有图片和音频：
# 注：图片或音频名称不要用汉字和数字，因为文件的名字要充当变量名,名字也是地址
names = ["sta1.png", "bac.png","img.png","img_1.png","img_2.png","img_3.png"]
# 将names列表里面的文件以base64写到 base64_data.py 中
to_py(names, 'base64_data')
print("转码完成...")

'''
# 以下代码放在打包代码中，因为导入了base64_data，所以会一起打包
# 导入图片转码后所在的py文件
import base64
from base64_data import *
# 把py文件里的变量解码出来，以二进制写入文件中去
with open(r'sta1.png', 'wb') as w:
    # test_mp3变量是把 .改为_ 的文件名
    w.write(base64.b64decode(sta1_png))
# 运行上面程序后，会在open的路径处生成这个图片或音频，所以需要用到这个图片或音频时直接用相对路径即可
# 为了不留痕迹，文件用后即删，不想删的就不执行
# os.remove('D:/test.mp3')
'''
