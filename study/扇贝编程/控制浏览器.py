from selenium import webdriver
import time

browser = webdriver.Chrome()
# 打开博客
browser.get('https://wpblog.x0y1.com')
# 找到登录按钮
login_btn = browser.find_element('link text','登录')
# 点击登录按钮
login_btn.click()
# 等待 2 秒钟，等页面加载完毕
time.sleep(2)
# 找到用户名输入框
user_login = browser.find_element('id','user_login')
# 输入用户名
user_login.send_keys('codetime')
# 找到密码输入框
user_pass = browser.find_element('id','user_pass')
# 输入密码
user_pass.send_keys('shanbay520')
# 找到登录按钮
wp_submit = browser.find_element('id','wp-submit')
# 点击登录按钮
wp_submit.click()
# 找到 Python 分类文章链接
#python_cat = browser.find_element_by_css_selector('section#categories-2 ul li a')
# 上一句用新版本的写法如下
python_cat = browser.find_element('css selector', 'section#categories-2 ul li a')
# 点击该分类
python_cat.click()
# 找到跳转的页面中的所有文章标题元素
#titles = browser.find_elements_by_css_selector('h2.entry-title a')
# 上一句用新版本的写法如下
titles = browser.find_elements('css selector', 'h2.entry-title a')

# 找到标题元素中内含的链接
links = [i.get_attribute('href') for i in titles]
# 依次打开 links 中的文章链接
for link in links:
    browser.get(link)
    # 获取文章正文内容
    content = browser.find_element('class name','entry-content')
    print(content.text)

browser.quit()