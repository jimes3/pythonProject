import time
import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook

class JobSpider:
    # 初始化方法
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
            'cookie': '__gc_id=8ed110d37dbe4c49a622769fada6cf87; __uuid=1659492492170.97; __s_bid=9723a41b222435912a20db82f0d7d05d8ebf; __tlog=1659581622908.93%7C00000000%7C00000000%7C00000000%7C00000000; Hm_lvt_a2647413544f5a04f00da7eee0d5e200=1659492492,1659581623; __session_seq=2; __uv_seq=2; Hm_lpvt_a2647413544f5a04f00da7eee0d5e200=1659581639'
        }
        self.session.headers.update(self.headers)
    # 获取单页数据
    def get_one_page_data(self,url):
        one_page_data = []
        req = self.session.get(url)
              # 获取成功时，解析网页
        if req.status_code == 200:
            print('解析成功')
            soup = BeautifulSoup(req.text, 'html.parser')
               # 获取所有 40 条招聘信息
            job_items = soup.select('div.job-card-pc-container')
               # 遍历每条招聘信息，提取出所需数据
            for item in job_items:
                result = {}
                 # 提取职位信息
                result['job_name'] = item.select('div.ellipsis-1')[0].text # 职位名
                result['area'] = item.select('span.ellipsis-1')[0].text    # 地区
                result['salary'] = item.select('span.job-salary')[0].text  # 薪资范围
                 # 提取职位要求
                job_labels = item.select('span.labels-tag')
                result['working_exp'] = job_labels[0].text  # 经验要求
                result['edu_level'] = job_labels[1].text    # 学历要求
                 # 提取公司信息
                result['company_name'] = item.select('span.company-name')[0].text #公司名
                company_labels = item.select('.company-tags-box > span')
                 #
                if company_labels:
                    result['company_type'] = company_labels[0].text
                else:
                    result['company_type'] = '无'
                one_page_data.append(result)
            return one_page_data
             # 获取失败时，打印失败信息
        else:
            print('解析失败')

    # 获取多页数据
    def get_data(self):
        data = []
        for i in range(10):
            url = 'https://www.liepin.com/zhaopin/?headId=50338e415cb4837971583f47ec6ca922&ckId=8etpnmsoj43n21931zg4hvznlv6vcfta&oldCkId=50338e415cb4837971583f47ec6ca922&fkId=32vtkhyyf8yw5crx6huiu58hrspb67jb&skId=32vtkhyyf8yw5crx6huiu58hrspb67jb&sfrom=search_job_pc&key=python&currentPage={}&scene=page'.format(i)
             # 调用 get_one_page_data() 方法，获取单页数据
            data = data + self.get_one_page_data(url)
            time.sleep(1) # 防止爬取速度过快被封 ip
        return data
    # 处理数据
    def process_data(self):
        data = self.get_data()
        total = 0  # 累计平均年薪
        count = 0  # 统计非面议职位个数
        rows = [['职位名', '地区', '薪资范围', '平均薪资', '学历要求', '经验要求', '公司名', '公司类型']]
        # 遍历每条数据
        for item in data:
            # 计算该职位平均年薪
            salary = self.get_salary_value(item['salary'])
            if salary != '面议':
                total += salary
                count += 1
                # 将数据添加到 rows 中
                row = list(item.values())
                row.insert(3, salary)
                rows.append(row)

        salary_avg = round(total / count, 2)  # 计算平均年薪并保留两位小数
        print('Python 相关职位平均年薪是{}元'.format(salary_avg))
        return rows
        # 计算平均年薪

    def get_salary_value(self, salary):
        if salary == '面议':
            return '面议'
        else:
            if '薪' in salary:
                salary_range, salary_times_str = salary.split('k·')  # 分割成两部分
                salary_times = int(salary_times_str.strip('薪'))  # 一年发多少薪
            else:
                salary_range = salary.strip('k')
                salary_times = 12

            if '-' in salary_range:
                salary_min_str, salary_max_str = salary_range.split('-')  # 分割薪资范围
                salary_min = int(salary_min_str)  # 最低月薪
                salary_max = int(salary_max_str)  # 最高月薪
                salary_avg = (salary_min + salary_max) / 2  # 平均月薪
            else:
                salary_avg = int(salary_range)  # 给定月薪

            return salary_avg * salary_times * 1000  # 计算平均年薪

    # 保存数据
    def save_data(self):
      # 处理后的数据
        rows = self.process_data()
     # 新建工作簿
        wb = Workbook()
     # 选择默认的工作表
        sheet = wb.active
     # 给工作表重命名
        sheet.title = 'python职位信息'
     # 将数据一行一行写入
        for row in rows:
            sheet.append(row)

      # 保存文件
        wb.save('猎聘职位信息表.xlsx')

spider = JobSpider()
spider.save_data()