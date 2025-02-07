import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
np.set_printoptions(suppress=True) #不以科学计数法输出
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
plt.rcParams['axes.unicode_minus'] = False #显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 散点图标签可以显示中文


#为后续程序运行时间考虑，先得到城市的数据
country = ['Finland','Sweden','Norway','Denmark']
data_0 = pd.read_csv('city_temperature.csv',sep = ',',header = 0,  encoding = 'utf-8')
data = data_0[data_0['Country'].isin(country)]
data.to_csv('country.csv', index=False)

df = pd.read_csv('country.csv',sep = ',',header = 0,encoding = 'utf-8')

# 问题一
# 筛选出城市在2003年的数据
algiers_2003_data = df[(df['City'] == 'Copenhagen') & (df['Year'] == 2003)]
# 按月分组，计算每月的平均气温
monthly_avg_temperatures = algiers_2003_data.groupby('Month')['AvgTemperature'].mean().reset_index()
# 绘制2003年内每个月的平均气温变化曲线
plt.plot(monthly_avg_temperatures['Month'], monthly_avg_temperatures['AvgTemperature'], marker='o')
plt.title("Copenhagen2003年每月平均气温变化")
plt.xlabel("月份")
plt.ylabel("平均气温 (摄氏度)")
plt.grid(True)
plt.show()

# 问题二
# 筛选出城市在2000年到2019年3月的数据
algiers_data_2000_to_2019 = df[(df['City'] == 'Copenhagen') & (df['Year'].between(2000, 2019)) & (df['Month'] == 3)]
# 按年和月分组，计算每月的平均气温
monthly_avg_temperatures = algiers_data_2000_to_2019.groupby(['Year', 'Month'])['AvgTemperature'].mean().reset_index()
# 将Year和Month合并为日期列
monthly_avg_temperatures['Date'] = pd.to_datetime(monthly_avg_temperatures[['Year', 'Month']].assign(DAY=1))
# 绘制2000年到2019年3月每个月的平均气温变化曲线
plt.plot(monthly_avg_temperatures['Date'], monthly_avg_temperatures['AvgTemperature'], marker='o')
plt.title("Copenhagen2000年到2019年3月每月平均气温变化")
plt.xlabel("日期")
plt.ylabel("平均气温 (摄氏度)")
plt.grid(True)
plt.show()

# 问题三
# 筛选出城市在2003年的数据
algiers_data_2003 = df[(df['City'] == 'Copenhagen') & (df['Year'] == 2003)]
# 计算每月的温差（每月最高温和最低温的差值）
monthly_temperature_difference = (algiers_data_2003.groupby('Month')['AvgTemperature'].max() -
                                  algiers_data_2003.groupby('Month')['AvgTemperature'].min()).reset_index()
# 绘制2003年每月的温差变化曲线
plt.plot(monthly_temperature_difference['Month'], monthly_temperature_difference['AvgTemperature'], marker='o')
plt.title("Copenhagen2003年每月温差变化")
plt.xlabel("月份")
plt.ylabel("温差 (摄氏度)")
plt.grid(True)
plt.show()

# 问题四
# 筛选第一个城市的数据
city1_data = df[(df['City'] == 'Copenhagen') & (df['Year'] == 2003) & (df['Month'] == 3)]
# 筛选第二个城市的数据
city2_data = df[(df['City'] == 'Helsinki') & (df['Year'] == 2003) & (df['Month'] == 3)]
# 绘制两个城市的平均气温对比曲线
plt.plot(city1_data['Day'], city1_data['AvgTemperature'], label='Algiers', marker='o')
plt.plot(city2_data['Day'], city2_data['AvgTemperature'], label='AnotherCity', marker='o')
plt.title("2003年3月两个城市平均气温对比")
plt.xlabel("日期")
plt.ylabel("平均气温 (摄氏度)")
plt.legend()
plt.grid(True)
plt.show()

# 问题五
# 选择某城市的数据
city_data = df[df['City'] == 'Copenhagen']
# 选择2000年到2019年的数据
year_range = range(2000, 2020)
city_data_2000_to_2019 = city_data[city_data['Year'].isin(year_range)]
# 筛选超过30度的天数
hot_days_data = city_data_2000_to_2019[city_data_2000_to_2019['AvgTemperature'] > 30]
# 统计每年超过30度的天数
hot_days_by_year = hot_days_data.groupby('Year').size().reset_index(name='Count')
print(hot_days_by_year['Count'])
# 绘制柱状图
plt.bar(hot_days_by_year['Year'], hot_days_by_year['Count'])
plt.title("Copenhagen 2000-2019年超过30度的天数")
plt.xlabel("年份")
plt.ylabel("天数")
plt.grid(axis='y')
plt.show()

# 问题六
# 选择城市的数据
city_data = df[df['City'] == 'Copenhagen']
# 选择2000年到2019年的数据
year_range = range(2000, 2020)
city_data_2000_to_2019 = city_data[city_data['Year'].isin(year_range)]
# 筛选高温天数（>30度）
hot_days_data = city_data_2000_to_2019[city_data_2000_to_2019['AvgTemperature'] > 30]
# 筛选低温天数（<5度）
cold_days_data = city_data_2000_to_2019[city_data_2000_to_2019['AvgTemperature'] < 5]
# 统计每年高温和低温天数
hot_days_by_year = hot_days_data.groupby('Year').size().reset_index(name='HotDays')
cold_days_by_year = cold_days_data.groupby('Year').size().reset_index(name='ColdDays')
# 合并数据
merged_data = pd.merge(hot_days_by_year, cold_days_by_year, on='Year', how='outer').fillna(0)
# 绘制柱状图
width = 0.35
plt.bar(merged_data['Year'] - width/2, merged_data['HotDays'], width, label='高温天数 (>30度)')
plt.bar(merged_data['Year'] + width/2, merged_data['ColdDays'], width, label='低温天数 (<5度)')
plt.title("Copenhagen 2000-2019年高温和低温天数统计")
plt.xlabel("年份")
plt.ylabel("天数")
plt.legend()
plt.grid(axis='y')
plt.show()

# 问题七
# 读取全球平均气温数据集
df_global = data_0
# 选择1995年到2019年的数据
years_range = range(1995, 2020)
global_data_1995_to_2019 = df_global[df_global['Year'].isin(years_range)]
# 按年分组，计算每年的平均气温
avg_temperature_by_year = global_data_1995_to_2019.groupby('Year')['AvgTemperature'].mean().reset_index()
# 绘制全球平均气温变化趋势
plt.plot(avg_temperature_by_year['Year'], avg_temperature_by_year['AvgTemperature'], marker='o')
plt.title("1995-2019年全球平均气温变化趋势")
plt.xlabel("年份")
plt.ylabel("全球平均气温 (摄氏度)")
plt.grid(True)
plt.show()

# 问题八
df_finland = df[df['Country'] == 'Finland']
# 选择芬兰1995年到2019年的数据
years_range = range(1995, 2020)
finland_data_1995_to_2019 = df_finland[df_finland['Year'].isin(years_range)]
# 筛选低于0摄氏度的天数
cold_days_data = finland_data_1995_to_2019[finland_data_1995_to_2019['AvgTemperature'] < 0]
# 统计每年低于0摄氏度的天数
cold_days_by_year = cold_days_data.groupby('Year').size().reset_index(name='ColdDays')
# 绘制柱形图
plt.bar(cold_days_by_year['Year'], cold_days_by_year['ColdDays'])
plt.title("1995-2019年芬兰每年低于0摄氏度的天数")
plt.xlabel("年份")
plt.ylabel("天数")
plt.grid(axis='y')
plt.show()
