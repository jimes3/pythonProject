import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#显示所有列
pd.set_option('display.max_columns', None)

rnames = ['uid','mid','rating','timestamp']
ratings = pd.read_table('u.data',sep='\t', header = None, names = rnames)
ratings["timestamp"] = pd.to_datetime(ratings['timestamp'], unit = 's')

unames = ['uid','age','gender','occupation','zip']
users = pd.read_table('u.user',sep='|',header=None,names=unames)

mnames = ['mid','title','date1','date2','url','unknown','Action','Adventure','Animation',
          'Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir',
          'Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
movies = pd.read_table('u.item',sep='|',header=None,names=mnames,encoding='ISO-8859-1')
frame = pd.merge(pd.merge(ratings,users),movies)
# 对于电影数据，只保留movie_id和电影类别的列
movie_kind = movies.iloc[:, 5:]

############# 分析任务1：那种类型的影片观看的人最多。
genre_counts = movie_kind.sum()
genre_counts.plot(kind='bar', figsize=(12, 6))
plt.title('Most Watched Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

############### 分析任务2：什么电影最受女性的喜欢。那种电影最受男性的喜欢。
# 根据性别分组并计算不同电影类型的平均评分
gender_ratings = frame.groupby(['gender'])[movie_kind.columns].mean()
# 根据性别分组并计算不同电影类型的观看次数
gender_counts = frame.groupby(['gender'])[movie_kind.columns].sum()
# 创建两个图表，每个图表都有双Y轴
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
bar_width = 0.35
index = np.arange(len(movie_kind.columns))
# 绘制男性观众的平均评分
ax1.bar(index, gender_ratings.loc['M'], bar_width, label='Male Ratings', alpha=0.7, color='b')
# 设置第一个Y轴标签
ax1.set_xlabel('Genre')
ax1.set_ylabel('Average Rating (Male)')
ax1.set_title('Ratings by Genre and Gender (Male)')
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(movie_kind.columns, rotation=90)
ax1.legend(loc='upper left')
# 创建第二个Y轴，表示男性观众的观看次数
ax1_twin = ax1.twinx()
ax1_twin.bar(index + bar_width, gender_counts.loc['M'], bar_width, label='Male Viewership', alpha=0.7, color='g')
ax1_twin.set_ylabel('Viewership Count (Male)')
ax1_twin.legend(loc='upper right')
# 绘制女性观众的平均评分
ax2.bar(index, gender_ratings.loc['F'], bar_width, label='Female Ratings', alpha=0.7, color='r')
# 设置第一个Y轴标签
ax2.set_xlabel('Genre')
ax2.set_ylabel('Average Rating (Female)')
ax2.set_title('Ratings by Genre and Gender (Female)')
ax2.set_xticks(index + bar_width / 2)
ax2.set_xticklabels(movie_kind.columns, rotation=90)
ax2.legend(loc='upper left')
# 创建第二个Y轴，表示女性观众的观看次数
ax2_twin = ax2.twinx()
ax2_twin.bar(index + bar_width, gender_counts.loc['F'], bar_width, label='Female Viewership', alpha=0.7, color='y')
ax2_twin.set_ylabel('Viewership Count (Female)')
ax2_twin.legend(loc='upper right')
plt.tight_layout()
plt.show()

############### 分析任务3：根据年龄，分析每个年龄段那种电影最受欢迎。
# 根据年龄分组并计算不同电影类型的平均评分
age_bins = [0, 18, 25, 35, 45, 55, 100]
age_labels = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55+']
frame['age_group'] = pd.cut(frame['age'], bins=age_bins, labels=age_labels)
age_genre_ratings = frame.groupby(['age_group'])[movie_kind.columns].mean()
# 创建图表
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.15
index = range(len(movie_kind.columns))
# 定义不同颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y']
# 遍历年龄段并绘制对应的柱状图
for i, age_group in enumerate(age_labels):
    ax.bar(index, age_genre_ratings.loc[age_group], bar_width, label=age_group, alpha=0.7, color=colors[i])
    index = [x + bar_width for x in index]
ax.set_xlabel('Genre')
ax.set_ylabel('Average Rating')
ax.set_title('Average Ratings by Genre for Different Age Groups')
ax.set_xticks([r + bar_width for r in range(len(movie_kind.columns))])
ax.set_xticklabels(movie_kind.columns, rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

# 分析任务4：分析男女观众兴趣差异。
gender_ratings = frame.groupby(['gender'])[movie_kind.columns].mean()
gender_ratings.T.plot(kind='bar', figsize=(12, 6))
plt.title('Average Genre Ratings by Gender')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.show()

# 分析任务5：根据职业分析出，每种职业最喜欢的电影种类。
# 关联观众的职业和评分数据
user_ratings = pd.merge(users, ratings, on='uid')
user_occupation_ratings = pd.merge(user_ratings, movies, on='mid')
# 计算每种职业对不同电影类型的平均评分
occupation_genre_ratings = user_occupation_ratings.groupby(['occupation'])[movie_kind.columns].mean()
# 找出每种职业对应的平均评分最高的电影类型
favorite_genres_by_occupation = occupation_genre_ratings.idxmax(axis=1)
# 打印每种职业最喜欢的电影种类
print(favorite_genres_by_occupation)


# 分析任务6：战争片各年龄段的人的观看次数，以及评分，并分析结果。
war_movies = frame[frame['War'] == 1]
war_age_counts = war_movies['age_group'].value_counts()
war_age_ratings = war_movies.groupby('age_group')['rating'].mean()

plt.figure(figsize=(12, 6))
plt.subplot(121)
war_age_counts.plot(kind='bar')
plt.title('War Movie Viewership by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Viewership Count')

plt.subplot(122)
war_age_ratings.plot(kind='bar')
plt.title('Average Rating of War Movies by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Rating')

plt.tight_layout()
plt.show()



