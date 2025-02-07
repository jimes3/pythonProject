import pandas as pd

rnames = ['uid','mid','rating','timetamp']
ratings = pd.read_table('u.data',sep='\t', header = None, names = rnames)
ratings["Datetime"] = pd.to_datetime(ratings['timetamp'], unit = 's')
print(ratings[:5])

unames = ['uid','age','gender','occupation','zip']
users = pd.read_table('u.user',sep='|',header=None,names=unames)
print(users.head(5))

mnames = ['mid','title','date1','date2','url','unknown','Action','Adventure','Animation',
          'Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir',
          'Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
movies = pd.read_table('u.item',sep='|',header=None,names=mnames,encoding='ISO-8859-1')
print(movies.head(5))

frame = pd.merge(pd.merge(ratings,users),movies)
print(frame.head(5))

#############################################  ??2??
print(frame['rating'].groupby(frame['gender']).mean())
print(frame['rating'].groupby(frame['age'].apply(round,args=[-1])).mean())
print(frame['rating'].groupby([frame['age'].apply(round,args=[-1]),frame['gender']]).mean())

#############################################  ??3??
print(frame['rating'].groupby([frame['gender'],frame['title']]).agg(['mean','count']).
      sort_values(by=['count','mean'],ascending=[False,False]))

############################################  ??4??
frame1 = frame['rating'].groupby([frame['gender'],frame['title']]).agg(['mean','count'])
print(frame1[frame1['count']>100].sort_values(by='mean',ascending=False))

ratings_by_title = frame.groupby('title').size()
print(frame.pivot_table('rating',index='title',columns='gender',aggfunc='mean')
      .sort_values(by='F',ascending=False).
      loc[ratings_by_title.index[ratings_by_title > 100]])

#############################################  ??5??
frame1 = frame.pivot_table('rating',index='title',columns='gender',aggfunc='mean')
frame1['diff'] = (frame1['M'] - frame1['F']).apply(abs)
print(frame1.sort_values(by='diff',ascending=False))

