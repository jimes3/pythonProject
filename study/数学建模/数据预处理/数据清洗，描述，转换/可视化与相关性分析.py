import ydata_profiling as pp
import pandas as pd
import webbrowser
import seaborn as sns
import matplotlib.pyplot as plt


df =  pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                  usecols=['lstat','rm', 'rad','chas'])
#数据分析
report = df.profile_report(title='数据分析')
report.to_file(output_file='analyse.html')
webbrowser.open_new_tab('analyse.html')
sns.pairplot(df,kind='reg',diag_kind='hist',hue='chas') #太慢了，不弄
plt.savefig('关联图.png',  dpi=600)