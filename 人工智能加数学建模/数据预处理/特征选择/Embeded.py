from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

##########################     导入数据      #################################
df = pd.read_csv("ID,crim,zn,indus,chas,nox,rm,age,di.csv",
                 usecols=['crim','age', 'medv','rad'])
X = df[['crim', 'age','rad']].values
y = df['medv'].values

###################      L1正则化       ############################
la = Lasso(alpha=3)  #  alpha值越大，筛选越严格. 值越大，越相关
la.fit(X,y)
for i in range( X.shape[1] ):
    print ('L1正则化:\n',df.columns.values[i+1], format( la.coef_[i], '.9f'))

###################    Tree−based  FeatureSelection    #####################
rf = RandomForestRegressor(n_estimators=15, max_depth=6)  # 值越大，越相关
rf.fit(X,y)
for i in range( X.shape[1] ):
    print ('随机森林:\n',df.columns.values[i+1], format( rf.feature_importances_[i], '.9f'))
