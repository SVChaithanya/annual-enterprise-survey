from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
engine = create_engine("mysql+pymysql://root:Surya%40851973@localhost/ml_projects")
df = pd.read_sql("SELECT * FROM ml_projects.annual",engine)
print(df)
print(df.shape)
print(df.isna().sum())
print(df.describe())
print(df.info())

import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x='Variable_code',y='Value',hue='Units',data=df)
plt.show()
sns.barplot(y='Variable_name',x='Value',hue='Variable_category',data=df)
plt.show()
sns.boxplot(x='Variable_code',hue='Variable_category',data=df)
plt.show()

from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error

y=df['Value']
X=df.drop(columns=['Value'])
#Seperating the X into num and cat
num = ['Year','Industry_code_NZSIOC']
cat = ['Industry_code_ANZSIC06','Variable_category','Variable_name','Variable_code','Units','Industry_name_NZSIOC','Industry_aggregation_NZSIOC']
#doing pipeline for the (num and cat ) of X
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num),
    ('cat', cat_pipeline, cat)
])

# Base models without redundant pipelines
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=42)
gb_model = GradientBoostingRegressor(learning_rate=0.1, random_state=42)

stack_model = StackingRegressor(
    estimators=[('lr', lr_model), ('rf', rf_model), ('gb', gb_model)],
    final_estimator=RandomForestRegressor(n_estimators=100, max_depth=50, random_state=42)
)

# Full pipeline
module = Pipeline([
    ('preprocess', preprocessor),
    ('stack', stack_model)
])


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
module.fit(X_train,y_train)
y_pred = module.predict(X_test)
print('r2_score:',r2_score(y_test,y_pred))
print('RMS:',root_mean_squared_error(y_test,y_pred))
print('MSE:',mean_squared_error(y_test,y_pred))
#CROSS_VAL_SCORE
kf=KFold(n_splits=5,shuffle=True,random_state=42)
score = cross_val_score(module,X,y,cv=kf)
print('score mean:',score.mean())
print('score:',score)
