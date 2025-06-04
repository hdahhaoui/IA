import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

file='FI-CLAY-oedo14-282.xlsx'
df=pd.read_excel(file, sheet_name='data')
features=['e0','wn','Cc','Preconsolidation_stress','OCR']
target='Cs'
df=df[features+[target]].dropna()
X=df[features]
y=df[target]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
pipe=Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=300, random_state=42))
])
pipe.fit(X_train,y_train)

pred=pipe.predict(X_test)
print('Hold-out R2:', r2_score(y_test,pred))
print('Hold-out MAE:', mean_absolute_error(y_test,pred))

cv_scores=cross_val_score(pipe,X,y,cv=5,scoring='r2')
print('CV R2 mean:', cv_scores.mean())
print('CV R2 std:', cv_scores.std())
