import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from prepare_data import  prepare_data
import pickle

df=pd.read_csv('Z:/Documents/proj data/train.csv')



df=prepare_data(df)

X = df.drop(columns=['trip_duration'])
y = df['trip_duration']

ridge_model = Ridge(alpha=1)
ridge_model.fit(X, y)

pickle.dump(ridge_model,open('Z:/pic,vid/ml project/ridge_model_saved','wb'))




