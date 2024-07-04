from sklearn.metrics import mean_squared_error, r2_score
import pickle
from prepare_data import  prepare_data
import pandas as pd


dft=pd.read_csv('Z:/Documents/proj data/val.csv')

dft=prepare_data(dft)



X_test = dft.drop(columns=['trip_duration'])
y_test= dft['trip_duration']

ridge_model=pickle.load(open('Z:/pic,vid/ml project/ridge_model_saved','rb'))
y_pred = ridge_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)



