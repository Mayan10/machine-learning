import math
import numpy as np #Convert data to numpy arrays to feed scikit-learn
import pandas as pd
from sklearn import preprocessing, svm #preprocessing-clearning/scaling data before machine learning, cross_validation-for testing stage
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf #Because Quandl was depreciated
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

ticker = yf.Ticker("GOOGL")
df = ticker.history(period="max")
df.index = df.index.tz_localize(None)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]
print(df.head())

forecast_col = 'Close'
df.fillna(value=-99999, inplace=True)
forecast_out=int(math.ceil(0.01 * len(df))) #To determine how many days in future we want the forecast to predict,
                                            #for eg. if len(df) = 1000, it means we have 1000 days of data, so 0.01 of that means we want to predict
                                            #the price 10 days later.
                                            #Therefore, forecast_out = number of days in future we want to predict the data
df['label'] = df[forecast_col].shift(-forecast_out) #This line takes the value of 'forecast_col' and moves it to a new column named 'label'
                                                    #And it moves it forecast_out days up because that is the label value for those features.

X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] #Stores the last 'forecast_out' rows of attributes that do not have the label values because we moved it up
X = X[:-forecast_out] #Removes those attribues from X that do not have the label

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#clf = LinearRegression(n_jobs=-1)
#clf.fit(X_train, y_train)
#confidence = clf.score(X_test, y_test)
#print(confidence)
pickle_in = open('linearregression.pickle', 'rb') #Opens the file 'linearregression.pickle' in 'read-binary' mode to read its contents
clf = pickle.load(pickle_in) #Reads binary stream and deserilalizes it back to a python object. Now clf = trained Linear Regression model
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, confidence, forecast_out)

style.use('ggplot')
df['Forecast'] = np.nan

last_date = df.iloc[-1].name #Gives index (date-time) object
last_unix = last_date.timestamp() #convertes in seconds
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix) #Converts the next day from unix(seconds) to date-time object
    next_unix += 86400 #increases the unix value to return the next day on next iteration of loop
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i] #Add a new row with date 'next_date' sets all the attributes to NaN and adds the forecast value from 'forecast_set' stored in 'i'

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

with open('linearregression.pickle', 'wb') as f: #Creates and Opens a file named 'linearregression.pickle' in 'write binary' mode
    pickle.dump(clf, f) #Takes your trained model object 'clf', serealizes it into binary and writes it into the file object f which is linearregression.pickle