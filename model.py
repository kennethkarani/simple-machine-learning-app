# Importing the libraries
import numpy as np
import pandas as pd
import pickle

# pandas method to read csv files. basically excel
dataset = pd.read_csv('price.csv')

# in the column bedroom. any rows that are empty replace with 0
dataset['bed_room'].fillna(0, inplace=True)

# in the column area. any rows that are empty replace with the mean of the other area values
dataset['area'].fillna(dataset['area'].mean(), inplace=True)

# X is a subset of the original dataset. with all the rows and only the firs 3 columns
X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

# for each value for bedroom. perform the function to turn it into a integer
X['bed_room'] = X['bed_room'].apply(lambda x : convert_to_int(x))

# select all rows and only the last column - the price
y = dataset.iloc[:, -1]

#needed for the linear model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with training data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 2200, 5]]))
