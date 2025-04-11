import numpy as np
from flask import Flask, request,render_template
import pickle

#creates Flask app
app = Flask(__name__)

#loads trained model(one saved in model.py)
model = pickle.load(open('model.pkl', 'rb'))

#defines homepage route(/); when someone opens site, flask will show index.html
@app.route('/')
def home():
    return render_template('index.html')

#defines the /predict route; only response to POST requests
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #gets all input values from form; converts them to integers (the webpage takes values but in string, so converts them to integers) [2,2200,5]
    int_features = [int(x) for x in request.form.values()]
    
    #wraps the array inside another list to make shape [[2,2200,5]]; because model expects this 2D array for one row of inputs. model.predict([[2,2200,5]])
    final_features = [np.array(int_features)]
    
    #calls model to predict house price
    prediction = model.predict(final_features)

    #rounds the first (and only) prediction to 2 decimal places; 500000.23
    output = round(prediction[0], 2)

    #renders the html again, but now adds the predicted value so it shows on the page
    return render_template('index.html', prediction_text='House price should be $ {}'.format(output))

#runs the flask app if this file is exectued directly; it will reload on code chnges
if __name__ == "__main__":
    app.run(debug=True)
