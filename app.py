# importing necessary libraries and functions
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

dt=pd.read_csv("admins.csv")
college=np.unique(dt['College'])
clg_id=[]
for i in range(len(college)):
    clg_id.append(i+1)
dt['College_id']=dt['College'].replace(college,clg_id)

app = Flask(__name__, template_folder = 'template') #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    #prediction = model.predict(final_features) # making prediction

    pred = model.predict(final_features)
    pr=pd.DataFrame()
    pred=int(pred)
    pr["id"]=[pred]
    clg_name=pr["id"].replace(clg_id,college)
    
    return render_template('index.html', prediction_text='Your Predicted College is: {}'.format(clg_name.to_string(index=False))) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)