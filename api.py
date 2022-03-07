# This is basically the heart of my flask 
from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings


warnings.filterwarnings("ignore")
# import xgboost


app = Flask(__name__)

# with open('Model/model.pkl','rb') as fp:
# 	model = pickle.load(fp)
with open('Model/lr_model.pkl','rb') as fp:
	lr_model = pickle.load(fp)
with open('Model/tf_idf_vectorizermodel.pkl','rb') as fp:
	tf_idf_model = pickle.load(fp)
with open('Model/recommendation_model.pkl','rb') as fp:
	recommend_model = pickle.load(fp)

PORT = process.env.PORT | '8000'
app = express()
app.set("port",PORT)
@app.route('/')
def home():
	return render_template('index_recommend.html')

@app.route('/predict',methods=['POST'])
def predict():
	SL = request.form.get('sepal_length')
	SW = request.form.get('sepal_width')
	PL = request.form.get('petal_length')
	PW = request.form.get('petal_width')
	Input = [[SL,SW,PL,PW]]
	prediction = model.predict(Input)[0]
	return render_template('index.html', OUTPUT=str(prediction))

@app.route('/recommend',methods=['POST'])
def recommend():
	user_name = request.form.get('user_name')
	products_list = recommend_model.loc[user_name].sort_values(ascending=False)[0:20]
	products = []
	for product in products_list.keys():
		products.append(product)


	return render_template('index_recommend.html',OUTPUT = products)

if __name__ == "__main__":
    app.run()