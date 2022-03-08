# This is basically the heart of my flask 

from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings

from sklearn.metrics import adjusted_rand_score


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

@app.route('/')
def home():
	return render_template('index_recommend.html')

# @app.route('/predict',methods=['POST'])
# def predict():
# 	SL = request.form.get('sepal_length')
# 	SW = request.form.get('sepal_width')
# 	PL = request.form.get('petal_length')
# 	PW = request.form.get('petal_width')
# 	Input = [[SL,SW,PL,PW]]
# 	prediction = model.predict(Input)[0]
# 	return render_template('index.html', OUTPUT=str(prediction))

@app.route('/recommend',methods=['POST'])
def recommend():
	user_name = request.form.get('user_name')
	products_list = recommend_model.loc[user_name].sort_values(ascending=False)[0:20]
	products = []
	for product in products_list.keys():
		products.append(product)

	sample_df = pd.read_csv('Data/sample30.csv')
	filter_df = pd.DataFrame(columns=['product','feedback'])
	final_recommend = pd.DataFrame(columns=['product','positive_rate'])

	for product in products:
		feedbacks = sample_df[sample_df['name']==product]['reviews_text'].tolist()
		for fb in feedbacks:
			temp_dict = {'product':product,'feedback':fb}
			filter_df = filter_df.append(temp_dict,ignore_index = True)
	
	for product in products:
		X_test = filter_df[filter_df['product']==product]['feedback'].tolist()
		X_test_transformed = tf_idf_model.transform(X_test)
		y_pred =  lr_model.predict(X_test_transformed)
		positive_percent = (pd.Series(y_pred).value_counts()[0] / len(y_pred))*100
		temp_dict = {'product': product ,'positive_rate':positive_percent} 
		final_recommend = final_recommend.append(temp_dict,ignore_index=True)
		final_recommend = final_recommend.sort_values(by='positive_rate',ascending=False)[0:5]
	# prod_list = []
	# psr_list = []
	# for i in final_recommend.index:
	# 	prod_list.append(final_recommend['product'][i])
	# 	psr_list.append(final_recommend['positive_rate'][i])
	final_recommend = final_recommend.to_dict('records')
	header_dummy = ['head_once']

	return render_template('index_recommend.html',products = final_recommend,title1="Product Name",title2 = "Positive Sentiment Rate",len = len(header_dummy))

if __name__ == "__main__":
    app.run()