# This is basically the heart of my flask 

from flask import Flask, render_template, request, redirect, url_for
from scipy import sparse
import pandas as pd
import numpy as np
import pickle
import warnings


warnings.filterwarnings("ignore")
app = Flask(__name__)


with open('Model/lr_model.pkl','rb') as fp:
	lr_model = pickle.load(fp)
with open('Model/tf_idf_vectorizermodel.pkl','rb') as fp:
	tf_idf_model = pickle.load(fp)
with open('Model/recommendation_model.pkl','rb') as fp:
	recommend_model = pickle.load(fp)
with open('Model/sample_csv.pkl','rb') as fp:
	sample_df = pickle.load(fp)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend():
	user_name = request.form.get('user_name')
	products_list = recommend_model.loc[user_name].sort_values(ascending=False)[0:20]
	products = []
	for product in products_list.keys():
		products.append(product)

	filter_df = pd.DataFrame(columns=['product','feedback'])
	final_recommend = pd.DataFrame(columns=['product','positive_rate'])

	for product in products:
		feedbacks = sample_df[sample_df['name']==product]['reviews_text'].tolist()
		X_test_transformed = tf_idf_model.transform(feedbacks)
		y_pred =  lr_model.predict(X_test_transformed)
		positive_percent = (pd.Series(y_pred).value_counts()[0] / len(y_pred))*100
		temp_dict = {'product': product ,'positive_rate':positive_percent} 
		final_recommend = final_recommend.append(temp_dict,ignore_index=True)
		final_recommend = final_recommend.sort_values(by='positive_rate',ascending=False)[0:5]
	

	final_recommend = final_recommend.to_dict('records')

	return render_template('index.html',products = final_recommend,title1="Product Name",title2 = "Positive Sentiment Rate")

if __name__ == "__main__":
    app.run()