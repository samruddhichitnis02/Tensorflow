import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from flask import Flask, request, render_template


app =  Flask(__name__)  #created the instance of the Flask()
model = pickle.load(open('model.pkl', 'rb')) #Load the trained model in to model
lb_geography = model['LBGeography']
lb_gender = model['LBGender']
ohe = model['OHE']
sc = model['SC']


@app.route('/')
def home():
    return render_template('index.html')


#bounded /api with the method predict()
@app.route('/predict',methods=['POST'])
def predict():

    CreditScore = int(request.form['CreditScore'])
    Geography = request.form['Country']
    Gender = request.form['Gender']
    Age = int(request.form['Age'])
    Tenure = int(request.form['Tenure'])
    Balance = float(request.form['Balance'])
    NumOfProducts = int(request.form['Products'])
    HasCrCard = int(request.form['CreditCard'])
    IsActiveMember = int(request.form['Member'])
    EstimatedSalary = float(request.form['Salary'])

    features = [[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]
    data = pd.DataFrame(features)
    data.iloc[:, [0, 3, 5, 9]] = sc.transform(data.iloc[:, [0, 3, 5, 9]])
    X = data.iloc[:, :].values

    X[:, 1] = lb_geography.transform(X[:, 1])
    X[:, 2] = lb_gender.transform(X[:, 2])

    X = ohe.transform(X).toarray()
    X = X[:, 1:]

    X = np.reshape(X, (1, 11))

    with tf.compat.v1.Session() as ses:
        saver = tf.compat.v1.train.import_meta_graph('/home/admin1/Tensorflow/classification_models/Bank.ckpt.meta')
        model = saver.restore(ses, tf.train.latest_checkpoint('/home/admin1/Tensorflow/classification_models/'))

        graph = tf.compat.v1.get_default_graph()
        input_x = graph.get_tensor_by_name("x:0")
        input_y = graph.get_tensor_by_name("y:0")

        output = graph.get_tensor_by_name("output:0")

        feed_dict = {input_x: X}
        prediction = output.eval(feed_dict=feed_dict)
        result = tf.nn.sigmoid(prediction)
        pred = ses.run(result, feed_dict={input_x: X})

    if pred > 0.5:
        result = 'Exited the Bank'
    else:
        result = 'Did not Exit the Bank'

    return render_template('result.html', prediction_text = result)


if __name__ == '__main__':
    app.run(debug = True)