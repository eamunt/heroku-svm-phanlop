from flask import Flask,render_template,url_for,request
from flask_material import Material

import pandas as pd
import numpy as np

import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/dauvao', methods=['POST'])
def dauvao():
    if request.method == 'POST':
        gioitinh = request.form['gioitinh']
        diemToan = request.form['diemToan']
        diemVan = request.form['diemVan']
        model = request.form['model']
        sample_data = [gioitinh, diemToan, diemVan]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)
        if model == "svc-polynomial":
            ic_model = joblib.load("data/svc_poly_model_df.pkl")
        elif model == "svc-rbf":
            ic_model = joblib.load("data/svc_rbf_model_df.pkl")
        result_prediction = ic_model.predict(ex1)

        if result_prediction == [1]:
            result_prediction = "Nhóm 1"
        elif result_prediction == [2]:
            result_prediction = "Nhóm 2"
        elif result_prediction == [3]:
            result_prediction = "Nhóm 3"

        if gioitinh == "0":
            gender = "Nam"
        elif gioitinh == "1":
            gender = "Nữ"
    return render_template("index.html", gender=gender,
        diemToan=diemToan,
        diemVan=diemVan,
        clean_data=clean_data,
        result_prediction=result_prediction)

if __name__ == '__main__':
    app.run(debug=True)