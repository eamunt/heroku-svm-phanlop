from flask import Flask,render_template,url_for,request
from flask_material import Material

import pandas as pd
import numpy as np

import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index1.html")

@app.route('/dauvao', methods=['POST'])
def dauvao():
    try:
        if request.method == 'POST':
            gioitinh = request.form['gioitinh']
            diemToan = request.form['diemToan']
            diemVan = request.form['diemVan']


            model = request.form['model']
            normal_data = [gioitinh, diemToan, diemVan]
            #convert
            float_data = [float(i) for i in normal_data]
            print(float_data)
            t = np.array(float_data).reshape(1,-1)
            
            if model == "svc-polynomial":
                s_model = joblib.load("data/svc_poly_model_df.pkl")
            elif model == "svc-rbf":
                s_model = joblib.load("data/svc_rbf_model_df.pkl")
            result_prediction = s_model.predict(t)

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


        return render_template("index1.html", gender=gender,
            diemToan=diemToan,
            diemVan=diemVan,
            result_prediction=result_prediction)
    
    except:
        return render_template("index1.html",
            result_prediction="Chưa nhập đủ dữ liệu")


if __name__ == '__main__':
    app.run(debug=True)