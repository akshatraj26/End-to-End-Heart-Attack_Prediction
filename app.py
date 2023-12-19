from flask import Flask, render_template, request, url_for, flash, redirect
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import pandas as pd
import pytz
from datetime import datetime


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'this is mproject'
db = SQLAlchemy(app)
app.app_context().push()

df = pd.read_csv('descriptive.csv', index_col =False)


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    age = db.Column(db.Integer, nullable = False)
    sex = db.Column(db.Integer, nullable = False)
    chest_pain = db.Column(db.Integer, nullable =False)
    normal_bp = db.Column(db.Integer, nullable =False)
    cholestrol = db.Column(db.Integer, nullable = False)
    fastingbloodsugar = db.Column(db.Integer, nullable = False)
    rest_esg = db.Column(db.Integer, nullable = False)
    max_heart_rate = db.Column(db.Integer, nullable = False)
    exng = db.Column(db.Integer, nullable = False)
    oldpeak = db.Column(db.Integer, nullable = False)
    slp = db.Column(db.Integer, nullable = False)
    ca = db.Column(db.Integer, nullable = False)
    thalassemia = db.Column(db.Integer, nullable = False)
    target = db.Column(db.String(64), nullable = False)
    indian_time = pytz.timezone('Asia/Kolkata')
    created_on = db.Column(db.DateTime, default = datetime.now(indian_time))
    
    
    
    


@app.route('/')
def index():
    table_html = df.to_html(classes= 'table table-stripped')
    return render_template('index.html', table = table_html)




target = {0 : 'Less Chance of Heart Attack', 1 : 'More chance of Heart Attack'}

model = joblib.load('models/XGBoostscale.joblib')


@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        chest_pain = int(request.form['cp'])
        normal_bp = int(request.form['trbps'])
        cholestrol = int(request.form['chol'])
        fastingbloodsugar = int(request.form['fbs'])
        rest_esg = int(request.form['rest_ecg'])
        max_heart_rate = int(request.form['thalach'])
        exng = int(request.form['exng'])
        oldpeak = float(request.form['oldpeak'])
        slp = int(request.form['slp'])
        ca = int(request.form['ca'])
        thal = request.form['thall']



        feature_variables = np.array([age, gender, chest_pain, normal_bp, cholestrol, fastingbloodsugar,rest_esg, max_heart_rate, exng, oldpeak, slp, ca, thal])
        predict = model.predict([feature_variables])[0]
        print(predict)
        class_label = target[predict]
        
        print(feature_variables)
        
        patient = Patient(age = age, sex =gender, chest_pain=chest_pain, normal_bp =normal_bp, cholestrol=cholestrol, 
                        fastingbloodsugar=fastingbloodsugar, rest_esg =rest_esg, max_heart_rate= max_heart_rate, exng = exng,
                        oldpeak=oldpeak, slp = slp, ca =ca, thalassemia =thal, target = class_label
                        )
        
        label = f"According to the data it is likely to be {class_label}"
        print(label)
        db.session.add(patient)
        try:
            db.session.commit()
            flash('Your data has been saved', 'success')
            return render_template('index.html', predict=label)
        except Exception as e:
            flash('An Error occured while saving your data', 'danger')
            print(str(e))
            return redirect(url_for('index'))
            
        
    return render_template('index.html')
        
        

if __name__ == '__main__':
    app.run(debug=True)