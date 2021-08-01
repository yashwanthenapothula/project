# -*- coding: utf-8 -*-

from flask import Flask, render_template,request
app = Flask(__name__)
import pickle
import numpy  as np
model = pickle.load(open("lr.pkl","rb"))
@app.route('/')
def hello_world():
    return render_template("index.html")
@app.route('/guest' , methods = ["POST"])
def Guest():
    gap = request.form["gap"]
    grp = request.form["grp"]
    v = request.form["v"]
    gi = request.form["gi"]
    s1 = request.form["s1"]
    s2 = request.form["s2"]
    s3 = request.form["s3"]
    
    arr= np.array([gap,grp,v,gi,s1,s2,s3])
    user_input_prediction = arr.astype('float32')
    prediction = model.predict([user_input_prediction])
    
    
    return render_template("output.html", y="Total Power Consumption is  " +str(prediction))
if __name__ == '__main__':
     app.run(debug = True)
     
   