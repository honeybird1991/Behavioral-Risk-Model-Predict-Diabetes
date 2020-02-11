from flask import render_template,Response,jsonify
from Risk_Diabetes import app
import pandas as pd
from flask import request
from Risk_Diabetes.Model import Model_One, Model_Batch
import numpy as np
import io
import random
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()

demo_features = ['RIDAGEYR','RIAGENDR','DR1TKCAL','BPQ020','ALQ120Q','DBQ197','SLD010H','DPQ040','PAQ665','BMXHT','BMXWT']

@app.route('/')
@app.route('/input')
def input():
    return render_template("input.html",title = 'Home')

def get_data():
    #pull 'data' from input field and store it
    global data
    data = []
    for i in demo_features:
      tmp = request.args.get(i)
      data.append(float(tmp))
    data = np.array(data)
    return data

def predict_batch(data,name):
  j = demo_features.index(name)
  l = len(data)
  if name == 'BMXWT':
    w_max = data[j]*1.30
    w_min = data[j]*0.80
    w = [i for i in np.linspace(w_min,w_max,num = 15)]
  if name == 'DBQ197': 
    w = [0, 1, 2, 3]
  if name == 'BPQ020':
    w = [1,2]
  if name == 'ALQ120Q':
    w = [25,50,100,150,200]
  if name == 'DR1TKCAL':
    w = [250,750,1500,2200]
  if name == 'SLD010H':
    w = [i for i in np.linspace(3,9,num = 6)]
  if name == 'DPQ040':
    w = [0,1,2,3]
  if name == 'PAQ665':
    w = [1,2]
  s = len(w)
  batch_data = np.empty([s,l])
  for i in range(s):
    batch_data[i,:] = data
    batch_data[i,j] = w[i]
  result = Model_Batch(batch_data)
  return w,result

def create_weight():
  
    fig, axs = plt.subplots(1,1,figsize=(8,8))
    # Change the following quote format
    xs, ys = predict_batch(data,'BMXWT')
    j = demo_features.index('BMXWT')
    x = data[j]
    y = Model_One(data)
    axs.set_xlim(min(xs), max(xs))
    axs.set_ylim(min(ys), max(ys))
    axs.plot(xs, ys,'o-',markersize=10)
    axs.plot(x, y,'o',markersize=12,color='#aa3333')
    axs.hlines(y, xmin=min(xs), xmax=x,linestyles='dashed',color='#aa3333')
    axs.vlines(x, ymin=min(ys), ymax=y,linestyles='dashed',color='#aa3333')
    axs.set_title('Body weight vs. Risk', size=20)
    axs.set_ylabel('Risk', fontsize=20)
    axs.set_xlabel('Body weight', fontsize=20)
    axs.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    return fig
  
def create_milk():
  
    fig, axs = plt.subplots(1,1,figsize=(8,8))
    # Change the following quote format
    xs, ys = predict_batch(data,'DBQ197')
    t = ['Never','Rarely','Sometimes','Often']
    axs.set_ylim(min(ys)*0.7, min(1,max(ys)*1.2))
    axs.bar(t, ys)
    j = demo_features.index('DBQ197')
    pos = np.argwhere(xs == data[j])[0][0]
    axs.patches[pos].set_facecolor('#aa3333')
    axs.set_title('Milk product consumption vs. Risk',size=20)
    axs.set_ylabel('Risk', fontsize=20)
    axs.set_xlabel('Frequency', fontsize=20)
    axs.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    return fig

def create_blood():

    fig, axs = plt.subplots(1,1,figsize=(8,8))
    xs, ys = predict_batch(data,'BPQ020')
    t = ['high','low']
    axs.set_ylim(min(ys)*0.7, min(1,max(ys)*1.2))
    axs.bar(t, ys)
    j = demo_features.index('BPQ020')
    pos = np.argwhere(xs == data[j])[0][0]
    axs.patches[pos].set_facecolor('#aa3333')
    axs.set_title('High blood pressure vs. Risk',size=20)
    axs.set_ylabel('Risk', fontsize=20)
    axs.set_xlabel('Blood pressure', fontsize=20)
    axs.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    return fig

def create_mental():
  
    fig, axs = plt.subplots(1,1,figsize=(8,8))
    xs, ys = predict_batch(data,'DPQ040')
    t = ['Never','Rarely','Sometimes','Often']
    axs.set_ylim(min(ys)*0.7, min(1,max(ys)*1.2))
    axs.bar(t, ys)
    j = demo_features.index('DPQ040')
    pos = np.argwhere(xs == data[j])[0][0]
    axs.patches[pos].set_facecolor('#aa3333')
    axs.set_title('Mental health vs. Risk',size=20)
    axs.set_ylabel('Risk', fontsize=20)
    axs.set_xlabel('Feeling tired or having little energy', fontsize=20)
    axs.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    return fig

def create_alcohol():
  
    fig, axs = plt.subplots(1,1,figsize=(8,8))
    xs, ys = predict_batch(data,'ALQ120Q')
    t = ['<1 a week','1 per week','2 per week','3 per week','>3 per week']
    axs.set_ylim(min(ys)*0.7, min(1,max(ys)*1.2))
    axs.bar(t, ys)
    j = demo_features.index('ALQ120Q')
    pos = np.argwhere(xs == data[j])[0][0]
    axs.patches[pos].set_facecolor('#aa3333')
    axs.set_title('Alcohol drinking vs. Risk',size=20)
    axs.set_ylabel('Risk', fontsize=20)
    axs.set_xlabel('Frequency of drinking alcohol', fontsize=20)
    axs.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    return fig

def create_sleep():
  
    fig, axs = plt.subplots(1,1,figsize=(8,8))
    # Change the following quote format
    xs, ys = predict_batch(data,'SLD010H')
    j = demo_features.index('SLD010H')
    x = data[j]
    y = Model_One(data)
    axs.set_xlim(min(xs), max(xs))
    axs.set_ylim(min(ys), max(ys))
    axs.plot(xs, ys,'o-',markersize=10)
    axs.plot(x, y,'o',markersize=12,color='#aa3333')
    axs.hlines(y, xmin=min(xs), xmax=x,linestyles='dashed',color='#aa3333')
    axs.vlines(x, ymin=min(ys), ymax=y,linestyles='dashed',color='#aa3333')
    axs.set_title('Sleep hour vs. Risk', size=20)
    axs.set_ylabel('Risk', fontsize=20)
    axs.set_xlabel('Sleep hour', fontsize=20)
    axs.tick_params(axis='both', which='major', labelsize=20)
    fig.tight_layout()
    return fig

@app.route('/predict')
def predict():
  global risk_1
  data = get_data()
  risk_1 = Model_One(data)
  if risk_1 < 0.4:
    msg = "Good news, you have healthy lifestyle and low risk!" 
  elif risk_1 > 0.6:
    msg = "Attention, we think you have high risk of diabetes, please make appointment with your doctor."
  else:
    msg = "You are at the margin of high diabetes risk, you may need take some actions to reduce the risk."
  return render_template("predict.html",the_result = risk_1, message = msg)

@app.route('/plot_weight.png')
def plot_weight():
    fig = create_weight()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_milk.png')
def plot_milk():
    fig = create_milk()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_blood.png')
def plot_blood():
    fig = create_blood()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
  
@app.route('/plot_mental.png')
def plot_mental():
    fig = create_mental()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_sleep.png')
def plot_sleep():
    fig = create_sleep()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot_alcohol.png')
def plot_alcohol():
    fig = create_alcohol()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
  
@app.route('/predict_next',methods=['POST'])
def predict_next():
  #pull 'data' from input field and store it
  global risk_2
  tmp_a = request.form["BMXWT_A"]
  tmp_b = request.form["BMXWT_B"]
  j = demo_features.index('BMXWT')
  if float(tmp_b) == 0:
    data[j] = data[j]*float(tmp_a)
  else:
    data[j] = float(tmp_b)
  tmp = request.form["SLD010H"]
  j = demo_features.index('SLD010H')
  data[j] = data[j]+float(tmp)
  tmp = request.form["DBQ197"]
  j = demo_features.index('DBQ197')
  data[j] = float(tmp)
  risk_2 = Model_One(data)
  p = int((risk_1-risk_2)*100)
  return render_template("predict_next.html",result = risk_2,p=p)

  








