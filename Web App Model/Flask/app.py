from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from joblib import dump, load
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import zipfile

app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('travel1.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        dcity=request.form['Source_city'].lower()
        acity=request.form['Arrival_city'].lower()
        cabin=request.form['Cabin']
        date=request.form['Date']
        time=request.form['Time']
        month=pd.to_datetime(date).month
        hour=time
        day=pd.to_datetime(date).day
        weekday=pd.to_datetime(date).dayofweek
        time=int(time)
        if time>=16 and time<21:
            temp_time = 1         
        if time>=21 or time<5:
            temp_time = 3    
        if time>=5 and time<11:
            temp_time = 2
        if time>=11 and time<16:
            temp_time = 0
        day=np.asarray(day)
        hour=np.asarray(hour)
        
        
        with zipfile.ZipFile('B/B_price_predict.zip', 'r') as zip_ref:
            zip_ref.extractall('B/')
                
        
        with zipfile.ZipFile('B/B_time_predict.zip', 'r') as zip_ref1:
            zip_ref1.extractall('B/')
                
        
        with zipfile.ZipFile('E/E_price_predict.zip', 'r') as zip_ref2:
            zip_ref2.extractall('E/')
                
        
        with zipfile.ZipFile('E/E_time_predict.zip', 'r') as zip_ref3:
            zip_ref3.extractall('E/')
            
                
        
        with zipfile.ZipFile('PE/PE_price_predict.zip', 'r') as zip_ref4:
            zip_ref4.extractall('PE/')
                
        
        with zipfile.ZipFile('PE/PE_time_predict.zip', 'r') as zip_ref5:
            zip_ref5.extractall('PE/')
        
        if cabin=='B':
            print("HII_B")
            with open("B/B_price_predict.pkl","rb") as f:
                model_price=pickle.load(f)
            with open("B/B_time_predict.pkl","rb") as f:
                model_time=pickle.load(f)
            with open("B/B_airline_dict.pkl","rb") as f:
                airline_dict=pickle.load(f)
            with open("B/B_arrival_city_dict.pkl","rb") as f:
                arrival_dict=pickle.load(f)
            with open("B/B_dept_city_dict.pkl","rb") as f:
                dept_dict=pickle.load(f)
            sc=load("B/B_Dept_date_std_scaler.bin")
            Dept_date=sc.transform(day.reshape(-1,1))

            sc=load("B/B_dept_hours_std_scaler.bin")
            dept_hours=sc.transform(hour.reshape(-1,1))

            with open('B/B_duration.pkl',"rb") as f:
                duration=pkl.load(f)
            
            k_0=dcity+","+acity+" ,"+str(0)
            k_1=dcity+","+acity+" ,"+str(1)
            k_0=np.asarray(duration[k_0])
            k_1=np.asarray(duration[k_1])
            sc=load('B/B_duration_std_scaler.bin')
            scale_0=sc.transform(k_0.reshape(-1,1))
            sc=load('B/B_duration_std_scaler.bin')
            scale_1=sc.transform(k_1.reshape(-1,1))
            for j in range(2):
                p_0=model_time.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,0,scale_0 ,weekday,dept_hours,temp_time]])

            for j in range(2):
                p_1=model_time.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,1,scale_1 ,weekday,dept_hours,temp_time]])

            
            sc=load('B/B_price_dept_hours_std_scaler.bin')
            h_0=sc.transform(p_0.reshape(-1,1))
            h_1=sc.transform(p_1.reshape(-1,1))
            #x = PrettyTable(
            #x.field_names=["Airline", "Stops", "Source_city", "Arrival_city","Duration","Optimal_time","Price"]
            
            headings=("Airline", "Stops", "Source_city", "Arrival_city","Duration","Optimal_time","Price")
            data=[]
            
            for j in range(2):
                pred=model_price.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,0,scale_0 ,weekday,h_0,temp_time]])
                data.append([airline_dict[j].capitalize() ,0, dcity.capitalize(),acity.capitalize(),k_0 ,int(p_0[0]) ,"₹"+str(int(pred[0]))])

            for j in range(2):
                pred=model_price.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,1,scale_1 ,weekday,h_1,temp_time]])
                data.append([airline_dict[j].capitalize() ,1, dcity.capitalize(),acity.capitalize(),k_1 ,int(p_1[0]) ,"₹"+str(int(pred[0]))])
            data=tuple(data)





        
        if cabin=='PE':
            with open("PE/PE_price_predict.pkl","rb") as f:
                model_price=pickle.load(f)
            with open("PE/PE_time_predict.pkl","rb") as f:
                model_time=pickle.load(f)
            with open("PE/PE_airline_dict.pkl","rb") as f:
                airline_dict=pickle.load(f)
            with open("PE/PE_arrival_city_dict.pkl","rb") as f:
                arrival_dict=pickle.load(f)
            with open("PE/PE_dept_city_dict.pkl","rb") as f:
                dept_dict=pickle.load(f)
            sc=load("PE/PE_Dept_date_std_scaler.bin")
            Dept_date=sc.transform(day.reshape(-1,1))

            sc=load("PE/PE_dept_hours_std_scaler.bin")
            dept_hours=sc.transform(hour.reshape(-1,1))

            with open('PE/PE_duration.pkl',"rb") as f:
                duration=pkl.load(f)
             
            k_0=dcity+","+acity+" ,"+str(0)
            k_1=dcity+","+acity+" ,"+str(1)
            k_0=np.asarray(duration[k_0])
            k_1=np.asarray(duration[k_1])
            sc=load('PE/PE_duration_std_scaler.bin')
            scale_0=sc.transform(k_0.reshape(-1,1))
            sc=load('PE/PE_duration_std_scaler.bin')
            scale_1=sc.transform(k_1.reshape(-1,1))

            for j in range(2):
                p_0=model_time.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,0,scale_0 ,weekday,dept_hours,temp_time]])

            for j in range(2):
                p_1=model_time.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,1,scale_1 ,weekday,dept_hours,temp_time]])

            
            sc=load('PE/PE_price_dept_hours_std_scaler.bin')
            h_0=sc.transform(p_0.reshape(-1,1))
            h_1=sc.transform(p_1.reshape(-1,1))
            #x = PrettyTable(
            #x.field_names=["Airline", "Stops", "Source_city", "Arrival_city","Duration","Optimal_time","Price"]
            
            headings=("Airline", "Stops", "Source_city", "Arrival_city","Duration","Optimal_time","Price")
            data=[]
            for j in range(2):
                pred=model_price.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,0,scale_0 ,weekday,h_0,temp_time]])
                data.append([airline_dict[j].capitalize() ,0, dcity.capitalize(),acity.capitalize(),k_0 ,int(p_0[0]) ,"₹"+str(int(pred[0]))])

            for j in range(2):
                pred=model_price.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,1,scale_1 ,weekday,h_1,temp_time]])
                data.append([airline_dict[j].capitalize() ,1, dcity.capitalize(),acity.capitalize(),k_1 ,int(p_1[0]) ,"₹"+str(int(pred[0]))])
            data=tuple(data)
            


        if cabin=='E':
            print("HII_E")
            with open("E/E_price_predict.pkl","rb") as f:
                model_price=pickle.load(f)
            with open("E/E_time_predict.pkl","rb") as f:
                model_time=pickle.load(f)
            with open("E/E_airline_dict.pkl","rb") as f:
                airline_dict=pickle.load(f)
            with open("E/E_arrival_city_dict.pkl","rb") as f:
                arrival_dict=pickle.load(f)
            with open("E/E_dept_city_dict.pkl","rb") as f:
                dept_dict=pickle.load(f)
            sc=load("E/E_Dept_date_std_scaler.bin")
            Dept_date=sc.transform(day.reshape(-1,1))

            sc=load("E/E_dept_hours_std_scaler.bin")
            dept_hours=sc.transform(hour.reshape(-1,1))

            with open('E/E_duration.pkl',"rb") as f:
                duration=pkl.load(f)
             
            k_0=dcity+","+acity+" ,"+str(0)
            k_1=dcity+","+acity+" ,"+str(1)
            k_0=np.asarray(duration[k_0])
            k_1=np.asarray(duration[k_1])
            sc=load('E/E_duration_std_scaler.bin')
            scale_0=sc.transform(k_0.reshape(-1,1))
            sc=load('E/E_duration_std_scaler.bin')
            scale_1=sc.transform(k_1.reshape(-1,1))
            print("HII_E_END")


            for j in range(2):
                p_0=model_time.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,0,scale_0 ,weekday,dept_hours,temp_time]])

            for j in range(2):
                p_1=model_time.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,1,scale_1 ,weekday,dept_hours,temp_time]])

            sc=load('E/E_price_dept_hours_std_scaler.bin')
            h_0=sc.transform(p_0.reshape(-1,1))
            h_1=sc.transform(p_1.reshape(-1,1))
 
            
            headings=("Airline", "Stops", "Source_city", "Arrival_city","Duration","Optimal_time","Price")
            data=[]

            for j in range(5):
                pred=model_price.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,0,scale_0 ,weekday,h_0,temp_time]])
                data.append([airline_dict[j].capitalize() ,0, dcity.capitalize(),acity.capitalize(),k_0 ,int(p_0[0]) ,"₹"+str(int(pred[0]))])

            for j in range(5):
                pred=model_price.predict([[ j ,dept_dict[dcity],Dept_date, arrival_dict[acity] ,1,scale_1 ,weekday,h_1,temp_time]])
                data.append([airline_dict[j].capitalize() ,1, dcity.capitalize(),acity.capitalize(),k_1 ,int(p_1[0]) ,"₹"+str(int(pred[0]))])
            data=tuple(data)
         
        
        return render_template('travel1.html',headings=headings,data=data)
    #else:
     #   return render_template('travel1.html',="NON FRAUD CLAIM")
     
       

if __name__=="__main__":
    app.run(debug=True)

