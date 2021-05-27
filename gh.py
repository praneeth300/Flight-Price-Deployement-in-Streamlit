
#import the libraries that are required
import streamlit as st
import datetime
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import streamlit_theme as stt

#---------------------------------Page title------------------------------------------

st.set_page_config(page_title="Flight Price Prediction Using reinforcement learning")

st.title("Flight Price Prediction Using reinforcement learning")
img1 = Image.open('new_flight.jpg')
img1 = img1.resize((700,350))
st.image(img1, use_column_width=False)

#----------------------------------------Load pickle models---------------------------
#-----------Price pickles-----------
economy_model_price = pickle.load(open("EC_PRICE_MODEL.pickle","rb"))
bussiness_model_price = pickle.load(open("BUSSINESS_PRICE_MODEL.pickle","rb"))
pe_model_price = pickle.load(open("PE_PRICE_MODEL.pickle","rb"))

#-------------Time models-------------------
economy_model_time = pickle.load(open("EC_time_model.pickle","rb"))
bussiness_model_time = pickle.load(open("BUSSINE_TIME_MODEL.pickle","rb"))
pe_model_time = pickle.load(open("PE_TIME_MODEL.pickle","rb"))

#-----------------------------------------Load encoding files--------------------------
ec_encoding_dep = pickle.load(open("EC_dep_city_label.pickle","rb"))
ec_encoding_arr = pickle.load(open("EC_arr_city_label.pickle","rb"))
ec_encoding_airline = pickle.load(open("EC_ARLINE_CITY_label.pickle","rb"))

bs_encoding_dep = pickle.load(open("BS_DEP_CITY.pickle","rb"))
bs_encoding_arr = pickle.load(open("BS_ARR_CITY.pickle","rb"))
bs_encosing_airline = pickle.load(open("BS_Airline.pickle","rb"))

pe_encoding_dep = pickle.load(open("PE_DEP_CITY.pickle","rb"))
pe_encoding_arr = pickle.load(open("PE_ARR_CITY.pickle","rb"))
pe_encoding_airline = pickle.load(open("PE_AIRLINE.pickle","rb"))
#------------------------------------------Inputs-------------------------------------

dcity = st.selectbox("Departure City",('New Delhi', 'Mumbai', 'Bengaluru', 'Hyderabad', 'Kolkata', 'Chennai', 'Patna', 'Srinagar', 'Goa',
    'Lucknow', 'Guwahati', 'Amritsar', 'Kochi', 'Pune', 'Jaipur', 'Varanasi', 'Bhubaneswar', 'Bagdogra',
    'Chandigarh', 'Ranchi', 'Visakhapatnam', 'Indore', 'Raipur', 'Nagpur', 'Blair', 'Mangalore', 'Coimbatore',
    'Thiruvananthapuram', 'Kozhikode', 'Tiruchirappalli'))

acity = st.selectbox("Arrival City",('New Delhi ', 'Mumbai ', 'Bengaluru ', 'Kolkata ', 'Chennai ',
                       'Hyderabad ', 'Patna ', 'Lucknow ', 'Bhubaneswar ', 'Goa ', 'Guwahati ',
                       'Amritsar ', 'Srinagar ', 'Jaipur ', 'Kochi ', 'Pune ',
                       'Visakhapatnam ', 'Varanasi ', 'Raipur ', 'Ranchi ', 'Chandigarh ',
                       'Bagdogra ', 'Port Blair ', 'Indore ', 'Nagpur ', 'Coimbatore ',
                       'Mangalore ', 'Thiruvananthapuram ', 'Tiruchirappalli ', 'Kozhikode '))

today = datetime.date.today()

date = st.date_input("Date you want to flight",today)

hour2 = st.time_input("Enter the time",datetime.time(0,00))

cab = st.selectbox("Cabin",("E","B","PE"))

stops = int(st.selectbox("Stops",(0,1,2,3,4,5,6)))

#---------------------------------------End of inputs-------------------------------------------

hour=int(hour2.hour)
day=pd.to_datetime(date).day
weekday=pd.to_datetime(date).dayofweek

def time(hour):
  if hour>=0 and hour <=10:
    return 2
  elif hour >= 10 and hour <=15:
    return 0
  elif hour >= 16 and hour <=20:
    return 1
  elif hour >= 21 and hour <= 23:
    return 3
  else:
    return None

flight_time = time(hour)
optimal_hour=hour-4

#---------------------------------------Extract data-----------------------------------------


#---------------------------------------Prediction phase-------------------------------------
def run():
    pred_final=[]
    if st.button("Check"):
      if cab == "E":
        for j in range(6):
            if dcity in ec_encoding_dep.keys():
                if acity in pe_encoding_arr.keys():
                  p_time=economy_model_time.predict([[ j ,ec_encoding_dep[dcity] ,day ,ec_encoding_arr[acity] , stops , weekday ,hour,flight_time]])
                  p_load = p_time.reshape(-1,1)
                  p_price = economy_model_price.predict(
                      [[j, ec_encoding_dep[dcity], day, ec_encoding_arr[acity], stops, weekday, p_load, flight_time]])
                  #st.info(f"Airline: {ec_encoding_airline[j]}  {dcity} <---->  {acity} Optimal hour {i} Price {round(np.exp(p[0]),2)}")
                  pred_final.append([ec_encoding_airline[j] ,dcity,acity ,p_time[0],np.exp(p_price[0])])
                else:
                    st.write(f"Sorry, Please check another route currently we don't have a service to this {dcity} <---> {acity} ")
                    break
            else:
                st.write(f"Sorry, Please check another route currently we don't have a service to this {dcity} <---> {acity} ")
                break
      elif cab == "B":
        for j in range(2):
            if dcity in bs_encoding_dep.keys():
                if acity in bs_encoding_arr.keys():
                  p_time=bussiness_model_time.predict([[ j ,bs_encoding_dep[dcity] ,day ,bs_encoding_arr[acity] , stops , weekday ,hour,flight_time]])
                  p_load=p_time.reshape(-1,1)
                  p_price = bussiness_model_price.predict(
                      [[j, ec_encoding_dep[dcity], day, ec_encoding_arr[acity], stops, weekday, p_load, flight_time]])
                  #st.write(f"Airline: {bs_encosing_airline[j]}  {dcity} <---->  {acity} Optimal hour {i} Price {round(np.exp(p[0]),2)}")
                  pred_final.append([bs_encosing_airline[j] ,dcity,acity ,p_time[0],np.exp(p_price[0])])
                else:
                  st.write(f"Sorry, Please check another route currently we don't have a service to {dcity} <---> {acity} ")
                  break
            else:
                st.write(f"Sorry, Please check another route currently we don't have a service to this {dcity} <---> {acity} ")
                break
      elif cab == "PE":
        for j in range(2):
            if dcity in pe_encoding_dep.keys():
                if acity in pe_encoding_arr.keys():
                  p_time=pe_model_time.predict([[ j ,pe_encoding_dep[dcity] ,day ,pe_encoding_arr[acity] , stops , weekday ,hour,flight_time]])
                  p_load=p_time.reshape(-1,1)
                  p_price = pe_model_price.predict(
                      [[j, ec_encoding_dep[dcity], day, ec_encoding_arr[acity], stops, weekday, p_load, flight_time]])
                  #st.success(f"Airline: {pe_encoding_airline[j]}  {dcity} <---->  {acity} Optimal hour {i} Price {round(np.exp(p[0]),2)}")
                  pred_final.append([pe_encoding_airline[j] ,dcity,acity ,p_time[0],np.exp(p_price[0])])
                else:
                    st.write(f"Sorry, Please check another route currently we don't have a service to this {dcity} <---> {acity} ")
                    break
            else:
                st.write(f"Sorry, Please check another route currently we don't have a service to this {dcity} <---> {acity} ")
                break
      else:
        return None

    if param_ascend == "Low to High":
        pred = pd.DataFrame(pred_final,columns=["Airline","Departure City","Arrival City","Optimal time","Price"])
        pred_2 = pred.sort_values(by="Price",ascending=True)
        pred_22 = pred_2.reset_index(drop=True)
        st.dataframe(pred_22)
        #for i in range(len(pred_2)):
            #st.write(f"{pred_2.iloc[i,0]}   **{pred_2.iloc[i,1]}** -----> **{pred_2.iloc[i,2]}** Optimal hour: {pred_2.iloc[i,4]} Price = **{round(pred_2.iloc[i,5],2)}**")

    elif param_ascend == "High to Low":
        pred = pd.DataFrame(pred_final, columns=["Airline","Departure City","Arrival City","Optimal time","Price"])
        pred_2 = pred.sort_values(by="Price", ascending=False)
        pred_22 = pred_2.reset_index(drop=True)
        st.dataframe(pred_22)
        #for i in range(len(pred_2)):
            #st.write(
                #f"{pred_2.iloc[i, 0]}    **{pred_2.iloc[i, 1]}** <====> **{pred_2.iloc[i, 2]}** Optimal hour: {pred_2.iloc[i, 4]} Price = **{round(pred_2.iloc[i, 5], 2)}**")
    else:
        return None

#-----------------------------------Sidebar------------------------------------------

st.sidebar.header("Project on Flight Price Prediction")
st.sidebar.write("Powered by Technocolabs")
with st.sidebar.subheader("Filter Price"):
    param_ascend = st.sidebar.selectbox("Price",options=["Low to High","High to Low"])
st.sidebar.subheader("Sources")
st.sidebar.markdown("""
[Github for this project](https://github.com/Technocolabs100/Reinforcement-Learning-for-Flight-Ticket-Pricing-DST-1)""")
st.sidebar.markdown("""
[Linkedln](https://www.linkedin.com/company/technocolabs/)""")
st.sidebar.markdown("""
[Project Architeture](https://drive.google.com/file/d/1CvPsTD9EYfvhREMhK1GsoOqIZxVb0dfT/view)""")
st.sidebar.markdown("""
[Reasearch Paper](https://drive.google.com/file/d/1lf7GRaAmg5lwJF3Ik7irdFncvgc64nzo/view?usp=sharing)""")
st.sidebar.markdown("""
[Tablue Report](https://public.tableau.com/app/profile/deepika.goel/viz/flights_data_analysis/Dashboard1)""")

st.sidebar.subheader("Presented by:")
st.sidebar.write("1. Praneeth kumar Pinni")
st.sidebar.write("2. Deepika Goel")

run()

