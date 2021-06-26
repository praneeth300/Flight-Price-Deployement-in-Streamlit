# Machine-Learning-for-Flight-Ticket-Pricing-DST-1
## Project Report

![](img.jpg)


## Problem Statement
#### Can we use Machine Learning to help a customer decide the optimal time to purchase a flight ticket?



## ABSTRACT

Airlines employ complex, secretly-kept algorithms to vary flight ticket prices over time based on several factors,including seat availability,airline capacity, the price of oil, seasonality, etc. At any point in time, a customer looking to purchase a flight ticket has the option to buy or wait (in the hope of the flight price reducing in future).However, since they lack knowledge of these algorithms, customers often default to purchasing a ticket as early as possible rather than trying to optimize their time of purchase.However, vast quantities of data regarding flight ticket prices are available on the Internet. Through this project,we hoped to use this data to help customers make their decisions. We created an airline ticket-buying agent that tries to buy a customer’s flight ticket to optimize for price of purchase.We have selected ![MakeMyTrip](https://www.makemytrip.com/flights/) website to scrap the Indian flights data.

## Prerequisites

You need to have installed following softwares and libraries in your machine before running this project.

Python 3
Anaconda: It will install ipython notebook and most of the libraries which are needed like sklearn, pandas, seaborn, matplotlib, numpy, scipy,streamlit.

For more details refer repo path : Web App Model/Flask/requirements.txt

## Sample Data
Below is the small sample of our dataset:

![](Screenshot.png)

## Data Overview
#### Data Source --> Dataset/
#### Data points --> 330939 rows
#### Dataset date range --> April 2021 to May 2021
#### Dataset Attributes:

Price - flight price

departure_time	- flight schedule time

arrival_time - arrival time of flight

Airline	Cabin - There are three type

E - Economy

PE - Premium Economy

B - Business

Dept_city - Departure city

Dept_date - Departure Date	

arrival_city - Arrival city	

stops - Number of stops

duration - Flight duration in minutes

weekday	dept_hours	

Dept_flights_time	

optimal_hours

## BLUEPRINT	
The blueprint file structure follows the following pattern:
Data --> Data Processing-->EDA-->Training Model-->Test Model & Evaluation-->Model Prediction-->Model Deployment
![image](https://user-images.githubusercontent.com/67750027/120107633-700f7280-c17f-11eb-9088-76f539f84da7.png)
![image](https://user-images.githubusercontent.com/67750027/120107270-08a4f300-c17e-11eb-86ca-3b47fc205d9b.png)
![image](https://user-images.githubusercontent.com/67750027/120107281-1b1f2c80-c17e-11eb-8e7c-dacab91f4731.png)
![image](https://user-images.githubusercontent.com/67750027/120107416-9254c080-c17e-11eb-8721-37eae088bf80.png)
![image](https://user-images.githubusercontent.com/67750027/120107444-a7c9ea80-c17e-11eb-9d0b-993872f06a56.png)
![image](https://user-images.githubusercontent.com/67750027/120107464-bb755100-c17e-11eb-8864-d1222e01947b.png)
![image](https://user-images.githubusercontent.com/67750027/120107507-e495e180-c17e-11eb-9301-948b27c66294.png)


## Machine Learning Framework:

Assume a customer decides to purchase a ticket for a particular flight at time = X 
hours before departure. The optimal time to purchase the ticket t0pt is:
- in the range [X hours before dep., 4 hours before dep.]
- time at which we achieve minimum flight price until departure

We have used 'LightGBM' algorithm to predict first optimal time then to predict price for each of the cabin classes whose architecture is as below:

#### Predict optimal time architecture for Economy Class

ec=LGBMRegressor(n_estimators=1200)
#ec=RandomForestRegressor()
ec.fit(x_train,y_train)
#pred = rfg.predict(x_cv)
pred = ec.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2= r2_score(y_test, pred)
print("RMSE : % f" %(rmse))
print("R2 : % f" %(r2))

#### RMSE: 0.00000
#### R2: 1.0000

![image](https://user-images.githubusercontent.com/80449168/120228944-c527b300-c269-11eb-8c4b-23acbf0c0aa1.png)

#### Predict Price architecture for Economy Class

ec_price=LGBMRegressor(n_estimators=1200)
ec_price.fit(x_train,y_train)
pred = ec_price.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2= r2_score(y_test, pred)
print("RMSE : % f" %(rmse))
print("R2 : % f" %(r2))

#### RMSE: 0.141890
#### R2: 0.869850

![image](https://user-images.githubusercontent.com/80449168/120229193-39faed00-c26a-11eb-914d-1579b27805e7.png)

#### Predict Optimal Time architecture for Business Class

bs=LGBMRegressor(n_estimators=1000)
#bs=RandomForestRegressor(n_estimators=100 )
bs.fit(x_train,y_train)
#pred = rfg.predict(x_cv)
pred = bs.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2= r2_score(y_test, pred)
print("RMSE : % f" %(rmse))
print("R2 : % f" %(r2))

#### RMSE: 0.0000
#### R2: 1.00000

![image](https://user-images.githubusercontent.com/80449168/120229313-7dedf200-c26a-11eb-99b6-a3fe99303af7.png)

#### Predict Price architecture for Business Class

bs_price=LGBMRegressor(n_estimators=1000)
#bs=RandomForestRegressor(n_estimators=100 )
bs_price.fit(x_train,y_train)
#pred = rfg.predict(x_cv)
pred = bs_price.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2= r2_score(y_test, pred)
print("RMSE : % f" %(rmse))
print("R2 : % f" %(r2))

#### RMSE: 0.121185
#### R2: 0.899513

![image](https://user-images.githubusercontent.com/80449168/120229500-e4731000-c26a-11eb-9f09-d0f03bfda0d3.png)

#### Predict Optimal Time architecture for Premium Economy Class

pe=LGBMRegressor(n_estimators=1500)
#pe = CatBoostRegressor()
#rfg=RandomForestRegressor(n_estimators=100 )
pe.fit(x_train,y_train)
#pred = rfg.predict(x_cv)
pred = pe.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2= r2_score(y_test, pred)
print("RMSE : % f" %(rmse))
print("R2 : % f" %(r2))

#### RMSE: 0.00000
#### R2: 1.00000

#### Predict Price architecture for Premium Economy Class

pe=LGBMRegressor(n_estimators=1500)
#pe = CatBoostRegressor()
#rfg=RandomForestRegressor(n_estimators=100 )
pe.fit(x_train,y_train)
#pred = rfg.predict(x_cv)
pred = pe.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2= r2_score(y_test, pred)
print("RMSE : % f" %(rmse))
print("R2 : % f" %(r2))

#### RMSE: 0.093673
#### R2: 0.839008

## Final Model output on WebApp
## Using Flask Heroku Web App

Team Members : ![Ms. Deepika Goel](https://github.com/goeld9911)![Mr. Praneeth Kumar Pinni](https://github.com/praneeth300)!

#### Deployment Steps : 

To deploy model on Heroku we have 2 options, by Heroku CLI or by GitHub. We have selected deployment by GitHub

Step 1 : Create an account on heroku.com

Step 2 : Upload all files on GitHub 

Step 3 : Deployment method --> GitHub

Step 4 : App connected to GitHub 

Step 5 : Select Manual deploy Deploy--> the current state of a branch to this app should be Master.

Step 6 : Resolve package error if occurs and test your Pulic URL

Flask Code : Web App Model/Flask/

Public URL : https://prediction-price-for-flight.herokuapp.com/

### Frontend Of the Streamlit
https://docs.google.com/presentation/d/15HfriKFJ5acUQJ1qqCTX2-4JDUT5InTOfC6PH3KF9EE/e

**Here the results template**
"https://docs.google.com/presentation/d/15HfriKFJ5acUQJ1qqCTX2-4JDUT5InTOfC6PH3KF9EE/edit#slide=id.gb69d85bd22_0_12"

#### Demo:

https://user-images.githubusercontent.com/80449168/120231906-002ce500-c270-11eb-9669-58b80fb37599.mp4

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Team Members : ![Mr. Prasad Pawar.](https://github.com/Prasad993)![Mr. Makarand Anna Rayate.](https://github.com/mak-rayate)![Mr. Rudra Kumawat.](https://github.com/Rudra-kumawat-22)

We have used `RandomForestRegressor` algorithm to predict first optimal time then to predict price whose architecture is as below:

#### Predict optimal time architecture

    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  37 tasks      | elapsed: 15.4min
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed: 19.7min finished
    RandomizedSearchCV(cv=5, error_score=nan,
                       estimator=RandomForestRegressor(bootstrap=True,
                                                       ccp_alpha=0.0,
                                                       criterion='mse',
                                                       max_depth=None,
                                                       max_features='auto',
                                                       max_leaf_nodes=None,
                                                       max_samples=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0,
                                                       n_estimators=100,
                                                       n_jobs=None, oob_score=False,
                                                       random_state=None, verbose=0,
                                                       warm_start=False),
                       iid='deprecated', n_iter=10, n_jobs=-1,
                       param_distributions={'max_depth': [5, 10, 15, 20, 50],
                                            'min_samples_split': [2, 3, 5, 10]},
                       pre_dispatch='2*n_jobs', random_state=None, refit=True,
                       return_train_score=False, scoring=None, verbose=2)

Selected best_params_ after hyperparameter tunning : `{'min_samples_split': 5, 'max_depth': 20}`
 
###### Accuracy = 0.9017739133612731

#### Predict price architecture

    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  37 tasks      | elapsed: 14.5min
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed: 19.6min finished
    RandomizedSearchCV(cv=5, error_score=nan,
                       estimator=RandomForestRegressor(bootstrap=True,
                                                       ccp_alpha=0.0,
                                                       criterion='mse',
                                                       max_depth=None,
                                                       max_features='auto',
                                                       max_leaf_nodes=None,
                                                       max_samples=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0,
                                                       n_estimators=100,
                                                       n_jobs=None, oob_score=False,
                                                       random_state=None, verbose=0,
                                                       warm_start=False),
                       iid='deprecated', n_iter=10, n_jobs=-1,
                       param_distributions={'max_depth': [5, 10, 15, 20, 50],
                                            'min_samples_split': [2, 3, 5, 10]},
                       pre_dispatch='2*n_jobs', random_state=None, refit=True,
                       return_train_score=False, scoring=None, verbose=2)
                       
Selected best_params_ after hyperparameter tunning : `{'min_samples_split': 5, 'max_depth': 20}`

##### Accuracy = 0.9351213338653643

## Final Model output on WebApp
## Using Flask Heroku Web App



#### Deployment Steps : 

To deploy model on Heroku we have 2 options, by Heroku CLI or by GitHub. We have selected deployment by GitHub

Step 1 : Create an account on heroku.com

Step 2 : Upload all files on GitHub 

Step 3 : Deployment method --> GitHub

Step 4 : App connected to GitHub 

Step 5 : Select Manual deploy Deploy--> the current state of a branch to this app should be Master.

Step 6 : Resolve package error if occurs and test your Pulic URL

Flask Code : Web App Model/Flask/

Public URL : https://mlflightpred.herokuapp.com/

#### Demo :

https://user-images.githubusercontent.com/67750027/119845145-a4d4bd00-bf26-11eb-951d-f1b7caeaff32.mp4


### Steps that we performed:

- [x] Web scrapped
- [x] Data Loading
- [x] Data Preprocessing
- [x] Exploratory data analysis
- [x] Feature engineering
- [x] Feature selection
- [x] Feature transformation
- [x] Model building
- [x] Model evalutaion
- [x] Model tuning
- [x] Prediction's
- [x] Model deployement Flask & Heroku
- [x] Published the URL
- [x] Submitting the Reports using Tableu

### Tools used:
> Python <br>
> Pycharm <br>
> Jupyter Notebook <br>
> Google Colab <br>
> DataBricks <br>
> Streamlit <br>
> Flask <br>
> GitHub <br>
> GitBash <br>
> SublimeTextEditor <br>
<br>
### Libraries used: <br>
* Pandas <br>
* Numpy <br>
* scipy <br>
* sklearn <br>
* lightgbm <br>
* Boosting <br>
* selenium <br>
* Matplotlib <br>
* Seaborn <br>
* Plotly <br>
* Cufflinks <br>

Commands that we used for deployement:
```
git init
git add .
git status
git commit -m "First commit"
git status

heroku create
git remote -v
git push origin master

heorku logs --tail
```

Procfile:
```
web: sh setup.sh && streamlit run gh.py
```

Setup.sh:
``` 
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

## Author
-Yasin Shah

## DECLARATION

#### A project report on Machine-Learning-for-Flight-Ticket-Pricing project Successfully submitted By

![Ms. Deepika Goel.](https://github.com/goeld9911/)

![Mr. Prasad Pawar.](https://github.com/Prasad993)

![Mr. Makarand Anna Rayate.](https://github.com/mak-rayate)

![Mr. Chakradhar Reddy Yerragudi.](https://github.com/chakradhar123)

![Mr. Mervana Prit Jitendrabhai.](https://github.com/Prit005)

![Mr. Praneeth Kumar.](https://github.com/praneeth300?tab=repositories)

![Ms. Himadri Chutia.](https://github.com/Himadrichutia)

![Mr. Rudra Kumawat.](https://github.com/Rudra-kumawat-22)
