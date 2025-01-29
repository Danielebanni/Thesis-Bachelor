#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgboost')


# In[18]:


get_ipython().system('pip install catboost')


# In[2]:


import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import random
import math
from math import sqrt
import seaborn as sns
from sklearn import datasets
from sklearn import svm, datasets
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# In[3]:


# Carica il file Excel "Sample_Submission"
sample_submission_data = pd.read_excel("C:\\Users\\bdani\\Desktop\\Tesi triennale - Copia\\eCommerce Sales Data\\Sample_Submission.xlsx")

# Carica i CSV nei DataFrame di pandas
training_data = pd.read_csv("C:\\Users\\bdani\\Desktop\\Tesi triennale - Copia\\eCommerce Sales Data\\Train.csv")
test_data = pd.read_csv("C:\\Users\\bdani\\Desktop\\Tesi triennale - Copia\\eCommerce Sales Data\\Test.csv")

# Visualizza i primi 5 record di ogni DataFrame per verificare che tutto sia stato importato correttamente
print("Training Data:")
print(training_data.head())

print("\nTest Data:")
print(test_data.head())

# Estrai la colonna 'Selling_Price' dal DataFrame del file Excel
true_selling_prices = sample_submission_data['Selling_Price']


# In[4]:


training_data.T

sns.set(style='whitegrid', palette='muted')
fig, ax = plt.subplots(1,2, figsize=(12,6))

sns.distplot(training_data['Selling_Price'], kde=True, ax=ax[0])
sns.scatterplot(x='Item_Rating', y='Selling_Price', data=training_data, marker='o', color='r', ax=ax[1])

plt.tight_layout()
plt.show()


# In[5]:


# Applicare la trasformazione logaritmica
training_data['Log_Selling_Price'] = np.log(training_data['Selling_Price'])

# Creare un istogramma della variabile trasformata
plt.figure(figsize=(10, 6))
sns.histplot(training_data['Log_Selling_Price'], bins=30, color='green', kde=True, edgecolor='black')
plt.title('Distribuzione dei Prezzi di Vendita (Trasformazione Logaritmica)')
plt.xlabel('Logaritmo del Prezzo di Vendita')
plt.ylabel('Frequenza')
plt.grid(True)
plt.show()


# In[6]:


training_data.describe().T


# In[7]:


# Convertire la colonna 'Date' in datetime
training_data['Date'] = pd.to_datetime(training_data['Date'], format='%d/%m/%Y')

# Aggiungere colonne con le variabili temporali
training_data['Month'] = training_data['Date'].dt.month
training_data['Day'] = training_data['Date'].dt.day
training_data['DayofYear'] = training_data['Date'].dt.dayofyear
training_data['Week'] = training_data['Date'].dt.isocalendar().week
training_data['Quarter'] = training_data['Date'].dt.quarter
training_data['Is_month_start'] = training_data['Date'].dt.is_month_start
training_data['Is_month_end'] = training_data['Date'].dt.is_month_end

# Eliminare la colonna 'Date'
training_data = training_data.drop(columns=['Date'])


# Visualizzare il dataframe risultante
training_data


# In[8]:


# Codifica one-shot delle colonne specificate
training_data['Unique_Item_category_per_product_brand'] = pd.factorize(training_data['Item_Category'])[0]
training_data['Unique_Subcategory_1_product_brand'] = pd.factorize(training_data['Subcategory_1'])[0]
training_data['Unique_Subcategory_2_product_brand'] = pd.factorize(training_data['Subcategory_2'])[0]


# Visualizza il dataframe risultante
training_data


# In[9]:


#Convertire la colonna 'Date' in datetime
test_data['Date'] = pd.to_datetime(test_data['Date'], format='%d/%m/%Y')

# Aggiungere colonne con le variabili temporali
test_data['Month'] = test_data['Date'].dt.month
test_data['Day'] = test_data['Date'].dt.day
test_data['DayofYear'] = test_data['Date'].dt.dayofyear
test_data['Week'] = test_data['Date'].dt.isocalendar().week
test_data['Quarter'] = test_data['Date'].dt.quarter
test_data['Is_month_start'] = test_data['Date'].dt.is_month_start
test_data['Is_month_end'] = test_data['Date'].dt.is_month_end

# Eliminare la colonna 'Date'
test_data = test_data.drop(columns=['Date'])


# Codifica one-shot delle colonne specificate
test_data['Unique_Item_category_per_product_brand'] = pd.factorize(test_data['Item_Category'])[0]
test_data['Unique_Subcategory_1_product_brand'] = pd.factorize(test_data['Subcategory_1'])[0]
test_data['Unique_Subcategory_2_product_brand'] = pd.factorize(test_data['Subcategory_2'])[0]



# Visualizza il dataframe risultante
test_data


# In[11]:


# Definizione delle features

X_train = training_data[['Item_Rating', 'Month', 'Day', 'DayofYear', 'Is_month_start', 'Is_month_end','Unique_Item_category_per_product_brand', 'Unique_Subcategory_1_product_brand', 'Unique_Subcategory_2_product_brand' ]]
X_test = test_data[['Item_Rating', 'Month', 'Day', 'DayofYear', 'Is_month_start', 'Is_month_end','Unique_Item_category_per_product_brand', 'Unique_Subcategory_1_product_brand', 'Unique_Subcategory_2_product_brand' ]]

# Definizione del target
y_train = training_data['Log_Selling_Price']


# In[29]:


# Definisci il modello XGBoost
xgb_regressor = XGBRegressor(random_state=42)

# Parametri per la ricerca
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# GridSearchCV con 5-fold cross-validation
grid_search_xgb = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid_xgb, cv=5, 
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

# Ottieni i migliori parametri
best_params_xgb = grid_search_xgb.best_params_
print(f"Migliori parametri trovati: {best_params_xgb}")


# In[30]:


# Addestramento del Random Forest Regressor con i migliori parametri
best_params_xgb = {'colsample_bytree': 0.6, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.6}

# Addestra il modello con i migliori parametri
best_xgb_regressor = XGBRegressor(**best_params_xgb, random_state=42)
best_xgb_regressor.fit(X_train, y_train)

# Previsioni sui dati di test
predictions_log_xgb = best_xgb_regressor.predict(X_test)
#predictions_non_log_xgb = np.exp(predictions_log_xgb)
# Definisci il numero di iterazioni di resampling
num_iterations = 10
mse_scores = []
rmse_scores = []
baseline_rmse_scores = []



for i in range(num_iterations):
    # Dividi il dataset di training in training e validation set
    X_train_resample, X_val_resample, y_train_resample, y_val_resample = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    # Addestra il modello sul training set di resampling
    best_xgb_regressor = XGBRegressor(**best_params_xgb, random_state=42)
    best_xgb_regressor.fit(X_train_resample, y_train_resample)
    
    # Valuta il modello sul validation set di resampling
    predictions = best_xgb_regressor.predict(X_val_resample)
    mse = mean_squared_error(y_val_resample, predictions)
    rmse = np.sqrt(mse)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    
    # Calcola la baseline RMSE in scala logaritmica
    baseline_pred_log = np.mean(y_train_resample)  # Media dei logaritmi dei valori target nel training set
    baseline_predictions_log = np.full_like(y_val_resample, baseline_pred_log)
    baseline_rmse_log = np.sqrt(mean_squared_error(y_val_resample, baseline_predictions_log))
    baseline_rmse_scores.append(baseline_rmse_log)

# Calcola la media e la deviazione standard delle MSE ottenute
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
mean_baseline_rmse = np.mean(baseline_rmse_scores)
std_baseline_rmse = np.std(baseline_rmse_scores)

print(f"Mean MSE over {num_iterations} iterations: {mean_mse}")
print(f"Standard Deviation of MSE over {num_iterations} iterations: {std_mse}")
print(f"Mean RMSE over {num_iterations} iterations: {mean_rmse}")
print(f"Standard Deviation of RMSE over {num_iterations} iterations: {std_rmse}")
print(f"Mean Baseline RMSE over {num_iterations} iterations: {mean_baseline_rmse}")
print(f"Standard Deviation of Baseline RMSE over {num_iterations} iterations: {std_baseline_rmse}")


# In[32]:


# Definisci il RandomForestRegressor e i parametri per la ricerca
rf_regressor = RandomForestRegressor(random_state=42)
# Parametri per la random search
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': ['auto', 'sqrt', 'log2']
}

# RandomizedSearchCV con 100 iterazioni
random_search = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_dist, 
                                   n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=2)
random_search.fit(X_train, y_train)

# Ottieni i migliori parametri
best_params = random_search.best_params_
print(f"Migliori parametri trovati: {best_params}")


# In[34]:


# Addestramento del Random Forest Regressor con i migliori parametri
best_params = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 323}
# Addestra il modello con i migliori parametri
best_rf_regressor = RandomForestRegressor(**best_params, random_state=42)
best_rf_regressor.fit(X_train, y_train)


# Definisci il numero di iterazioni di resampling
num_iterations = 10
mse_scores = []
rmse_scores = []
baseline_rmse_scores = []


for i in range(num_iterations):
    # Dividi il dataset di training in training e validation set
    X_train_resample, X_val_resample, y_train_resample, y_val_resample = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    # Addestra il modello sul training set di resampling
    best_rf_regressor.fit(X_train_resample, y_train_resample)
    
    # Valuta il modello sul validation set di resampling
    predictions = best_rf_regressor.predict(X_val_resample)
    mse = mean_squared_error(y_val_resample, predictions)
    rmse = np.sqrt(mse)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
 
    # Calcola la baseline RMSE in scala logaritmica
    baseline_pred_log = np.mean(y_train_resample)  # Media dei logaritmi dei valori target nel training set
    baseline_predictions_log = np.full_like(y_val_resample, baseline_pred_log)
    baseline_rmse_log = np.sqrt(mean_squared_error(y_val_resample, baseline_predictions_log))
    baseline_rmse_scores.append(baseline_rmse_log)


# Calcola la media e la deviazione standard delle MSE ottenute
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
print(f"Mean MSE over {num_iterations} iterations: {mean_mse}")
print(f"Standard Deviation of MSE over {num_iterations} iterations: {std_mse}")
print(f"Mean RMSE over {num_iterations} iterations: {mean_rmse}")
print(f"Standard Deviation of RMSE over {num_iterations} iterations: {std_rmse}")
print(f"Mean Baseline RMSE over {num_iterations} iterations: {mean_baseline_rmse}")
print(f"Standard Deviation of Baseline RMSE over {num_iterations} iterations: {std_baseline_rmse}")





# In[35]:


from catboost import CatBoostRegressor

# Identificazione delle colonne categoriche
categorical_features_indices = np.where(X_train.dtypes == 'object')[0]

# Definisci il modello CatBoost
catboost_regressor = CatBoostRegressor(random_state=42, silent=True)

# Parametri per la ricerca
param_grid_catboost = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bylevel': [0.6, 0.8, 1.0]
}

# GridSearchCV con 5-fold cross-validation
grid_search_catboost = GridSearchCV(estimator=catboost_regressor, param_grid=param_grid_catboost, cv=5, 
                                    scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_catboost.fit(X_train, y_train, cat_features=categorical_features_indices)

# Ottieni i migliori parametri
best_params_catboost = grid_search_catboost.best_params_
print(f"Migliori parametri trovati: {best_params_catboost}")



# In[36]:


# Addestramento del Random Forest Regressor con i migliori parametri
best_params = {'colsample_bylevel': 1.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 1.0}
# Addestra il modello con i migliori parametri
best_catboost_regressor = CatBoostRegressor(**best_params_catboost, random_state=42, silent=True)
best_catboost_regressor.fit(X_train, y_train, cat_features=categorical_features_indices)



# Definisci il numero di iterazioni di resampling
num_iterations = 10
mse_scores = []
rmse_scores = []
baseline_rmse_scores = []


for i in range(num_iterations):
    # Dividi il dataset di training in training e validation set
    X_train_resample, X_val_resample, y_train_resample, y_val_resample = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    # Addestra il modello sul training set di resampling
    best_catboost_regressor.fit(X_train_resample, y_train_resample)
    
    # Valuta il modello sul validation set di resampling
    predictions = best_catboost_regressor.predict(X_val_resample)
    mse = mean_squared_error(y_val_resample, predictions)
    rmse = np.sqrt(mse)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
 
    # Calcola la baseline RMSE in scala logaritmica
    baseline_pred_log = np.mean(y_train_resample)  # Media dei logaritmi dei valori target nel training set
    baseline_predictions_log = np.full_like(y_val_resample, baseline_pred_log)
    baseline_rmse_log = np.sqrt(mean_squared_error(y_val_resample, baseline_predictions_log))
    baseline_rmse_scores.append(baseline_rmse_log)


# Calcola la media e la deviazione standard delle MSE ottenute
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
print(f"Mean MSE over {num_iterations} iterations: {mean_mse}")
print(f"Standard Deviation of MSE over {num_iterations} iterations: {std_mse}")
print(f"Mean RMSE over {num_iterations} iterations: {mean_rmse}")
print(f"Standard Deviation of RMSE over {num_iterations} iterations: {std_rmse}")
print(f"Mean Baseline RMSE over {num_iterations} iterations: {mean_baseline_rmse}")
print(f"Standard Deviation of Baseline RMSE over {num_iterations} iterations: {std_baseline_rmse}")





# In[ ]:




