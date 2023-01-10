
import joblib
from joblib import dump, load
import dill
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from sklearn.inspection import permutation_importance
from pydantic import BaseModel # classe qui constomise les entrées
import matplotlib.pyplot as plt
import sys
from lime import lime_tabular
sys.setrecursionlimit(1000000)

app = Flask(__name__) # initialise l'aplication flask

# Load the best model
model_lightgbm = joblib.load('./model_lightgbm')
thresh = joblib.load('./best_threshold')

# pour récuperer les clients donnés d'entrée d'API
class Client(BaseModel):
    index: int
    
    
# Load the data 

X_test=pd.read_csv('./X_testing.csv')
y_test=pd.read_csv('./y_test.csv')
X_train=pd.read_csv('./X_res.csv')
y_train=pd.read_csv('./y_res.csv')

#app_x_test=pd.read_csv('./data/app_x_test.csv')

x_test=X_test.iloc[0:100]
y_y_test=y_test[0:100]['TARGET']
x_train=X_train.iloc[0:300]
y_y_train=y_train[0:300]['TARGET']

# Send clients indexes if requested
@app.get('/get_client_indexes/')
def get_clients_indexes():
    # get the indexes of all clients in x_test dataset 
    clients_ids = pd.Series(list(x_test.index.sort_values()))
    # Convert pd.Series to JSON
    clients_ids_json = json.loads(clients_ids.to_json())
    # Return the data 
    return jsonify({"data": clients_ids_json})


# Return data of selected client 
@app.route('/data_client/')
def data_client():
    # Parse the http request to get arguments (sk_id_client)
    ## 'SK_ID_CURR' is the column name with the ID of clients
    sk_id_client = int(request.args.get('SK_ID_CURR'))
    # Get the personal data for the customer (pd.Series)
    X_client_series = x_test.loc[sk_id_client, :]
    # Convert the data to JSON
    X_client_json = json.loads(X_client_series.to_json())
    # Return the data
    return jsonify({'status': 'ok',
                    'data': X_client_json})

# Return all data of training set
#local url:
#Heroku url : 
@app.route('/all_training_data/')
def all_training_data():
    # get all data from X_train and y_train data
    # and convert the data to JSON
    X_train_json = json.loads(X_train.to_json())
    y_train_json = json.loads(y_train.to_json())
    # Return the data
    return jsonify({'status': 'ok',
                    'X_train': X_train_json,
                    'y_train': y_train_json})

# Return predictions (score and decision) for selected client 
# local url :
#Heroku url: 
@app.route('/clients_score/')
def clients_score():
    # Parse http request to get arguments (sk_id_client)
    sk_id_client = int(request.args.get('SK_ID_CURR'))
    # Get the data for the customer (pd.DataFrame)
    X_client = x_test.loc[sk_id_client:sk_id_client]
    # Compute the score for slected client
    score_client = model_lightgbm.predict_proba(X_client)[:,1][0]
    # Return score
    return jsonify({'status': 'ok',
                    'SK_ID_CURR': sk_id_client,
                    'score': score_client,
                    'thresh': thresh})

# Return the feature importance data for selected client
@app.post('/local_features_importance')
def local_features_importance(sample_client:Client): # Voir c'est quoi sample client
    explainer = lime_tabular.LimeTabularExplainer(training_data =X_train.values,
                                                feature_names=X_train.columns.values,
                                                class_names =['Non_defaulter_0','Defaulter_1'],
                                                mode="classification",
                                                verbose=False,
                                                random_state=10)
   
    client_index = sample_client.dict()
    num=int(client_index['index'])
    test_sample=x_test.iloc[num,:]
    lim_exp =explainer.explain_instance(data_row=test_sample, predict_fn=model_lightgbm.predict_proba, num_features=117) # See if i can change number of features to display
    
    weights = []
    feature_column_names= X_test.columns
    #Create DataFrame

    #Iterate over first 100 rows in feature matrix
    for x in X_test.values[0:100]:
    
        #Get explanation
        exp = lim_exp
        exp_list = exp.as_map()[1]
        exp_list = sorted(exp_list, key=lambda x: x[0])
        exp_weight = [x[1] for x in exp_list]
        
        #Get weights
        weights.append(exp_weight)
        
        #Create DataFrame
        lime_weights_df = pd.DataFrame(data=weights,columns=X_test.columns)
        
        # Convert the dataframe of client's feature importance weights to JSON
        lime_weights_df_json = json.loads(lime_weights_df.to_json())
        
        # Return the data of weights
        return jsonify({'status': 'ok',
                        'weights_df': lime_weights_df_json})

# Return json object of feature importance for LightGBM model
@app.route('/feature_importance/')
def feature_importance():
    
    feature_names = X_train.columns
    perm_importance = permutation_importance(model_lightgbm, X_test, y_test)
    mean_perm_importances = np.array(sorted(perm_importance.importances_mean, reverse=True)).tolist() 
    sorted_idx =np.array(mean_perm_importances).argsort().tolist()
    features_sorted_= feature_names[sorted_idx].tolist()
 
    # Convert pd.Series to JSON
    feature_importance_json = json.loads(sorted_idx.to_json())
    features_sorted_json = json.loads(features_sorted_.to_json())
    
    # Return the data as json object
    return jsonify({'status': 'ok',
                    'feature_importances': feature_importance_json,
                  'features_sorted_': features_sorted_json })
