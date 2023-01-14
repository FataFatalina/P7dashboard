import os
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import json
import joblib
from joblib import dump, load
import sklearn
import lightgbm as lgb
from lightgbm import LGBMClassifier
import lime
from lime import lime_tabular
from sklearn.inspection import permutation_importance

app = Flask(__name__) # initialise l'aplication flask

# Load the best model and threshold of classification probability
path = os.path.join('model_lightgbm')
with open(path, 'rb') as f:
    model_lightgbm = joblib.load(f)
    
path = os.path.join('data','best_threshold')
with open(path, 'rb') as th:
    thresh = joblib.load(th)
 
## Load the data 
path = os.path.join('data', 'x_tessting.csv')
X_test=pd.read_csv(path)

path = os.path.join('data', 'y_testing.csv')
y_test=pd.read_csv(path)

path = os.path.join('data', 'x_training.csv')
X_train=pd.read_csv(path)

path = os.path.join('data', 'y_training.csv')
y_train=pd.read_csv(path)



x_test=X_test.iloc[0:100]
y_y_test=y_test[0:100]['TARGET']
x_train=X_train.iloc[0:300]
y_y_train=y_train[0:300]['TARGET']



# Send clients indexes if requested
#local url:http://127.0.0.1:5000//get_client_indexes/
@app.get('/get_client_indexes/')
def get_clients_indexes():
    # get the indexes of all clients in x_test dataset 
    clients_ids = pd.Series(list(x_test.index.sort_values()))
    # Convert pd.Series to JSON
    clients_ids_json = json.loads(clients_ids.to_json(indent=None))
    # Return the data 
    return jsonify({"data": clients_ids_json})

# Return data of selected client 
#local url:http://127.0.0.1:5000//data_client/?id_client=0 ##-> for client index = 0
@app.route('/data_client/')
def data_client():
    # Parse the http request to get arguments (sk_id_client)
    sk_id_client = int(request.args.get('id_client')) 
    # Get the personal data for the customer (pd.Series)
    X_client_series = x_test.loc[sk_id_client, :]
    # Convert the data to JSON
    X_client_json = json.loads(X_client_series.to_json())
    # Return the data
    return jsonify({'status': 'ok',
                    'data': X_client_json})

# Return all data of training set
#local url:http://127.0.0.1:5000//all_training_data/
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
# local url :http://127.0.0.1:5000///clients_score/?id_client=0
#Heroku url: 
@app.route('/clients_score/')
def clients_score():
    # Parse http request to get arguments (sk_id_client)
    sk_id_client = int(request.args.get('id_client'))
    # Get the data for the customer (pd.DataFrame)
    X_client = x_test.loc[sk_id_client:sk_id_client]
    # Compute the score for slected client
    score_client = model_lightgbm.predict_proba(X_client)[:,1][0]
    # Return score
    return jsonify({'status': 'ok',
                    'id_client': sk_id_client,
                    'score': score_client,
                    'thresh': thresh})

# Return the feature importance data for selected client
# local_url : http://127.0.0.1:5000/local_features_importance/?id_client=1
@app.route('/local_features_importance/')
def local_features_importance(): 
    explainer = lime_tabular.LimeTabularExplainer(training_data = X_train.values,
                                                feature_names = X_train.columns.values,
                                                class_names = ['Non_defaulter_0','Defaulter_1'],
                                                mode = "classification",
                                                verbose = False,
                                                random_state = 10)
   
    id_client = int(request.args.get('id_client')) 
    # Get the personal data for the customer (pd.Series)
    X_client_series = x_test.loc[id_client, :]
    test_sample_array = X_client_series.to_numpy()
    lim_exp = explainer.explain_instance(data_row = test_sample_array, predict_fn = model_lightgbm.predict_proba, num_features = 114) 
   
    exp = lim_exp
    exp_list = exp.as_list()
    exp_list_json = json.dumps(exp_list)
    return jsonify({'status': 'ok','lime_explanations': exp_list_json})
    
    # This returns a dictionary of list of tuples {1: [(0, 0.2595477001301397),(12, -0.10418098817904746),...(52, -0.0487838001049165)]}
    # for the id_client =1
    exp = lim_exp
    exp_dict = exp.as_map() 
    
    #  This retuns the list of tuples 
    def get_value():
        for key,value in exp_dict.items():
            return value
        
    exp_list = get_value()
    # Convert the list of tuples to dictionary
    exp_list_dict = dict(exp_list)
    
    # Retrieve the keys and values of the dictionary into lists
    keys = []
    values = []
    for key, value in exp_list_dict.items():
        key, value  =int(key), float(value)
        keys.append(key)
        values.append(value)
    
    # Define explanation values and features names for ech value
    exp_weight_values = values
    exp_weights_features= X_test.columns[keys].tolist()
    
    # Convert to JSON
    exp_weight_values_json = json.dumps(exp_weight_values)
    exp_weights_features_json = json.dumps(exp_weights_features)
    
    # Return the data 
    return jsonify({'status': 'ok','lime_explanations': exp_weight_values_json, 'lime_explanations_features': exp_weights_features_json})
    
# Return json object of feature importance for LightGBM model
# Local url : 
@app.get('/feature_importance')
def feature_importance():
    feature_names = X_train.columns
    perm_importance = permutation_importance(model_lightgbm, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    mean_perm_importances = (np.array(perm_importance.importances_mean)[sorted_idx]).tolist()
    #sorted_idx =np.array(mean_perm_importances).argsort().tolist()
    features_sorted_= feature_names[sorted_idx].tolist()
 
    
    return {"mean_perm_importances": mean_perm_importances, "features_sorted_": features_sorted_}

if __name__ == "__main__":
    app.run(debug=True)