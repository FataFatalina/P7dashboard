import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from PIL import Image
import matplotlib.pyplot as plt
import ast

########## Fucntions for requesting API ##########

local_api_url = "http://127.0.0.1:5000/"
api_url = "https://p7flaskapp.herokuapp.com/"

@st.cache
def get_clients_indexes_list():
    # URL of the clients indexes API
    clients_ids_api_url = api_url + "/get_client_indexes/"
    # Requesting the API and saving the response
    response = requests.get(clients_ids_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # Getting the clients indexes from the content
    clients_ids = pd.Series(content['data']).values
    return clients_ids


# Get data of selected client 
@st.cache
def get_clients_data(selected_clients_index): # parameter= the selected index of the client on sidebar
    # url of the clients data
    selected_clients_data_api_url = api_url + "data_client/?id_client=" + str(selected_clients_index)
    # save the response to API request
    response = requests.get(selected_clients_data_api_url)
    # convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # convert data to pd.Series
    data_client = pd.Series(content['data']).rename(selected_clients_index)
    return data_client

@st.cache
def get_all_training_data():
    # URL of the scoring API
    all_training_data_api_url = api_url + "/all_training_data/"
    # save the response of API request
    response = requests.get(all_training_data_api_url)
    # convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # convert data to pd.Series
    X_train_df = pd.DataFrame(content['X_train'])
    y_train_series = pd.Series(content['y_train']['TARGET']).rename('TARGET')
    return X_train_df, y_train_series
    
# Get score prediction for selected client
@st.cache
def get_clients_score(selected_clients_index):
    # URL of the scoring API
    clients_score_api_url = api_url + "/clients_score/?id_client=" + str(selected_clients_index)
    # API request and save results
    response = requests.get(clients_score_api_url)
    # convert from JSON to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # Get values from the content
    score = content['score']
    thresh = content['thresh']
    return score, thresh

# Get the local feature importance of the selected client for LightGBM model
@st.cache
def get_local_feature_importance(selected_clients_index):
    # url of the local feature importance api
    local_features_importance_api_url = api_url + "local_features_importance/?id_client=" + str(selected_clients_index)
    # save the response of API request
    response = requests.get(local_features_importance_api_url)
    # convert from JSON format to Python list
    content = json.loads(response.content.decode('utf-8'))
    # # Get values from the content 
    lime_explanations = content['lime_explanations']
    #return lime_features, lime_explanations
    return lime_explanations
@st.cache
def get_global_features_importances():
    # url of the global feature importance api
    feature_importance_api_url = api_url + "feature_importance"
    # Requesting the API and save the response
    response = requests.get(feature_importance_api_url)
    # convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # convert back to pd.Series
    feature_importances = pd.Series(content['mean_perm_importances'])
    features_sorted_ =  pd.Series(content['features_sorted_'])
    
    global_feat_imp_df = pd.DataFrame(list(zip(feature_importances,features_sorted_)), columns=['Importances', 'Features'])
    return global_feat_imp_df, features_sorted_, feature_importances


    ########## Setting up Streamlit application  ##########
    
# Configuration of the streamlit page
st.set_page_config(page_title='Loan attribution prediction',
                   page_icon='random',
                    layout='centered',
                    initial_sidebar_state='auto')
# Set title
st.title('Credit loan attribution prediction')
st.header("Adèle Souleymanova - Data Science project 7")

st.markdown("""<style> body {color: #fff; background-color: #273346; textColor = '#FFFFFF'; secondaryBackgroundColor = '#B9F1C0'}
</style> """, unsafe_allow_html=True) 


# show an image on the sidebar  
img = Image.open("data/credit_bank.JPG")
st.sidebar.image(img, width=250)

# ------------------------------------------------
# Select the customer's index (Id)
# ------------------------------------------------
    
clients_indexes = get_clients_indexes_list()
selected_clients_index = st.sidebar.selectbox('''Choose client's ID:''', clients_indexes, key=18)
st.write('You selected: ', selected_clients_index)
    

# ------------------------------------------------
# Get client's data 
# ------------------------------------------------
data_client = get_clients_data(selected_clients_index)
    
X_training, y_training = get_all_training_data()
y_training = y_training.replace({0: 'loan repaid', 1: 'loan not repaid '})
    
local_features_importance_values = get_local_feature_importance(selected_clients_index)


#-------------------------------------------------
    ########## Predictions and scoring ##########
#-------------------------------------------------
    
if st.sidebar.checkbox("Score and prediction ", key=38):
    st.header("Score and prediction of LightGBM model")

    #  Get score
    score, thresh = get_clients_score(selected_clients_index)

    # Display score (default probability)
    st.write('Classification probability: {:.0f}%'.format(score*100))
    # Display default threshold
    st.write('''Tuned model's threshold: {:.0f}%'''.format(thresh*100))
        
    # Compute decision according to the best threshold (True: loan refused)
    bool_client = (score >= thresh)

    if bool_client is True:
        decision = "LOAN REJECTED" 
        
    else:
        decision = "LOAN APPROVED"
        
    st.write('Decision:', decision)
        
    expander = st.expander("How this works:")

    expander.write("Predictions made with Light Gradient Boosting Model ")

    expander.write(""""When a client is selected, a probability is predicted by the LGBM model. 
    It allows us to decide whether or not a client is more likely to repay the loan. If a client's probability is higher than the threshold value, he falls into the defaulter category. However, if the predicted probability is lower than the threshold, the client falls into the non defaulter category. 
    The threshold is calculated to minimize the bank costs. It takes in  consideration the number of false negatives (predicted as solvent but in reality defaulter clients) and false positives ( predicted as defaulter but in reality solvent clients.""")
        
#-------------------------------------------------------      
########## Client's data ###########
#-------------------------------------------------------
        
if st.sidebar.checkbox("Clients data ", key=35):
    st.header("Clients data")
        
# # If checkbox selected show clients dataframe
# if st.checkbox('Display clients data', key=37): 
# Diplay dataframe of the client 
    clients_df = get_clients_data(selected_clients_index)
    st.dataframe(clients_df)
    

# Display local interpretation figure for LIME explanantions for the selected client
    if st.checkbox('LIME - Local Interpretable Model-Agnostic Explanations', key=33): 
        with st.spinner('Plot loading...'):
            nb_features = st.slider("Nb of features to display",
                                    min_value=2, max_value=42,
                                    value=10, step=None, format=None, key=14)
                
    
        lime_exps_str = get_local_feature_importance(selected_clients_index)
    
        # Convert a string representation of list to list
        lime_exps = ast.literal_eval(lime_exps_str)
        
        fig = plt.figure(figsize=(4,4))
        vals = [x[1] for x in lime_exps[0:int(nb_features)]]
        names = [x[0] for x in lime_exps[0:int(nb_features)]]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(lime_exps[0:int(nb_features)])) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        title = 'Local explanation'
        plt.title(title)
        st.pyplot(plt)

        st.markdown('Locale explanation plot for selected client')
    
        expander = st.expander("What this plot means...")

        expander.write("""This technique measures the impact of features in prediction making.
                   It's role's to show which of the features contribute more to the decision (solvent or not) for the given client.
                   
                   In RED : correlations to label 0 = non defaulter
    In GREEN : correlations to the label 1 = defaulter""")

               
#----------------------------------------------------
    ########## Slider with feature names#########
#----------------------------------------------------

# Load features importances data
global_feat_imp_df, features_sorted_, feature_importances  = get_global_features_importances()
X_train, y_train = get_all_training_data()
features = X_train.columns.tolist()
    
    
if st.sidebar.checkbox("Feature importance ", key=27):
    st.header("Global importance of the features for the LightGBM model")
    
    
    n = st.slider("Nb of features to display",
                      min_value=4, max_value=50,
                      value=5, step=None, format=None, key= 23)
        
    if st.checkbox('Features importance for Lightgbm model', key=43):
        disp_cols = list(features_sorted_.iloc[:n])
            
        disp_box_cols = st.multiselect('Choose the features to display (default: global importance for lgbm calssifier):',
                                       sorted(features),
                                       default=disp_cols, key=53)
        fig = plt.figure()
        plt.figure(figsize = [10, 25])
        plt.barh(features_sorted_[0:n], feature_importances[0:n])
        plt.xlabel("Permutation Importance", labelpad= 10)
        plt.ylabel("feature names",labelpad= 10)
        st.pyplot(plt)
    
        
    # # If checkbox selected show dataframe with features importances
    if st.checkbox('Display data of feature importance', key=37):
        # Diplay dataframe  
        st.dataframe(global_feat_imp_df)



    