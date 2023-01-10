import streamlit as st
import pandas as pd
import lime
from lime import lime_tabular
import requests
import json
from types import SimpleNamespace



def main(): 
    # local API (à remplacer par l'adresse de l'application déployée)
    # api_url = 
#---------------------------------------------------------#
########## Fucntions for requesting API ##########

# To get the clients indexes

 @st.cache
    def get_clients_indexes_list():
        # URL of the clients indexes API
        clients_ids_api_url = api_url + "/get_client_indexes/"
        # Requesting the API and saving the response
        response = requests.get(clients_ids_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # Getting the clients indexes from the content
        clients_ids = pd.Series(content['data'])
        return clients_ids
    
    # Get data of selected client 
    @st.cache
    def get_clients_data(selected_clients_index): # parameter= the selected index of the client on sidebar
        # URL of the scoring API (ex: SK_ID_CURR = 99999)
        selected_clients_data_api_url = api_url + "data_client/?SK_ID_CURR=" + str(selected_clients_index)
        # save the response to API request
        response = requests.get(selected_clients_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        data_client = pd.Series(content['data']).rename(selected_clients_index)
        
        # Get all training data
    @st.cache
    def get_all_training_data():
        # URL of the scoring API
        all_training_data_api_url = api_url + "all_training_data/"
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
        clients_score_api_url = api_url + "clients_score/?SK_ID_CURR=" + str(selected_clients_index)
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
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # Get values from the content and creat dataframe
        #lime_explanations = pd.DataFrame(content['lime_explanations'])
        return lime_explanations
        
        
        # Get the list of feature importances for lightGBM model
    @st.cache
    def get_global_features_importances():
        # url of the global feature importance api
        feature_importance_api_url = api_url + "feature_importance"
        # Requesting the API and save the response
        response = requests.get(feature_importance_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feature_importances = pd.Series(content['feature_importances']).sort_values(ascending=False)
        return feature_importances

    ########## Setting up Streamlit application  ##########
    
    # Configuration of the streamlit page
    st.set_page_config(page_title='Loan attribution prediction',
                       page_icon='random',
                       layout='centered',
                       initial_sidebar_state='auto')
    # Set title
    st.title('Credit loan attribution prediction')
    st.header("Adèle Souleymanova - Data Science project 7")
    


    # show an image on the sidebar  
    img = Image.open("credit_bank.JPG")
    st.sidebar.image(img, width=250)
    
    # ------------------------------------------------
    # Select the customer's index (Id)
    # ------------------------------------------------
    
    clients_indexes = get_clients_indexes_list()
    selected_clients_index = st.sidebar.selectbox('''Choose client's ID:''', clients_indexes, key=18)
    st.write('You selected: ', selected_clients_index)
    
    # Get locale features importance values for the client
    local_importance_features = get_local_feature_importance(selected_clients_index)
    
    # ------------------------------------------------
    # Get client's data 
    # ------------------------------------------------
    data_client = get_clients_data(selected_clients_index)
    
    X_training, y_training = get_all_training_data()
    y_training = y_training.replace({0: 'loan repaid',
                                 1: 'loan not repaid '})
    
    local_features_importance_values = get_local_feature_importance(selected_clients_index)
    #----------------------------------------------------
    ##### Slider with feature names####
    #----------------------------------------------------
    
    def get_list_display_features(local_features_importance_values, def_n, key):
    
        all_features = X_training.columns.to_list()
        
        n_features = st.slider("Number of features",
                      min_value=2, max_value=50,
                      value=def_n, step=None, format=None, key=key)
        
        if st.checkbox('Features explaning the scorong and decision for selected client', key=key):
            columns_to_display = list(local_importance_features.abs()
                                .sort_values(ascending=False)
                                .iloc[:n_features].index)
        else:
            columns_to_display = list(get_global_features_importances().sort_values(ascending=False)\
                                            .iloc[:n].index)
            
        box_with_columnNames = st.multiselect('Choose the features to display (default: global feature importance for lightGBM model):',
                                        sorted(all_features),
                                        default=columns_to_display, key=key)
        return box_with_columnNames
    
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
        bool_cust = (score >= thresh)

        if bool_cust is False:
            decision = "LOAN APPROVED" 
            # st.balloons()
            # st.warning("The loan has been accepted but...")
        else:
            decision = "LOAN REJECTED"
        
        st.write('Decision:', decision)
        
        expander = st.beta_expander("Light Gradient Boosting Model")

        expander.write("How this works:")

        expander.write(""""When a client is selected, a probability is predicted by the LGBM model. 
    It allows us to decide whether or not a client is more likely to repay the loan. If a client's probability is higher than the threshold value, he falls into the defaulter category. However, if the predicted probability is lower than the threshold, the client falls into the non defaulter category. 
    The threshold is calculated to minimize the bank costs. It takes in  consideration the number of false negatives (predicted as solvent but in reality defaulter clients) and false positives ( predicted as defaulter but in reality solvent clients.""")
        
        
    #-------------------------------------------------
    ########## Clients data ##########
    #-------------------------------------------------
    
    if st.sidebar.checkbox("Clients data ", key=38):

        st.header("Clients data")
        
        # # If checkbox selected show clients dataframe
        # if st.checkbox('Display clients data', key=37): 
        # Diplay dataframe of the client 
        clients_df = get_clients_data(selected_clients_index)
        st.dataframe(clients_df)


        
    # Display local interpretation figure for LIME explanantions for the selected client
    if st.checkbox('LIME - Local Interpretable Model-Agnostic Explanations', key=37): 
        with st.spinner('Plot loading...'):
            nb_features = st.slider("Nb of features to display",
                                        min_value=2, max_value=42,
                                        value=10, step=None, format=None, key=14)
                
            # Plotting the results
            # get datafrme of features weights for the client 
            client_feature_importances_df = get_local_feature_importance(selected_clients_index)
                
            # Convert from dataframe to list of tuples (for it to be usable for plotting)
            client_feature_importances_tuples = [tuple(r) for r in client_feature_importances_df.to_numpy().tolist()]

            plt=client_feature_importances_tuples.as_pyplot_figure()
            plt.tight_layout()
               
                # Plot the graph on the dashboard
                st.pyplot(plt.tight_layout())

                st.markdown('LIME explanations for the client_')

                expander = st.beta_expander("What this plot means...")

                expander.write("This technique measures the impact of features in prediction making. It'ss role's to show which of the features contribute more to the decision ( solvent or not) for the given client")
                
                #Show global features importance for the model

                
    #-------------------------------------------------
    ########## Features importance ##########
    #-------------------------------------------------
    
    if st.sidebar.checkbox("Importance of the features", key=29):
        st.header("Global feature importances")
        
        global_features_importances= get_global_features_importances()

        st.bar_chart(global_features_importances)
        

if __name__ == '__main__':
    main()
        
