# import libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import evaluationer,models, null_value_handling

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
# st.set_page_config(layout="wide")

st.set_page_config(
    page_title="LazyML App",
    page_icon="🧊",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

import streamlit as st

# Title with Rainbow Transition Effect and Neon Glow
html_code = """
<div class="title-container">
  <h1 class="neon-text">
    LazyML
  </h1>
</div>

<style>
@keyframes rainbow-text-animation {
  0% { color: red; }
  16.67% { color: orange; }
  33.33% { color: yellow; }
  50% { color: green; }
  66.67% { color: blue; }
  83.33% { color: indigo; }
  100% { color: violet; }
}

.title-container {
  text-align: center;
  margin: 1em 0;
  padding-bottom: 10px;
  border-bottom: 4  px solid #fcdee9; /* Magenta underline */
}

.neon-text {
  font-family: Arial, sans-serif;
  font-size: 4em;
  margin: 0;
  animation: rainbow-text-animation 5s infinite linear;
  text-shadow: 0 0 5px rgba(255, 255, 255, 0.8),
               0 0 10px rgba(255, 255, 255, 0.7),
               0 0 20px rgba(255, 255, 255, 0.6),
               0 0 40px rgba(255, 0, 255, 0.6),
               0 0 80px rgba(255, 0, 255, 0.6),
               0 0 90px rgba(255, 0, 255, 0.6),
               0 0 100px rgba(255, 0, 255, 0.6),
               0 0 150px rgba(255, 0, 255, 0.6);
}
</style>
"""

st.markdown(html_code, unsafe_allow_html=True)


# file uploader 
csv_upload = st.file_uploader("Input CSV File for ML modelling", type=['csv'])
csv_upload2 = st.file_uploader("Input CSV File of Test Data Prediction",type = ["csv"])
test = pd.DataFrame()
if csv_upload is not None:
    # read the uploaded file into dataframe
    df = pd.read_csv(csv_upload)

    # saving the dataframe to a CSV file
    df.to_csv('csv_upload.csv', index=False)
    st.write("Train File uploaded successfully. ✅")

    if csv_upload2 is not None:
        test = pd.read_csv(csv_upload2)
        id_col = st.selectbox("select column for submission i.e, ID",test.columns)
        submission_id = test[id_col]
        # st.write("Train File upl",submission_id)

        


    if len(test) >0:
        # saving the test dataframe to a CSV file
        test.to_csv('csv_upload_test.csv', index=False)
        st.write("Test File uploaded successfully. ✅")
    
    if st.radio("Display Train Data",["Yes","No"],index = 1) == "Yes":
        st.dataframe(df.head())

    if len(test) >0:
        if st.radio("Display Test Data",["Yes","No"],index = 1) == "Yes":
            st.dataframe(test.head())

    
    if st.radio("Select Supervision Category",["Supervised","Un-Supervised"],index =0) == "Supervised":
        
        selected_column = st.selectbox('Select Target column', df.columns, index=(len(df.columns)-1))
        
        # Display the selected column
        st.write('You selected:', selected_column)
        
        y = df[selected_column]
        
        if y.dtype == "O":
            st.write("⚠️⚠️⚠️ Target Column is Object Type ⚠️⚠️⚠️")
            if st.radio("Proceed for Label Encoding ",["Yes","No"],index = 1) == "Yes": 
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y= pd.Series(le.fit_transform(y))
                st.write("Label Encoding Completed ✅")
                
        if st.radio("Display Target Column",["Yes","No"],index =1) == "Yes":
                st.dataframe(y.head())
        
        X = df.drop(columns = selected_column)

        if st.radio("Display X-Train Data",["Yes","No"],index =1) == "Yes":
            st.dataframe(X.head())

        # dropping not important columns
        if st.radio("Drop Un-Important Column(s)",["Yes","No"],index = 1) == "Yes":
            selected_drop_column = st.multiselect('Select columns to be dropped', X.columns)
            X = X.drop(columns = selected_drop_column)
            if len(test) >0:
                test = test.drop(columns = selected_drop_column)
            st.write("Un-Important column(s) Delected ✅")
            st.dataframe(X.head())

        num_cols = X.select_dtypes(exclude = "O").columns 
        cat_cols = X.select_dtypes(include = "O").columns
        st.write("Numerical Columns in Train Data: ", tuple(num_cols))
        st.write("Categorical Columns in Train Data: ", tuple(cat_cols))

        if st.radio("Select method for ML modelling", ["Manual","Auto Optimized"],index = 0) == "Manual":

            if X.isnull().sum().sum() >0 :
                st.write("⚠️⚠️⚠️ There are missing values in Train Data ⚠️⚠️⚠️")

                if st.selectbox("Drop null values or Impute",["Drop Null Values","Impute Null Values"],index = 1) == "Drop Null Values":

                    X = X.dropna()
                    if len(test) >0:
                        st.write("⚠️⚠️⚠️ If choosing drop values, test dataset will also drop those values please choose missing value imputation method befittingly.⚠️⚠️⚠️ ")
                        test = test.dropna()

                clean_num_nvh_df = pd.DataFrame()
                if X[num_cols].isnull().sum().sum() >0:
                    st.write("Numerical Columns with Percentage of Null Values: ")
                    num_cols_nvh = X[num_cols].isnull().sum()[X[num_cols].isnull().sum()>0].index
                    st.dataframe(round(X[num_cols].isnull().sum()[X[num_cols].isnull().sum()>0]/len(X)*100,2))
                    dict_1= {}
                    for nvh_method in null_value_handling.null_value_handling_method_num_cols :
                        
                        selected_nvh_num_cols = st.multiselect(f'method:- \"{nvh_method}\" for Numerical columns', num_cols_nvh,)
                        dict_1[nvh_method] = selected_nvh_num_cols

                        num_cols_nvh = set(num_cols_nvh) - set(selected_nvh_num_cols)
                        if len(num_cols_nvh) ==0:
                            break
                    num_nvh_df = pd.DataFrame(data=dict_1.values(), index=dict_1.keys())

                    clean_num_nvh_df = num_nvh_df.T[num_nvh_df.T.count()[num_nvh_df.T.count()>0].index]
                    
                    st.write("Methods for Numerical columns null value handling",clean_num_nvh_df )

                if len(test) >0:
                    if test[num_cols].isnull().sum().sum() >0:    
                        test_num_cols_nvh = test[num_cols].isnull().sum()[test[num_cols].isnull().sum()>0].index
                        st.write("sdgs",test_num_cols_nvh)
                        test[num_cols] = IterativeImputer(max_iter = 200,random_state= 42).fit_transform(test[num_cols])
                

                clean_num_nvh_df_cat = pd.DataFrame()
                if X[cat_cols].isnull().sum().sum() >0:
                    st.write("Categorical Columns with Percentage of Null Values: ")
                    cat_cols_nvh = X[cat_cols].isnull().sum()[X[cat_cols].isnull().sum()>0].index
                    st.dataframe(round(X[cat_cols].isnull().sum()[X[cat_cols].isnull().sum()>0]/len(X)*100,2))

                    dict_2= {}
                    for nvh_method in null_value_handling.null_value_handling_method_cat_cols :
                        st.write("dsff",nvh_method)
                        
                        selected_nvh_num_cols = st.multiselect(f'method:- \"{nvh_method}\" for Numerical columns', cat_cols_nvh,)
                        dict_2[nvh_method] = selected_nvh_num_cols

                        cat_cols_nvh = set(cat_cols_nvh) - set(selected_nvh_num_cols)
                        if len(cat_cols_nvh) ==0:
                            break
                    num_nvh_df_cat = pd.DataFrame(data=dict_2.values(), index=dict_2.keys())
                    clean_num_nvh_df_cat = num_nvh_df_cat.T
                    st.write("Methods for Categorical columns null value handling",[clean_num_nvh_df_cat])  

                if len(test) >0:
                    if test[cat_cols].isnull().sum().sum() >0:    
                        test_num_cols_nvh_cat = test[cat_cols].isnull().sum()[test[cat_cols].isnull().sum()>0].index
                        st.write("sdgs",test_num_cols_nvh_cat)
                        test[cat_cols] = SimpleImputer(strategy = "most_frequent").fit_transform(test[cat_cols])

                
                null_value_handling.null_handling(X,clean_num_nvh_df,clean_num_nvh_df_cat)
                st.write("X Data after Null value handling", X.head())

            new_df = pd.concat([X,y[X.index]],axis = 1)
            
            csv = new_df.to_csv(index = False)
            if st.radio("Download Null Value Handled DataFrame as CSV File ? ",["Yes","No"],index = 1) == "Yes": 
                st.download_button(label="Download Null Value Handled CSV File",data=csv,file_name='NVH_DataFrame.csv',mime='text/csv')

            ord_enc_cols = []

            if len(cat_cols) == 0:
                st.write("No Categorical Columns in Train")
            else:
                st.write("Select Columns for Ordinal Encoding")
                for column in cat_cols:

                    selected = st.checkbox(column)
                    if selected:
                        st.write(f"No. of Unique value in {column} column are", X[column].nunique())
                        ord_enc_cols.append(column)
            ohe_enc_cols = set(cat_cols) -set(ord_enc_cols)
            ohe_enc_cols = list(ohe_enc_cols)
            if len(ord_enc_cols)>0:
                st.write("ordinal encoded columns" ,tuple(ord_enc_cols))
            if len(ohe_enc_cols)>0:
                st.write("one hot encoded columns" ,tuple(ohe_enc_cols))

            if len(ord_enc_cols)>0:
                if st.radio("proceed for ordinal encoding",["Yes","No"],index = 1) == "Yes":
                    ordinal_order_vals = []

                    for column in ord_enc_cols:
                        unique_vals = X[column].unique()
                        # st.write(f"No. of Unique value in {column} column are", len(unique_vals))
                                        
                        ordered_unique_vals = st.multiselect("Select values in order for Ordinal Encoding",unique_vals,unique_vals)
                        ordinal_order_vals.append(ordered_unique_vals)
                    
                    st.write("order of values for Ordinal Encoding",tuple(ordinal_order_vals))   
                    # import ordinal encoder
                    from sklearn.preprocessing import OrdinalEncoder
                    ord = OrdinalEncoder(categories=ordinal_order_vals,handle_unknown= "use_encoded_value",unknown_value = -1 )
                    X[ord_enc_cols] = ord.fit_transform(X[ord_enc_cols])
                    if len(test) >0:
                        test[ord_enc_cols] = ord.transform(test[ord_enc_cols])
                    st.write("DataFrame after Ordinal Encoding",X.head())
                    st.write("Ordinal Encoding Completed ✅")

            if len(ohe_enc_cols)>0:
                if st.radio("proceed for OnehotEncoding ",["Yes","No"],index = 1) == "Yes":    # import one hot encoder
                    from sklearn.preprocessing import OneHotEncoder
                    ohe = OneHotEncoder(sparse_output = False,handle_unknown = "ignore")
                    pd.options.mode.chained_assignment = None
                    X.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(X[ohe_enc_cols])
                    X.drop(columns = ohe_enc_cols,inplace = True)
                    if len(test) >0:
                        test.loc[:, ohe.get_feature_names_out()] = ohe.transform(test[ohe_enc_cols])
                        test.drop(columns = ohe_enc_cols,inplace = True)

                    pd.options.mode.chained_assignment = 'warn'

                    st.write("DataFrame after One Hot Encoding",X.head())
                    st.write("OneHot Encoding Completed ✅")
            
            new_df = pd.concat([X,y],axis = 1)
            
            csv = new_df.to_csv(index = False)
            if st.radio("Download Encoded DataFrame as CSV File ? ",["Yes","No"],index = 1) == "Yes": 
                st.download_button(label="Download Ordinal Encoded CSV File",data=csv,file_name='Encoded_DataFrame.csv',mime='text/csv')

            
            random_state = st.number_input("Enter Random_state",max_value=100,min_value=1,value=42) 
            test_size = st.number_input("Enter test_size",max_value=0.99, min_value = 0.01,value =0.2)      
            if st.radio("select Train Validation Split Method",
                        [f"Train_Test_split, Default (Random_state = {random_state},Test_size = {test_size})",
                        "KFoldCV, Default (CV = 5)"], index = 0)== f"Train_Test_split, Default (Random_state = {random_state},Test_size = {test_size})":
                ttsmethod = "Train_Test_split"
            else:
                ttsmethod = "KFoldCV"
            st.write('You selected:', ttsmethod)
            if ttsmethod == "Train_Test_split":
                X_train,X_Val,y_train,y_val = tts(X,y[X.index],random_state = random_state,test_size = test_size)
                st.write('X-Training Data shape:', (X_train.info()))
                
                st.write('X-Training Data shape:', X_train.shape)
                st.write('X-Validation Data shape:', X_Val.shape)

            ml_cat = st.radio("Select Machine Learning Category",["Regression","Classification"],index =0)

            if ml_cat =="Regression":
                method_name_selector = st.selectbox("Select Error Evaluation Method",evaluationer.method_df.index,index = 0)

                method = evaluationer.method_df.loc[method_name_selector].values[0]
                reg_algorithm = []
                selected_options = []

                for option in models.Regression_models.index:
                    selected = st.checkbox(option)
                    if selected:
                        selected_options.append(option)

                        param = models.Regression_models.loc[option][0].get_params()
                        Temp_parameter = pd.DataFrame(data=param.values(), index=param.keys())                    
                        Temp_parameter_transposed = Temp_parameter.T
                        parameter = pd.DataFrame(data=param.values(), index=param.keys())
                        def is_boolean(val):
                            return isinstance(val, bool)

                        # Apply the function to the DataFrame column and create a new column with the resuSlts
                        bool_cols= parameter[parameter[0].apply(is_boolean)].index
                        param_transposed = parameter.T
                        # st.write("hrweurgesj",param_transposed.loc[:, bool_cols])
                        # st.write("bool_cols",bool_cols)
                        remaining_cols = set(param_transposed.columns) - set(bool_cols)
                        remaining_cols = tuple(remaining_cols)
                        # st.write("rem_Cols",remaining_cols)

                        for col in remaining_cols:    
                            param_transposed[col] = pd.to_numeric(param_transposed[col],errors="ignore")
                        cat_cols = param_transposed.select_dtypes(include = ["O"]).T.index.to_list()
                        num_cols = set(remaining_cols) - set(cat_cols)
                        cat_cols = set(cat_cols) - set(bool_cols)
                        num_cols = tuple(num_cols)
                        # st.write("sdsafdsd",num_cols)
                        for i in num_cols:
                            param_transposed[i] = st.number_input(f"input \"{i}\" value \n{option}",value = parameter.T[i].values[0])
                        for i in cat_cols:
                            param_transposed[i] = st.text_input(f"input \"{i}\" value \n{option}",value = parameter.T[i].values[0])
                        for i in bool_cols:
                            st.write("default value to insert",Temp_parameter_transposed[i].values[0])
                            param_transposed[i] = st.selectbox(f"input \"{i}\" value \n{option}",[False, True], index=Temp_parameter_transposed[i].values[0])
                            
                        inv_param = param_transposed.T
                        new_param = inv_param.dropna().loc[:,0].to_dict()
                        # st.write("asad",new_param)
                        models.Regression_models.loc[option][0].set_params(**new_param)
                        a =  models.Regression_models.loc[option][0].get_params()
                        reg_algorithm.append(models.Regression_models.loc[option][0])
                if st.button("Train Regression Model"):
                    for algorithm in reg_algorithm:
                        evaluationer.evaluation(f"{algorithm} baseline",X_train,X_Val,y_train,y_val,algorithm,method,"reg")
                    st.write("Regression Model Trained Successfully",evaluationer.reg_evaluation_df)
                if len(test)>0:
                    if st.radio("Predict",["Yes","No"],index = 1) =="Yes":

                        if len(evaluationer.reg_evaluation_df) >0:
                            a = st.number_input("select index of best algorithm for test prediction",min_value = 0,max_value =len(evaluationer.reg_evaluation_df) -1, value = len(evaluationer.reg_evaluation_df) -1)
                            
                            test_prediction = evaluationer.reg_evaluation_df.loc[a,"model"].predict(test)
                            
                            submission_file = pd.DataFrame(index = [submission_id],data = test_prediction,columns = [selected_column])
                            st.write("Sample of Prediction File",submission_file.head())
                            csv_prediction = submission_file.to_csv()
                            if st.radio("Download Prediction File as CSV File ? ",["Yes","No"],index = 1) == "Yes": 
                                st.download_button(label="Download Prediction CSV File",data=csv_prediction,file_name='prediction.csv',mime='text/csv')

            

                
            if ml_cat =="Classification":
                
                
                
                cla_algorithm = []
                selected_options = []

                for option in models.Classification_models.index:
                    selected = st.checkbox(option)
                    if selected:
                        selected_options.append(option)
                        
                        param = models.Classification_models.loc[option][0].get_params()
                                
                        
                        parameter = pd.DataFrame(data=param.values(), index=param.keys())
                        Temp_parameter = parameter.copy()
                        Temp_parameter_transposed = (Temp_parameter.T).copy()    
                        def is_boolean(val):
                            return isinstance(val, bool)

                        # Apply the function to the DataFrame column and create a new column with the resuSlts
                        bool_cols= parameter[parameter[0].apply(is_boolean)].index
                        param_transposed = parameter.T
                        st.write("bool_cols",bool_cols)
                        remaining_cols = set(param_transposed.columns) - set(bool_cols)
                        remaining_cols = tuple(remaining_cols)
                        st.write("rem_Cols",remaining_cols)

                        for col in remaining_cols:    
                            param_transposed[col] = pd.to_numeric(param_transposed[col],errors="ignore")
                        cat_cols = param_transposed.select_dtypes(include = ["O"]).T.index.to_list()
                        num_cols = set(remaining_cols) - set(cat_cols)
                        num_cols = tuple(num_cols)
                        st.write("sdsafdsd",num_cols)
                        for i in num_cols:
                            param_transposed[i] = st.number_input(f"input \"{i}\" value \n{option}",value = parameter.T[i].values[0])
                        for i in cat_cols:
                            param_transposed[i] = st.text_input(f"input \"{i}\" value \n{option}",value = parameter.T[i].values[0])
                        for i in bool_cols:
                            st.write("default value to insert",Temp_parameter_transposed[i].values[0])
                            param_transposed[i] = st.selectbox(f"input \"{i}\" value \n{option}",[False,True], index=Temp_parameter_transposed[i].values[0])
                        inv_param = param_transposed.T
                        new_param = inv_param.dropna().loc[:,0].to_dict()
                        st.write("asad",new_param)
                        models.Classification_models.loc[option][0].set_params(**new_param)
                        a =  models.Classification_models.loc[option][0].get_params()
                        cla_algorithm.append(models.Classification_models.loc[option][0])
                # st.write("sada",reg_algorithm/)
                if st.button("Train Regression Model"):
                    method = None
                    for algorithm in cla_algorithm:
                        evaluationer.evaluation(f"{algorithm} baseline",X_train,X_Val,y_train,y_val,algorithm,method,eva ="class")
                    st.write("Regression Model Trained Successfully",evaluationer.classification_evaluation_df)

                if len(test)>0:
                    if st.radio("Predict",["Yes","No"],index = 1) =="Yes":
                        if len(evaluationer.classification_evaluation_df) >0:
                            a = st.number_input("select index of best algorithm for test prediction",min_value = 0,max_value =len(evaluationer.classification_evaluation_df) -1, value = len(evaluationer.classification_evaluation_df) -1)
                            
                            test_prediction = evaluationer.classification_evaluation_df.loc[a,"model"].predict(test)
                            
                            submission_file = pd.DataFrame(index = [submission_id],data = test_prediction,columns = [selected_column])
                            st.write("Sample of Prediction File",submission_file.head())
                            csv_prediction = submission_file.to_csv()
                            if st.radio("Download Prediction File as CSV File ? ",["Yes","No"],index = 1) == "Yes": 
                                st.download_button(label="Download Prediction CSV File",data=csv_prediction,file_name='prediction.csv',mime='text/csv')


        else :
            pass
