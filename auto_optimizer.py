import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import KNNImputer,SimpleImputer,IterativeImputer
import best_tts, evaluationer,models
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split as tts
from collections import Counter
#root_mean_squared_error
from sklearn.metrics import root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import outliers,best_tts
import feature_selections
def Auto_optimizer(X,y,eva,model,test= None):
    pass
    num_cols = X.select_dtypes(exclude = "O").columns
    cat_cols = X.select_dtypes(include = "O").columns
    st.write("Num_cols",tuple(num_cols))
    st.write("cat_cols",tuple(cat_cols))

# check for Duplicate and drop duplicated in X
    
    if len(X.isnull().sum()[(X.isnull().sum()/len(X)*100) >40]) >0:
        X = X.drop(columns = X.isnull().sum()[(X.isnull().sum()/len(X)*100) >40].index)
        st.write("Columns with more than 40% null values removed")
    # st.write("csx",X)

    len_null = X.isnull().sum().sum() 

    st.write(f"There are {len_null} null values in Train")

    knn_imputed_num_X = X.copy()
    si_mean_imputed_num_X = X.copy()
    # st.write("sf",si_mean_imputed_num_X)
    si_median_imputed_num_X = X.copy()
    si_most_frequent_imputed_num_X = X.copy()
    iter_imputed_num_X = X.copy()
    knn_imputed_X_cat_dropped = knn_imputed_num_X.copy()
    si_mean_imputed_X_cat_dropped = si_mean_imputed_num_X.copy()
    si_median_imputed_X_cat_dropped = si_median_imputed_num_X.copy()
    si_most_frequent_imputed_X_cat_dropped = si_most_frequent_imputed_num_X.copy()
    iter_imputed_X_cat_dropped = iter_imputed_num_X.copy()
    if len_null >0:
       
        if X[num_cols].isnull().sum().sum() >0:

            knn_imputer = KNNImputer(n_neighbors = 5) 
            knn_imputed_num_X[num_cols] = knn_imputer.fit_transform(knn_imputed_num_X[num_cols])
            si_imputer = SimpleImputer(strategy = "mean")
            si_mean_imputed_num_X[num_cols] = si_imputer.fit_transform(si_mean_imputed_num_X[num_cols])
            si_imputer = SimpleImputer(strategy = "median")
            si_median_imputed_num_X[num_cols] = si_imputer.fit_transform(si_median_imputed_num_X[num_cols])
            si_imputer = SimpleImputer(strategy = "most_frequent")
            si_most_frequent_imputed_num_X[num_cols] = si_imputer.fit_transform(si_most_frequent_imputed_num_X[num_cols])
            iter_imputer = IterativeImputer(max_iter = 200,random_state= 42)
            iter_imputed_num_X[num_cols] = iter_imputer.fit_transform(iter_imputed_num_X[num_cols])
        knn_imputed_X_cat_dropped = knn_imputed_num_X.copy()
        si_mean_imputed_X_cat_dropped = si_mean_imputed_num_X.copy()
        si_median_imputed_X_cat_dropped = si_median_imputed_num_X.copy()
        si_most_frequent_imputed_X_cat_dropped = si_most_frequent_imputed_num_X.copy()
        iter_imputed_X_cat_dropped = iter_imputed_num_X.copy()

        if X[cat_cols].isnull().sum().sum() >0:
            # treating missing values in categorical columns
            # st.write("si_mean_imputed_num_X",si_mean_imputed_num_X)
            si_imputer = SimpleImputer(strategy = "most_frequent")
            
            knn_imputed_num_X[cat_cols] = si_imputer.fit_transform(knn_imputed_num_X[cat_cols])
            si_imputer = SimpleImputer(strategy = "most_frequent")
            si_mean_imputed_num_X.loc[:,cat_cols] = si_imputer.fit_transform(si_mean_imputed_num_X.loc[:,cat_cols])
            # st.write("si_mean_imputed_num_X",si_mean_imputed_num_X)
            si_median_imputed_num_X[cat_cols] = si_imputer.fit_transform(si_median_imputed_num_X[cat_cols])
            si_most_frequent_imputed_num_X[cat_cols] = si_imputer.fit_transform(si_most_frequent_imputed_num_X[cat_cols])
            iter_imputed_num_X[cat_cols] = si_imputer.fit_transform(iter_imputed_num_X[cat_cols])

            knn_imputed_X_cat_dropped = knn_imputed_X_cat_dropped.dropna()
            si_mean_imputed_X_cat_dropped =si_mean_imputed_X_cat_dropped.dropna()
            si_median_imputed_X_cat_dropped =si_median_imputed_X_cat_dropped.dropna()
            si_most_frequent_imputed_X_cat_dropped =si_most_frequent_imputed_X_cat_dropped.dropna()
            iter_imputed_X_cat_dropped =iter_imputed_X_cat_dropped.dropna()
            st.write("sdds",knn_imputed_num_X)
            st.write("sddssd",knn_imputed_X_cat_dropped)
                                                          
    miss_val_dropped_X = X.dropna()
        
        # list of dataframes
        
    list_X_after_missing_values= [knn_imputed_num_X,
                            si_mean_imputed_num_X,
                            si_median_imputed_num_X,
                            si_most_frequent_imputed_num_X,
                            iter_imputed_num_X,
                            knn_imputed_X_cat_dropped,
                            si_mean_imputed_X_cat_dropped,
                            si_median_imputed_X_cat_dropped,
                            si_most_frequent_imputed_X_cat_dropped,
                            iter_imputed_X_cat_dropped,
                            miss_val_dropped_X]
    list_X_after_missing_values_names= ["knn_imputed_num_X",
                            "si_mean_imputed_num_X",
                            "si_median_imputed_num_X",
                            "si_most_frequent_imputed_num_X",
                            "iter_imputed_num_X",
                            "knn_imputed_X_cat_dropped",
                            "si_mean_imputed_X_cat_dropped",
                            "si_median_imputed_X_cat_dropped",
                            "si_most_frequent_imputed_X_cat_dropped",
                            "iter_imputed_X_cat_dropped",
                            "miss_val_dropped_X"]
    # st.write("si_most_frequent_imputed_num_X",si_most_frequent_imputed_num_X,)   
    ord_enc_cols = []
    ohe_enc_cols = []

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
        
        ordinal_order_vals = []

        for column in ord_enc_cols:
            unique_vals = X.dropna()[column].unique()
            # st.write(f"No. of Unique value in {column} column are", len(unique_vals))
                            
            ordered_unique_vals = st.multiselect("Select values in order for Ordinal Encoding",unique_vals,unique_vals)
            ordinal_order_vals.append(ordered_unique_vals)
        
        st.write("order of values for Ordinal Encoding",tuple(ordinal_order_vals))  

        if len_null > 0: 

            for df_name, df in enumerate(list_X_after_missing_values):
                # st.write(f"{list_X_after_missing_values_names[df_name]}",df)
                from sklearn.preprocessing import OrdinalEncoder
                ord = OrdinalEncoder(categories=ordinal_order_vals,handle_unknown= "use_encoded_value",unknown_value = -1 )
                df[ord_enc_cols] = ord.fit_transform(df[ord_enc_cols])
                # st.write(f"{list_X_after_missing_values_names[df_name]}",df)
        else :  
            from sklearn.preprocessing import OrdinalEncoder
            ord = OrdinalEncoder(categories=ordinal_order_vals,handle_unknown= "use_encoded_value",unknown_value = -1 )
            X[ord_enc_cols] = ord.fit_transform(X[ord_enc_cols])

        st.write("Ordinal Encoding Completed ✅")

    if len(ohe_enc_cols)>0:
        if len_null > 0:
            for df_name, df in enumerate(list_X_after_missing_values):
                from sklearn.preprocessing import OneHotEncoder
                ohe = OneHotEncoder(sparse_output = False,handle_unknown = "ignore")
                pd.options.mode.chained_assignment = None
                df.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(df[ohe_enc_cols])
                df.drop(columns = ohe_enc_cols,inplace = True)
                pd.options.mode.chained_assignment = 'warn'
        else:
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder(sparse_output = False,handle_unknown = "ignore")
            pd.options.mode.chained_assignment = None
            X.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(X[ohe_enc_cols])
            X.drop(columns = ohe_enc_cols,inplace = True)
            pd.options.mode.chained_assignment = 'warn'
        st.write("OneHot Encoding Completed ✅")


    if len(ohe_enc_cols)>0:
        if len_null > 0:
            for name,df in enumerate(list_X_after_missing_values):
                X_train,X_test,y_train,y_test = tts(df,y[df.index],test_size =.2 ,random_state = 42)
                #  best_tts.best_tts(df,y,model,eva)
                evaluationer.evaluation(f"{list_X_after_missing_values_names[name]}",X_train,X_test,y_train,y_test,model,root_mean_squared_error,eva)
        else:
            X_train,X_test,y_train,y_test = tts(X,y[X.index],test_size =.2 ,random_state = 42)
            #  best_tts.best_tts(X,y,model,eva)
                
            evaluationer.evaluation(f"baseline_model",X_train,X_test,y_train,y_test,model,root_mean_squared_error,eva)

    if len_null >0:
        for name,df in enumerate(list_X_after_missing_values):
            X_train,X_test,y_train,y_test = tts(df,y[df.index],test_size =.2 ,random_state = 42)
            st.write(f"this is test{list_X_after_missing_values_names[name]}",X_train.isnull().sum().sum())
            evaluationer.evaluation(f"{list_X_after_missing_values_names[name]}",X_train,X_test,y_train,y_test,model,root_mean_squared_error,eva)

    if eva == "class":
        counter = Counter(y)
        total = sum(counter.values())
        balance_ratio = {cls: count / total for cls, count in counter.items()}
        num_classes = len(balance_ratio)
        ideal_ratio = 1 / num_classes
        a = all(abs(ratio - ideal_ratio) <= 0.1 * ideal_ratio for ratio in balance_ratio.values())
        if a == True:
            st.write("Balanced Dataset ✅")
            st.write("Using accuracy for Evaluation")
            value = "test_acc"
        else:
            st.write("Unbalanced Dataset ❌")
            st.write("Using F1 score for Evaluation")
            value = "test_f1"
        st.write("SFdfs",evaluationer.classification_evaluation_df)
        evaluationer.classification_evaluation_df.sort_values(by = value,inplace= True)
        name = str(evaluationer.classification_evaluation_df.iloc[-1,0])
        st.write("df name",evaluationer.classification_evaluation_df.iloc[-1,0])
        if len_null >0:
            b = list_X_after_missing_values_names.index(name)
            st.write("Sdffsf",b)
            st.write("df",list_X_after_missing_values[b])
            X = list_X_after_missing_values[b]
    if eva == "reg":
        st.write("Using R2 score for Evaluation",evaluationer.reg_evaluation_df)
        value = "test_r2"
        evaluationer.reg_evaluation_df.sort_values(by = value,inplace= True)
        st.write("adfsdf",evaluationer.reg_evaluation_df.iloc[-1,0])
        name = str(evaluationer.reg_evaluation_df.iloc[-1,0])
        st.write("Sdffsf",name)
        if len_null >0:
            b = list_X_after_missing_values_names.index(name)
            st.write("Sdffsf",b)
            st.write("df",list_X_after_missing_values[b])
            X = list_X_after_missing_values[b]
    

    # Create a figure and axes
    num_plots = len(num_cols)
    cols = 2  # Number of columns in the subplot grid
    rows = (num_plots + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Flatten the axes array for easy iteration, and remove any excess subplots
    axes = axes.flatten()
    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    for i, col in enumerate(num_cols):
        sns.histplot(X[col], ax=axes[i],kde = True,color=sns.color_palette('Oranges', as_cmap=True)(0.7))
        axes[i].set_title(col)

    # Adjust layout
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)

    # Create a figure and axes
    num_plots = len(num_cols)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_plots + cols - 1) // cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    # Flatten the axes array for easy iteration, and remove any excess subplots
    axes = axes.flatten()
    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    for i, col in enumerate(num_cols):
        sns.boxplot(y=X[col], ax=axes[i],palette="magma")
        axes[i].set_title(col)

    # Adjust layout
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)
     
    outlier_cols = st.multiselect("De-Select columns for Detecting Outliers", num_cols,default= list(num_cols))

    st.write("Checking for Outliers")
    outliers_df_X,outlier_indexes = outliers.detect_outliers(X,list(outlier_cols))
    st.write("Outliers in Dataframe Summary",outliers_df_X)
    st.write("Columns for Outliers handling",tuple(outliers_df_X["columns name"]))

    select_outlier_cols = st.multiselect("Select columns for Outlier Handling",tuple(outliers_df_X["columns name"]),default =tuple(outliers_df_X["columns name"]))
    resultant,outlier_handled_df,outlier_handled_df_name= outliers.outlier_handling(X,y,model,outlier_indexes = outlier_indexes,outlier_cols = select_outlier_cols ,method = root_mean_squared_error,test_size = 0.2, random_state = 42,eva = "reg")
    st.write("outlier handling with methods",resultant)
    st.write("Best method with outlier handling",resultant.sort_values(by = "test_r2").tail(1).iloc[:,0].values[0])
    try :
        st.write("Best X Data Index No.",outlier_handled_df_name.index(resultant.sort_values(by = "test_r2").tail(1).iloc[:,0].values[0]))
    
        st.write("Best X DataFrame after outlier handling ",outlier_handled_df[outlier_handled_df_name.index(resultant.sort_values(by = "test_r2").tail(1).iloc[:,0].values[0])])
        X = outlier_handled_df[outlier_handled_df_name.index(resultant.sort_values(by = "test_r2").tail(1).iloc[:,0].values[0])]
    except :
        "evaluation of baseline model is better continuing with baseline model"
    
    # result_df ,X_train_b,X_test_b,y_train_b,y_test_b = best_tts.best_tts(X,y,model,eva)
    X_train,X_test,y_train,y_test = tts(X,y[X.index],random_state = 42,test_size = 0.2)
    st.write("result_df",X)
    st.write("fsdfs",X_train)
    result_df_1 = feature_selections.feature_selection(X_train,X_test,y_train,y_test,model,alpha = 0.05)
    st.write("sdchsvdgj",result_df_1)

    


    









