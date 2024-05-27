import pandas as pd
import streamlit as st
# import simple imputer, iter imputer , knn inputer
from sklearn.model_selection import train_test_split as tts
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import evaluationer
# import label, ohe,ordinal encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# creating a function for null_handling with different methods for null value imputing, categorical columns encoding and evaluation

null_value_handling_method_num_cols = ["KNN Imputed","SI Mean Imputed","SI Median Imputed","SI Most Frequent Imputed","Iter Imputed"]
null_value_handling_method_cat_cols = ["SI Most Frequent Imputed (categorical)"]

# dict for null value handling method num cols 

dict1 = {"KNN Imputed" :KNNImputer(n_neighbors = 5),"SI Mean Imputed":SimpleImputer(strategy = "mean"),"SI Median Imputed":SimpleImputer(strategy = "median"),
         "SI Most Frequent Imputed":SimpleImputer(strategy = "most_frequent"),"Iter Imputed":IterativeImputer(max_iter = 200,random_state= 42)}

dict2 = {"SI Most Frequent Imputed (categorical)":SimpleImputer(strategy = "most_frequent")}

# creating dataframe from dict1 and dict2
num_nvh_method_df = pd.DataFrame(data=dict1.values(), index=dict1.keys())
cat_nvh_method_df = pd.DataFrame(data=dict2.values(), index=dict2.keys())

num_imputed_dict = {"KNN Imputed":[],"SI Mean Imputed":[],"SI Median Imputed":[],"SI Most Frequent Imputed":[],"Iter Imputed":[]}

cat_imputed_dict = {"SI Most Frequent Imputed (categorical)":[],"Iter Imputed":[]}

num_imputed_df = pd.DataFrame(data = num_imputed_dict.values(),index = num_imputed_dict.keys())

cat_imputed_df = pd.DataFrame(data = cat_imputed_dict.values(),index = cat_imputed_dict.keys())

final_df = []
def null_handling(X,clean_num_nvh_df,clean_num_nvh_df_cat):
    num_nvh_method = clean_num_nvh_df.columns #KNN Imputed","SI Mean Imputed","SI Media
    cat_nvh_method = clean_num_nvh_df_cat.columns
    for method in num_nvh_method:
        X[clean_num_nvh_df[method].dropna().values] = num_nvh_method_df.loc[method].values[0].fit_transform(X[clean_num_nvh_df[method].dropna().values])
    
    for method in cat_nvh_method:
        X[clean_num_nvh_df_cat[method].dropna().values] = cat_nvh_method_df.loc[method].values[0].fit_transform(X[clean_num_nvh_df_cat[method].dropna().values])

    final_df = X

    return final_df


