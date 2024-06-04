import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from scipy.stats import yeojohnson
import evaluationer
from sklearn.model_selection import train_test_split as tts
def detect_outliers(df,num_cols):
    global outlier_df,zscore_cols,outlier_indexes,iqr_cols
    outlier_df = pd.DataFrame({"method" :[],"columns name":[],"upper limit":[],
                           "lower limit":[],"no of Rows":[],"percentage outlier":[]})
    if type(num_cols) == list:
        if len(num_cols)!=0:
            num_cols = num_cols
        else:
            num_cols = df.select_dtypes(exclude = "object").columns.tolist()
    else:
        if num_cols.tolist() != None:
            num_cols = num_cols
        else:
            num_cols = df.select_dtypes(exclude = "object").columns.tolist()
    zscore_cols = []
    iqr_cols = []
    outlier_indexes =[]
    for col in num_cols:
        skewness = df[col].skew()
        if -0.5 <= skewness <= 0.5:
            method = "zscore"
            zscore_cols.append(col)

        else:
            method = "iqr"
            iqr_cols.append(col)
    if len(zscore_cols) >0:
        for col in zscore_cols:
            mean = df[col].mean()
            std = df[col].std()
            ul = mean + (3*std)
            ll = mean - (3*std)
            mask = (df[col] < ll) | (df[col] > ul)
            temp = df[mask]

            Zscore_index = temp.index.tolist()
            outlier_indexes.extend(Zscore_index)

            if len(temp)>0:

                temp_df = pd.DataFrame({"method" : ["ZScore"],
                "columns name" : [col],
                "upper limit" : [round(ul,2)],
                "lower limit" :[ round(ll,2)],
                "no of Rows" : [len(temp)],
                "percentage outlier" : [round(len(temp)*100/len(df),2)]})
                
                outlier_df = pd.concat([outlier_df,temp_df]).reset_index(drop = True)

    else:
        print("No columns for Zscore method")
       
    
    if len(iqr_cols) >0:
        for col in iqr_cols:
            q3 = df[col].quantile(.75)
            q1 = df[col].quantile(.25)
            IQR = q3 -q1
            ul = q3 + 1.5*IQR
            ll = q1 - 1.5*IQR
            mask = (df[col] < ll) | (df[col] > ul)
            temp = df[mask]

            IQR_index = temp.index.tolist()
            outlier_indexes.extend(IQR_index)

            if len(temp)>0:
                list(outlier_indexes).append(list(IQR_index))

                temp_df1 = pd.DataFrame({"method" : ["IQR"],
                "columns name" : [col],
                "upper limit" : [round(ul,2)],
                "lower limit" : [round(ll,2)],
                "no of Rows": [len(temp)],
                "percentage outlier" : [round((len(temp)*100/len(df)),2)]
                                    })
          
                outlier_df = pd.concat([outlier_df,temp_df1]).reset_index(drop = True)
            
    else:
        print("No columns for IQR method")

       
    outlier_indexes = list(set(outlier_indexes))
    
    return outlier_df,outlier_indexes


def outlier_handling(df,y,model,outlier_indexes = [],outlier_cols = None ,method = root_mean_squared_error,test_size = 0.2, random_state = 42,eva = "reg"):
    num_col = df.select_dtypes(exclude = "O").columns
    
    global outliers_dropped_df,log_transformed_df,sqrt_transformed_df,yeo_johnson_transformed_df,rank_transformed_df
    global std_scaler_df,winsorize_transformed_df,inverse_log_transformed_winsorize_df,inverse_sqrt_transformed_winsorize_df,minmaxscaler_df
    if eva == "reg":
        if len(outlier_indexes) ==0:
            print("no outlier indexes passed")
            outliers_dropped_df = df.copy()
        else:
            outliers_dropped_df = df.drop(index =outlier_indexes)
        
        if outlier_cols != None:
                    
            if df[outlier_cols][df[outlier_cols] <0].sum().sum() == 0:
                log_transformed_df = df.copy()
                log_transformed_df[outlier_cols] = np.log(log_transformed_df[outlier_cols] + 1e-5)
                sqrt_transformed_df = df.copy()
                sqrt_transformed_df[outlier_cols] = np.sqrt(sqrt_transformed_df[outlier_cols] + 1e-5)
                inverse_log_transformed_winsorize_df = log_transformed_df.copy()
                inverse_sqrt_transformed_winsorize_df = sqrt_transformed_df.copy()
                for column in outlier_cols:
                    inverse_log_transformed_winsorize_df[column] =  np.exp(winsorize(inverse_log_transformed_winsorize_df[column], limits=[0.05, 0.05]))
                    inverse_sqrt_transformed_winsorize_df[column] =  (winsorize(inverse_sqrt_transformed_winsorize_df[column], limits=[0.05, 0.05]))**2
            else:
                print("df have values less than zero")
            std_scaler_df = df.copy()
            std_scaler_df[outlier_cols] = StandardScaler().fit_transform(std_scaler_df[outlier_cols])
            
            minmaxscaler_df = df.copy()
            minmaxscaler_df[outlier_cols] = MinMaxScaler().fit_transform(minmaxscaler_df[outlier_cols])
     
            yeo_johnson_transformed_df = df.copy()
            for column in outlier_cols:
                try:
                    yeo_johnson_transformed_df[column], lambda_ = yeojohnson(yeo_johnson_transformed_df[column])
    
                except :
                    yeo_johnson_transformed_df[column] = yeo_johnson_transformed_df[column]
    
                    print(f"Yeo-Johnson transformation failed for column '{column}'. Original data used.")
                # yeo_johnson_transformed_df[column], lambda_ = yeojohnson(yeo_johnson_transformed_df[column])
            rank_transformed_df = df.copy()
            rank_transformed_df[outlier_cols] = rank_transformed_df[outlier_cols].rank()
            winsorize_transformed_df = df.copy()
            for column in outlier_cols:
                winsorize_transformed_df[column] = winsorize(winsorize_transformed_df[column], limits=[0.05, 0.05])
                
                
            
        else:
            
            
            if df[num_col][df[num_col] <0].sum().sum() == 0:
                log_transformed_df = df.copy()
                log_transformed_df[num_col] = np.log(log_transformed_df[num_col] + 1e-5)
                sqrt_transformed_df = df.copy()
                sqrt_transformed_df[num_col] = np.sqrt(sqrt_transformed_df[num_col] + 1e-5)
                inverse_log_transformed_winsorize_df = log_transformed_df.copy()
                inverse_sqrt_transformed_winsorize_df = sqrt_transformed_df.copy()
                for column in num_col:
                    inverse_log_transformed_winsorize_df[column] =  np.exp(winsorize(inverse_log_transformed_winsorize_df[column], limits=[0.05, 0.05]))
                    inverse_sqrt_transformed_winsorize_df[column] =  (winsorize(inverse_sqrt_transformed_winsorize_df[column], limits=[0.05, 0.05]))**2
            else:
                
                print("df have values less than zero")
                
            std_scaler_df = df.copy()
            std_scaler_df[outlier_cols] = StandardScaler().fit_transform(std_scaler_df[outlier_cols])
            
            minmaxscaler_df = df.copy()
            minmaxscaler_df[outlier_cols] = MinMaxScaler().fit_transform(minmaxscaler_df[outlier_cols])
            
            yeo_johnson_transformed_df = df.copy()
            for column in num_col:
                try:
                    yeo_johnson_transformed_df[column], lambda_ = yeojohnson(yeo_johnson_transformed_df[column])
    
                except :
                    yeo_johnson_transformed_df[column] = yeo_johnson_transformed_df[column]
    
                    print(f"Yeo-Johnson transformation failed for column '{column}'. Original data used.")
                # yeo_johnson_transformed_df[column], lambda_ = yeojohnson(yeo_johnson_transformed_df[column])
            rank_transformed_df = df.copy()
            rank_transformed_df[num_col] = rank_transformed_df[num_col].rank()
            winsorize_transformed_df = df.copy()
            for column in num_col:
                winsorize_transformed_df[column] = winsorize(winsorize_transformed_df[column], limits=[0.05, 0.05])
                 
        if (df[num_col][df[num_col] <0].sum().sum() == 0):        
            outlier_handled_df = [std_scaler_df,minmaxscaler_df,outliers_dropped_df,log_transformed_df,sqrt_transformed_df,yeo_johnson_transformed_df,
                                  rank_transformed_df,winsorize_transformed_df,inverse_log_transformed_winsorize_df,inverse_sqrt_transformed_winsorize_df]
        
            outlier_handled_df_name = ["std_scaler_df","minmaxscaler_df","outliers_dropped_df", "log_transformed_df","sqrt_transformed_df", "yeo_johnson_transformed_df","rank_transformed_df","winsorize_transformed_df",
                                       "inverse_log_transformed_winsorize_df", "inverse_sqrt_transformed_winsorize_df"]
        elif df[outlier_cols][df[outlier_cols] <0].sum().sum() == 0:
            outlier_handled_df = [std_scaler_df,minmaxscaler_df,outliers_dropped_df,log_transformed_df,sqrt_transformed_df,yeo_johnson_transformed_df,
                                  rank_transformed_df,winsorize_transformed_df,inverse_log_transformed_winsorize_df,inverse_sqrt_transformed_winsorize_df]
        
            outlier_handled_df_name = ["std_scaler_df","minmaxscaler_df","outliers_dropped_df","log_transformed_df", "sqrt_transformed_df","yeo_johnson_transformed_df","rank_transformed_df",
                                       "winsorize_transformed_df","inverse_log_transformed_winsorize_df","inverse_sqrt_transformed_winsorize_df"]
        
        else:
            outlier_handled_df = [std_scaler_df,minmaxscaler_df,outliers_dropped_df,yeo_johnson_transformed_df,rank_transformed_df,winsorize_transformed_df]
        
            outlier_handled_df_name = ["std_scaler_df","minmaxscaler_df","outliers_dropped_df","yeo_johnson_transformed_df","rank_transformed_df","winsorize_transformed_df"]
     
        for j,i in enumerate(outlier_handled_df):
            X_train, X_test, y_train, y_test = tts(i,y[i.index],test_size = test_size, random_state = random_state)
            evaluationer.evaluation(f"{outlier_handled_df_name[j]}",X_train,X_test,y_train,y_test,model,root_mean_squared_error,eva)
            
                
        return evaluationer.reg_evaluation_df , outlier_handled_df,outlier_handled_df_name
    elif eva =="class":
                   
        std_scaler_df = df.copy()
        
        std_scaler_df.loc[:,:] = StandardScaler().fit_transform(std_scaler_df.loc[:,:])
       
        minmaxscaler_df = df.copy()
        minmaxscaler_df.loc[:,:] = MinMaxScaler().fit_transform(minmaxscaler_df.loc[:,:])

        rank_transformed_df = df.copy()
        rank_transformed_df = rank_transformed_df.rank()

        outlier_handled_df = [std_scaler_df,minmaxscaler_df,rank_transformed_df]
        outlier_handled_df_name = ["std_scaler_df","minmaxscaler_df","rank_transformed_df"]

        for j,i in enumerate(outlier_handled_df):
           
            X_train, X_test, y_train, y_test = tts(i,y[i.index],test_size = test_size, random_state = random_state)
            evaluationer.evaluation(f"{outlier_handled_df_name[j]}", X_train,X_test,y_train,y_test,model,root_mean_squared_error,eva = "class")
        return evaluationer.classification_evaluation_df, outlier_handled_df,outlier_handled_df_name
# returning evaluating dataframe
        
