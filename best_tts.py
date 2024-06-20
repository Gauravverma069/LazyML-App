from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score,f1_score,accuracy_score, root_mean_squared_error
import evaluationer 
import pandas as pd
import numpy as np

def best_tts(X,y,model,eva):
    # def best_tts(X,y,test_size_range = range(10,25),random_state_range =range(1,100), stratify=None,shuffle=True,model = LinearRegression(),method = root_mean_squared_error,eva = "reg"):
   
    if eva == "reg":
        
        test_r2_,test_r2_ts,test_r2_rs = 0,0,0
        for k in range(10,25,3):
            i = k/100
            for j in range(1,100,10):
                X_train,X_test,y_train,y_test = tts(X,y[X.index],test_size = i, random_state = j,)
                
                model = model
                model.fit(X_train,y_train) # model fitting
                y_pred_train = model.predict(X_train) # model prediction for train
                y_pred_test = model.predict(X_test) # model prediction for test
    
                train_r2 = r2_score(y_train, y_pred_train) # evaluating r2 score for train
                
                
                test_r2 = r2_score(y_test, y_pred_test)  # evaluating r2 score for test
                if test_r2_ < test_r2:
                    test_r2_ = test_r2
                    test_r2_ts = i
                    test_r2_rs = j
    
                n_r_train, n_c_train = X_train.shape # getting no of rows and columns of train data
                n_r_test,  n_c_test = X_test.shape # getting no of rows and columns of test data
    
                adj_r2_train = 1 - ((1 - train_r2)*(n_r_train - 1)/ (n_r_train - n_c_train - 1))  # evaluating adjusted r2 score for train
                
                    
                adj_r2_test = 1 - ((1 - test_r2)*(n_r_test - 1)/ (n_r_test - n_c_test - 1)) # evaluating adjusted r2 score for test
                
                    
                train_evaluation = root_mean_squared_error(y_train, y_pred_train) # evaluating train error
                
                    
                test_evaluation = root_mean_squared_error(y_test, y_pred_test) # evaluating test error
                
        X_train,X_test,y_train,y_test = tts(X,y[X.index],test_size = test_r2_ts, random_state = test_r2_rs)           
        evaluationer.evaluation("best_tts",X_train,X_test,y_train,y_test,model,root_mean_squared_error,eva)
        return evaluationer.reg_evaluation_df,X_train,X_test,y_train,y_test

    
    
    elif eva == "class":        
        global  test_accuracies_,test_accuracies_ts,test_accuracies_rs
        test_accuracies_,test_accuracies_ts,test_accuracies_rs = 0,0,0
        
        for k in range(10,25):
            i = k/100
            for j in range(1,100):
                X_train,X_test,y_train,y_test = tts(X,y[X.index],test_size = i, random_state = j)
                model = model
                model.fit(X_train,y_train) # model fitting
                y_pred_train = model.predict(X_train) # model prediction for train
                y_pred_test = model.predict(X_test) # model prediction for test
                # y_pred_proba_train= model.predict_proba(X_train)
                # y_pred_proba_test= model.predict_proba(X_test)
                
                
                unique_classes = np.unique(y_train)
            
                # Determine the average method
                if len(unique_classes) == 2:
                    # Binary classification
                    # print("Using 'binary' average for binary classification.")
                    average_method = 'binary'
                elif len(unique_classes)!=2:
                    # Determine the distribution of the target column
                    class_counts = np.bincount(y_train)
                    
                    # Check if the dataset is imbalanced
                    imbalance_ratio = max(class_counts) / min(class_counts)
                    
                    if imbalance_ratio > 1.5:
                        # Imbalanced dataset
                        # print("Using 'weighted' average due to imbalanced dataset.")
                        average_method = 'weighted'
                    else:
                        # Balanced dataset
                        # print("Using 'macro' average due to balanced dataset.")
                        average_method = 'macro'
                        # F1 scores
                train_f1_scores = (f1_score(y_train, y_pred_train,average=average_method))
               
                    
                test_f1_scores = (f1_score(y_test, y_pred_test,average=average_method))    
                
                # Accuracies
                train_accuracies = (accuracy_score(y_train, y_pred_train))
                
                test_accuracies = (accuracy_score(y_test, y_pred_test))
                if test_accuracies_ <test_accuracies:
                    test_accuracies_,test_accuracies_ts,test_accuracies_rs =test_accuracies, i,j
        X_train,X_test,y_train,y_test = tts(X,y[X.index],test_size = test_accuracies_ts, random_state = test_accuracies_rs)   
        print(f"test_size = {test_accuracies_ts}, random_state = {test_accuracies_rs}")  

        evaluationer.evaluation("best_tts",X_train,X_test,y_train,y_test,model,root_mean_squared_error,eva)
        

        return evaluationer.classification_evaluation_df,X_train,X_test,y_train,y_test
            
             
