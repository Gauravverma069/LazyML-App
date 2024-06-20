from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm
import pandas as pd
import numpy as np
import evaluationer
import streamlit as st
from sklearn.feature_selection import RFE,RFECV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, RFECV, SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import root_mean_squared_error
def feature_selection(X_train, X_test,y_train,y_test,model_reg,alpha = 0.05):

    model = sm.OLS(y_train, sm.add_constant(X_train))
    model_fit = model.fit()
    pval_cols = model_fit.pvalues[model_fit.pvalues > 0.05].index.tolist()
    coef_cols = model_fit.params[abs(model_fit.params) < 0.001].index.tolist()
    pval_and_coef_cols = list(set(coef_cols) | set(pval_cols))

    mi_scores = mutual_info_regression(X_train, y_train)
    mi = pd.DataFrame()

    mi["col_name"] = X_train.columns
    mi["mi_score"] = mi_scores

    mi_cols = mi[mi.mi_score ==0].col_name.values.tolist()

    corr = X_train.corr()
    
    corru= pd.DataFrame(np.triu(corr),columns = corr.columns , index = corr.index)
    corr_u_cols = corru[corru[(corru > 0.5 )& (corru <1)].any()].index.tolist()
    
    corrl= pd.DataFrame(np.tril(corr),columns = corr.columns , index = corr.index)
    corr_l_cols = corrl[corrl[(corrl > 0.5 )& (corrl <1)].any()].index.tolist()
    
    X_new_vif = sm.add_constant(X_train)
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X_new_vif.columns
    vif["VIF"] = [variance_inflation_factor(X_new_vif.values, i) for i in range(X_new_vif.shape[1])]
    # st.write("gdfgdsdsdfad",vif)
    if len(vif[vif["variables"] == "const"]) == 1:
        vif = vif.drop(index = (vif[vif["variables"] == "const"].index[0]))
    # st.write("gdfgdsad",vif)
    # drop const in vif cols
    # vif_cols = X_new_vif.drop(columns = "const")
    vif_cols = vif[vif.VIF >10].variables.tolist()


    # lasso
    if alpha == "best":
        
        lasso_len = []
        alpha_i = []
        for i in range(1,1000,5):
            j = i/10000

            model_lasso = Lasso(alpha=j)
            model_lasso.fit(X_train, y_train)
            col_df = pd.DataFrame({
                "col_name": X_train.columns,
                "lasso_coef": model_lasso.coef_
            })
            a = len(col_df[col_df.lasso_coef ==0])
            lasso_len.append(a)
            alpha_i.append(j)
        for i in zip(lasso_len,alpha_i):
            print(i)
        input_alpha = float(input("enter alpha"))
        model_lasso = Lasso(alpha=input_alpha)
        model_lasso.fit(X_train, y_train)
        col_df = pd.DataFrame({
            "col_name": X_train.columns,
            "lasso_coef": model_lasso.coef_
        })

        lasso_cols =col_df[col_df.lasso_coef ==0].col_name.tolist()
    else:
        model_lasso = Lasso(alpha=alpha)
        model_lasso.fit(X_train, y_train)
        col_df = pd.DataFrame({
            "col_name": X_train.columns,
            "lasso_coef": model_lasso.coef_
        })

        lasso_cols =col_df[col_df.lasso_coef ==0].col_name.tolist()
        
    feature_cols = [pval_cols,coef_cols,pval_and_coef_cols,mi_cols,corr_u_cols,corr_l_cols,vif_cols,lasso_cols]
    
    for col in feature_cols:
        
        try:
            st.write(f"{col}",X_train.drop(columns = col))
        except:
            st.write(f"error IN col")
    feature_cols_name = ["pval_cols","coef_cols","pval_and_coef_cols","mi_cols","corr_u_cols","corr_l_cols","vif_cols","lasso_cols"]
    st.write("feature_cols", vif_cols)
    for i,j in enumerate(feature_cols):
        evaluationer.evaluation(f"{feature_cols_name[i]}" ,X_train.drop(columns = j),X_test.drop(columns = j),y_train,y_test,model_reg,method = root_mean_squared_error,eva = "reg")
    return evaluationer.reg_evaluation_df,feature_cols,feature_cols_name

def clas_feature_selection(X_train, X_test,y_train,y_test,model,n_features_to_select = None, step=1,importance_getter='auto',refcv_graph= False,C=0.05,k = 10):
    global rfe_cols,rfecv_cols,lasso_cols,chi2_imp_col,mi_imp_col
    rfe = RFE(estimator= model,n_features_to_select = n_features_to_select,importance_getter=importance_getter, step=1)
    rfe.fit(X_train,y_train)
    rfe_cols = X_train.columns[rfe.support_]
    cv = StratifiedKFold(5)
    rfecv = RFECV(estimator=model,
        step=1,
        cv=cv,
        scoring="f1",
        min_features_to_select=1,
        n_jobs=-1)
    rfecv.fit(X_train,y_train)
    rfecv_cols = X_train.columns[rfecv.support_]
    if refcv_graph == True:
        n_scores = len(rfecv.cv_results_["mean_test_score"])
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean test f1")
        plt.errorbar(range(min_features_to_select, n_scores + min_features_to_select),
            rfecv.cv_results_["mean_test_score"],
            yerr=rfecv.cv_results_["std_test_score"],
        )
        plt.grid(True)
        plt.title("Recursive Feature Elimination \nwith correlated features")
        plt.show()
    clf = LogisticRegression(penalty = "l1", C = C,
                         random_state = 42,
                         solver = "liblinear")
    clf.fit(X_train, y_train)
    lasso_cols = clf.feature_names_in_[clf.coef_[0] != 0]
    
    sk = SelectKBest(chi2, k=k)
    X_chi2 = sk.fit_transform(X_train, y_train)
    chi2_imp_col = X_train.columns[sk.get_support()]
    sk = SelectKBest(mutual_info_classif, k=k)
    X_mutual = sk.fit_transform(X_train, y_train)
    mi_imp_col = X_train.columns[sk.get_support()]

    feature_cols = [rfe_cols,rfecv_cols,lasso_cols,chi2_imp_col,mi_imp_col]
    feature_cols_name = ["rfe_cols","rfecv_cols","lasso_cols","chi2_imp_col","mi_imp_col"]
    
    for i,j in enumerate(feature_cols):
        # evaluationerevaluation(f"{feature_cols_name[i]} " ,X_train[j],X_test[j],y_train,y_test,model = model,eva = "class")
        evaluationer.evaluation(f"{feature_cols_name[i]}" ,X_train[j],X_test[j],y_train,y_test,model,method = root_mean_squared_error,eva = "class")
    return evaluationer.classification_evaluation_df  ,  feature_cols,    feature_cols_name