from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import streamlit as st
import evaluationer

from sklearn.metrics import root_mean_squared_error

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import MultinomialNB, CategoricalNB

param_grids_class = {
    "Logistic Regression": {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear', 'saga']
    },
    
    "SGD Classifier": {
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 5000, 10000]
    },
    
    "Ridge Classifier": {
        'alpha': [0.1, 1, 10, 100]
    },
    
    "Random Forest Classifier": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    "AdaBoost Classifier": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    
    "Gradient Boosting Classifier": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    "Hist Gradient Boosting Classifier": {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [20, 50, 100]
    },
    
    "K Neighbors Classifier": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    
    "Decision Tree Classifier": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    "SVC": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf'],
        'degree': [3, 4, 5],
        'gamma': ['scale', 'auto']
    },
    
    "XGB Classifier": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    "XGBRF Classifier": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    "MLP Classifier": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    },
    
    "LGBM Classifier": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [-1, 10, 20]
    },
    
    "Multinomial Naive Bayes": {
        'alpha': [0.1, 0.5, 1.0]
    },
    
    "Categorical Naive Bayes": {
        'alpha': [0.1, 0.5, 1.0]
    }
}

param_grids_reg = {
    "Linear Regression": {},
    
    "SGD Regressor": {
        'loss': ['squared_loss', 'huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 5000, 10000]
    },
    
    "Ridge Regressor": {
        'alpha': [0.1, 1, 10, 100],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr']
    },
    
    "Lasso Regressor": {
        'alpha': [0.1, 1, 10, 100]
    },
    
    "ElasticNet Regressor": {
        'alpha': [0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.5, 0.9]
    },
    
    "Random Forest Regressor": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    "AdaBoost Regressor": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    
    "Gradient Boosting Regressor": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    "Hist Gradient Boosting Regressor": {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [20, 50, 100]
    },
    
    "K Neighbors Regressor": {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    
    "Decision Tree Regressor": {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    
    "SVR": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf'],
        'degree': [3, 4, 5],
        'gamma': ['scale', 'auto']
    },
    
    "XGB Regressor": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    "XGBRF Regressor": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    
    "MLP Regressor": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    },
    
    "LGBM Regressor": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [-1, 10, 20]
    },
    
    "Gaussian Naive Bayes": {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
}

# Define the regressors
regressors = {
    "Linear Regression": LinearRegression(),
    "SGD Regressor": SGDRegressor(),
    "Ridge Regressor": Ridge(),
    "Lasso Regressor": Lasso(),
    "ElasticNet Regressor": ElasticNet(),
    "Random Forest Regressor": RandomForestRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(),
    "K Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "SVR": SVR(),
    "XGB Regressor": XGBRegressor(),
    "XGBRF Regressor": XGBRFRegressor(),
    "MLP Regressor": MLPRegressor(),
    "LGBM Regressor": LGBMRegressor(),
    "Gaussian Naive Bayes": GaussianNB()
}

classifiers = {
    "Logistic Regression": LogisticRegression(),
    "SGD Classifier": SGDClassifier(),
    "Ridge Classifier": RidgeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Hist Gradient Boosting Classifier": HistGradientBoostingClassifier(),
    "K Neighbors Classifier": KNeighborsClassifier(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "SVC": SVC(),
    "XGB Classifier": XGBClassifier(),
    "XGBRF Classifier": XGBRFClassifier(),
    "MLP Classifier": MLPClassifier(),
    "LGBM Classifier": LGBMClassifier(),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Categorical Naive Bayes": CategoricalNB()
}
def perform_grid_search(model,model_name,X_train,X_test,y_train,y_test,eva):
    if eva == "reg":
        regressor = regressors[model_name]
    
        param_grid_reg = param_grids_reg[model_name]

        grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid_reg, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train,y_train)
        st.write(f"Best Parameters for {model_name}: {grid_search.best_params_}")
        st.write(f"Best Score for {model_name}: {grid_search.best_score_}")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        evaluationer.evaluation("best hyperparams",X_train,X_test,y_train,y_test,model,root_mean_squared_error,eva)
    elif eva == "class":
        classifier = classifiers[model_name]
        param_grid_class = param_grids_class[model_name]

        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid_class, cv=5, scoring='accuracy')
        grid_search.fit(X_train,y_train)
        st.write(f"Best Parameters for {model_name}: {grid_search.best_params_}")
        st.write(f"Best Score for {model_name}: {grid_search.best_score_}")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        evaluationer.evaluation("best hyperparams",X_train,X_test,y_train,y_test,model,root_mean_squared_error,eva)
