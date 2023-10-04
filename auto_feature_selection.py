import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter
import math
from scipy import stats

# Importing all the feature selection models
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier



def preprocess_dataset(dataset_path):
    player_df = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
    player_df = player_df[numcols+catcols]
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
    features = traindf.columns
    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf,columns=features)
    y = traindf['Overall']>=87
    X = traindf.copy()
    del X['Overall']
    feature_name = list(X.columns)
    num_feats=30

    return X, y, num_feats, feature_name


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    # preprocessing
    X, y, num_feats, feature_name = preprocess_dataset(dataset_path)
    
    # PEARSON METHOD
    def cor_selector(X, y,num_feats):
        cor_list = []
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i],y)[0,1]
            cor_list.append(cor)
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
        cor_support = [True if i in cor_feature else False for i in feature_name]
        return cor_support, cor_feature
    
    # CHI-SQUARED METHOD
    def chi_squared_selector(X, y, num_feats): 
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)
        chi_selector = SelectKBest(chi2, k=num_feats)
        chi_selector.fit(X_norm, y)
        chi_support = chi_selector.get_support()
        chi_feature = X.loc[:,chi_support].columns.tolist()
        return chi_support, chi_feature

    # RFE METHOD
    def rfe_selector(X, y, num_feats):
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
        rfescaler = MinMaxScaler()
        X_norm = rfescaler.fit_transform(X)
        rfe_selector.fit(X_norm, y)
        rfe_support = rfe_selector.get_support()
        rfe_feature = X.loc[:,rfe_support].columns.tolist()
        return rfe_support, rfe_feature

    # LOGISTIC-REGRESSION / LASSO METHOD
    def embedded_log_reg_selector(X, y, num_feats):
        embeded_scaler = MinMaxScaler()
        X_norm = embeded_scaler.fit_transform(X)
        # embedded_lr_selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'), max_features = num_feats)
        embedded_lr_selector = SelectFromModel(LogisticRegression(penalty='l2'), max_features = num_feats)
        embedded_lr_selector.fit(X_norm,y)

        embedded_lr_support = embedded_lr_selector.get_support()
        embedded_lr_feature = X.loc[:,embedded_lr_support].columns.tolist()
        # Your code ends here
        return embedded_lr_support, embedded_lr_feature

    # RF METHOD
    def embedded_rf_selector(X, y, num_feats):
        embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features = num_feats)
        embedded_rf_selector.fit(X,y)
        embedded_rf_support = embedded_rf_selector.get_support()
        embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
        return embedded_rf_support, embedded_rf_feature
    
    # LGBM METHOD
    def embedded_lgbm_selector(X, y, num_feats):
        lgbc = LGBMClassifier(n_estimators=100, learning_rate=0.01, num_leaves=2, colsample_bytree=0.2,
                          reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=10,
                          force_col_wise=True)
        embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
        embedded_lgbm_selector.fit(X,y)

        embedded_lgbm_support = embedded_lgbm_selector.get_support()
        embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()
        return embedded_lgbm_support, embedded_lgbm_feature

    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    #### Your Code starts here (Multiple lines)
    pd.set_option('display.max_rows', None)
    # put all selection together
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 
                                        'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                        'Random Forest':embedded_rf_support, 'LightGBM':embedded_lgbm_support})
    feature_selection_df['Total'] = feature_selection_df.iloc[:,1:].sum(axis=1)

    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)

    # Getting top 30 features
    best_features = feature_selection_df['Feature'].head(num_feats)
    return best_features


dataset = input("Enter the path of dataset: ")

best_features = autoFeatureSelector(dataset_path = dataset, methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
print("\n\nTop Features of the datasets: ")
print(best_features)