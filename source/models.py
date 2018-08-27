from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

import lightgbm as lgbm
import xgboost as xgb
from catboost import CatBoostClassifier



# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


# LightGBM GBDT with KFold or Stratified KFold
def kfold_model(model, model_type, X_train, y_train, X_test, n_folds, stratified = False, debug= False):

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= n_folds, shuffle=True, random_state=42)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=42)
        
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(X_train.shape[0])
    sub_preds = np.zeros(X_test.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in X_train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
       
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train[feats], y_train)):
        train_x, train_y = X_train[feats].iloc[train_idx], y_train.iloc[train_idx]
        valid_x, valid_y = X_train[feats].iloc[valid_idx], y_train.iloc[valid_idx]
        
        clf = clone(model)
        
        if model_type == 'lightgbm':
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)
            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            sub_preds += clf.predict_proba(X_test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        elif model_type == 'xgboost':
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)
            oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
            sub_preds += clf.predict_proba(X_test[feats])[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_train, oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        X_test['TARGET'] = sub_preds
        test_output = X_test[['SK_ID_CURR', 'TARGET']]  #.to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df, test_output