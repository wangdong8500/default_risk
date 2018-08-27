import pandas as pd
from time import time
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from models import kfold_lightgbm, kfold_xgb, kfold_cat


def oof_regression_stacker(train_x, train_y, test_x,
                           estimators, 
                           pred_cols, 
                           train_eval_metric, 
                           compare_eval_metric,
                           n_folds = 5,
                           holdout_x=False,
                           debug = True):
    
    """
    Original script:
        Jovan Sardinha
        https://medium.com/weightsandbiases/an-introduction-to-model-ensembling-63effc2ca4b3
        
    Args:
        train_x, train_y, test_x (DataFrame).
        n_folds (int): The number of folds for crossvalidation.
        esdtimators (list): The list of estimator functions.
        pred_cols (list): The estimator related names of prediction columns.
        train_eval_metric (class): Fucntion for the train eval metric.
        compare_eval_metric (class): Fucntion for the crossvalidation eval metric.
        holdout_x (DataFrame): Holdout dataframe if you intend to stack/blend using holdout.
        
    Returns:
        train_blend, test_blend, model
    """
    
    if debug == True:
        train_x = train_x.sample(n=1000, random_state=seed_val)
        train_y = train_y.sample(n=1000, random_state=seed_val)
        test_x = test_x.sample(n=1000, random_state=seed_val)
        
    # Start timer:
    start_time = time()
    
    # List to save models:
    model_list = []
    
    # Initializing blending data frames:
    with_holdout = isinstance(holdout_x, pd.DataFrame)
    if with_holdout: 
        hold_blend = pd.DataFrame(holdout_x.index)
    
    train_blend = pd.DataFrame(train_x.index)
    val_blend = pd.DataFrame(train_x.index)
    test_blend = pd.DataFrame(test_x.index)

    # Note: StratifiedKFold splits into roughly 66% train 33% test  
    folds = StratifiedShuffleSplit(n_splits= n_folds, random_state=seed_val)
    
    # Arrays to hold estimators' predictions:
    dataset_blend_val = np.zeros((train_x.shape[0], len(estimators))) # Validfation prediction holder
    dataset_blend_test = np.zeros((test_x.shape[0], len(estimators))) # Mean test prediction holder
    dataset_blend_train = np.zeros((train_x.shape[0], len(estimators))) # Mean train prediction holder
    if with_holdout: 
        dataset_blend_test = np.zeros((holdout_x.shape[0], len(estimators))) # Same for holdout
        
    # For every estimator:
    for j, estimator in enumerate(estimators):
        
        # Array to hold folds number of predictions on test:
        dataset_blend_test_j = np.zeros((test_x.shape[0], n_folds))
        dataset_blend_train_j = np.zeros((train_x.shape[0], n_folds))
        if with_holdout: 
            dataset_blend_holdout_j = np.zeros((holdout_x.shape[0], n_folds))
        
        # For every fold:
        for i, (train, test) in enumerate(folds.split(train_x, train_y)):
            trn_x = train_x.iloc[train, :] 
            trn_y = train_y.iloc[train].values.ravel()
            val_x = train_x.iloc[test, :] 
            val_y = train_y.iloc[test].values.ravel()
            
            # Estimators conditional training:
            if estimator == 'lgb':
                model = kfold_lightgbm(trn_x, trn_y, num_folds = 3, seed_val = seed_val)
            elif estimator == 'xgb':
                model = kfold_xgb(trn_x, trn_y, num_folds = 3, seed_val = seed_val)
            else:
                model = kfold_cat(trn_x, trn_y, num_folds = 3,  seed_val = seed_val)
            
            # Validation:
            if estimator == 'xgb':
                pred_val = xgb_predict(val_x, model)
                pred_test = xgb_predict(test_x, model)
                pred_train = xgb_predict(train_x, model)
                if with_holdout:
                    pred_holdout = xgb_predict(holdout_x, model)
            else:
                pred_val = model.predict(val_x)
                pred_test = model.predict(test_x)
                pred_train = model.predict(train_x)
                if with_holdout:
                    pred_holdout = model.predict(holdout_x)
                    
            dataset_blend_val[test, j] = pred_val
            dataset_blend_test_j[:, i] = pred_test
            dataset_blend_train_j[:, i] = pred_train
            if with_holdout: 
                dataset_blend_holdout_j[:, i] = pred_holdout
                
            print('fold:', i+1, '/', n_folds,
                  '; estimator:',  j+1, '/', len(estimators),
                  ' -> oof cv score:', compare_eval_metric(val_y, pred_val))

            del trn_x, trn_y, val_x, val_y
            gc.collect()
            
        # Save curent estimator's mean prediction for test, train and holdout:
        dataset_blend_test[:, j] = np.mean(dataset_blend_test_j, axis=1)
        dataset_blend_train[:, j] = np.mean(dataset_blend_train_j, axis=1)
        if with_holdout: 
            dataset_blend_holdout[:, j] = np.mean(dataset_blend_houldout_j, axis=1)
        
        model_list += [model]
        
    print('--- comparing models ---')
    # comparing models
    for i in range(dataset_blend_val.shape[1]):
        print('model', i+1, ':', compare_eval_metric(train_y, dataset_blend_val[:,i]))
        
    for i, j in enumerate(estimators):
        val_blend[pred_cols[i]] = dataset_blend_val[:,i]
        test_blend[pred_cols[i]] = dataset_blend_test[:,i]
        train_blend[pred_cols[i]] = dataset_blend_train[:,i]
        if with_holdout: 
            holdout_blend[pred_cols[i]] = dataset_blend_holdout[:,i]
        else:
            holdout_blend = False
    
    end_time = time.time()
    print("Total Time usage: " + str(int(round(end_time - start_time))))
    return train_blend, val_blend, test_blend, holdout_blend, model_list