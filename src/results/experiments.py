import os
import pandas as pd
import pickle
import warnings

from src.model.SupportSubsetEstimator import *
from src.model.moess import *
from src.utils import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

for experiment in [
#  'ilpd',
#  'banknote',
#  'fourclass',
#  'mushrooms',
#  'svmguide3',
#  'transfusion',
#  'german_numer',
#  'liver-disorders',
#  'heart',
 'r2',
 'haberman',
 'svmguide1',
 'breastcancer',
 'australian',
 'diabetes',
 'mammographic',
 'ionosphere',
 'colon-cancer'
 ]:
    
# Experiment paths

    print(f'Experiment: {experiment}\n')

    results_folder = f'results/{experiment}'

    os.makedirs(results_folder, exist_ok=True)

    data = pd.read_parquet(f'data/prep_real_data/{experiment}.parquet')

    # hyperparameters gridsearch
    random_state = 1234

    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))
    y = data.y.values

    # Calcular particiones
    skf = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)
    ss_estimator = SupportSubsetEstimator()
    ss_idx = []
    train_idx = []
    test_idx = []

    for train_index, test_index in tqdm(list(skf.split(X, y)), desc="Estimating Support Subsets"):
        ss_estimator.fit(X[train_index], y[train_index])
        train_idx.append(train_index)
        test_idx.append(test_index)
        ss_idx.append(ss_estimator.supportsubset)

    folds = list(zip(train_idx, test_idx, ss_idx))

    grid_params = {
        'wrab': [True, False],
        'lam': [1, 3, 5],
        'prop_sample': [0.10, 0.20, 0.30],
        'n_learners': [10, 20, 30],
        'random_state': [random_state]
    }

    # MOE-SVM

    ensemble = MOESS

    ensemble_grid = GridSearch_moess(ensemble, grid_params, scoring=scaled_mcc, folds=folds, n_jobs=-1)

    print('MOESS svm : \n')

    ensemble_grid.fit(X, y)

    print(f'params: {ensemble_grid.best_params_}, score: {ensemble_grid.best_score_}\n')

    with open(f'results/{experiment}/MOESS_svm.p', 'wb') as fout:
        pickle.dump(ensemble_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)


    # MOESS-kNN

    kwargs = {'method':KNeighborsClassifier, 'params':{'n_neighbors' : list(range(1, 13, 2))}}

    ensemble = MOESS

    ensemble_grid = GridSearch_moess(ensemble, grid_params, scoring=scaled_mcc, folds=folds, n_jobs=-1, kwargs=kwargs)

    print('MOESS knn: \n')

    ensemble_grid.fit(X, y)

    print(f'params: {ensemble_grid.best_params_}, score: {ensemble_grid.best_score_}\n')

    with open(f'results/{experiment}/MOESS_knn.p', 'wb') as fout:
        pickle.dump(ensemble_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)


    # MOE-DT

    kwargs = {'method':DecisionTreeClassifier, 'params':{'criterion' : {"gini", "entropy"}, 'max_depth' : list(range(1, 10, 1))}}

    ensemble = MOESS

    ensemble_grid = GridSearch_moess(ensemble, grid_params, scoring=scaled_mcc, folds=folds, n_jobs=-1, kwargs=kwargs)

    print('MOESS dt: \n')

    ensemble_grid.fit(X, y)

    print(f'params: {ensemble_grid.best_params_}, score: {ensemble_grid.best_score_}\n')

    with open(f'results/{experiment}/MOESS_dt.p', 'wb') as fout:
        pickle.dump(ensemble_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)


