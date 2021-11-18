from collections import Counter
import random
from joblib import Parallel, delayed
import logging

import statistics
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import ParameterGrid
from sklearn.utils.fixes import _joblib_parallel_args


class BoostedSupportSubset():
    def __init__(self, 
                 method=SVC, 
                 params={'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}, 
                 sample_size=None, 
                 wrab=True, 
                 max_features='auto', 
                 lam=1, 
                 eval_metric=accuracy_score, 
                 prop_sample=0.1, 
                 n_learners=10,
                 random_state=1234):
        """Minimally Overfitted Ensemble. 

        Args:
            method (scikit-learn estimator, optional): Machine Learning (ML) algorithm. Defaults to SVC.
            params (dict, optional): Hyperparameters of the ML algorithm. Defaults to {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}.
            sample_size (int, optional): Number of samples for every single learner. Defaults to None.
            wrab (bool, optional): Weighted Random Bootstrap. If False standard bootstrap is executing. Defaults to True.
            max_features (str, optional): Number of features for each learner: {"auto", "sqrt", "log2"}. Defaults to 'auto'. 
            lam (int, optional): Grade of overfitting. Defaults to 1.
            eval_metric (scikit-learn scoring, optional): Metric function to measure the error in the overfitting loss function (OLF). Defaults to accuracy_score.
            prop_sample (float, optional): Proportion of samples for every single learner from the learning set. Defaults to 0.1.
            n_learners (int, optional): Number of learners of the ensemble. Defaults to 10.
            random_state (int, optional): Seed for random procedures . Defaults to 1234.
        """
        
        self.method = method
        self.params = params
        
        if self._is_param_grid():
            self.param_grid = list(ParameterGrid(params))
        else:
            self.param_grid = [params]
            
        self.sample_size = sample_size
        self.wrab = wrab
        self.rof = 0
        self.max_features = max_features
        self.lam = lam
        self.eval_metric = eval_metric
        self.prop_sample = prop_sample
        self.n_learners = n_learners
        self.random_state = random_state
        self.learners = None

    def _support_subset_estimation(self, sample, target, clf, prop=1):


        # Evaluación de distancia al hiperplano
        decision_function_values = clf.decision_function(sample)
        
        # Índices de los vectores soporte
        alphas_index = clf.support_
        
        # Identificación de los vectores soporte
        pos_support = alphas_index[np.where(decision_function_values[alphas_index] > 0)]
        neg_support = alphas_index[np.where(decision_function_values[alphas_index] < 0)]
        decision_function_values[pos_support] = float("inf")
        decision_function_values[neg_support] = float("-inf")
        
        # Conteo de vectores soporte según clase
        nsv_class = Counter(target[alphas_index])
        
        # Tamaño muestra de los sunconjuntos positivo y negativo
        samp_prop = [np.max([nsv_class[i] + 50, prop * nsv_class[i]]) for i in nsv_class]
        
        # # Límites de la función de decisión
        # bounds = [decision_function_values[alphas_index].min(), decision_function_values[alphas_index].max()]
        
        # Definición de subconjuntos positivo y negativo
        pos_values = np.where(decision_function_values>0)[0]
        neg_values = np.where(decision_function_values<0)[0]
        x_pos = pos_values[(decision_function_values[pos_values]).argsort().argsort()<samp_prop[1]]
        x_neg = neg_values[((-1)*decision_function_values[neg_values]).argsort().argsort()<samp_prop[-1]]
            
        return nsv_class, x_pos, x_neg
    
    
    def _is_param_grid(self):
        """Private function for checking param_grid format. """
        
        search_best = any([isinstance(i, list) for i in self.params.values()])

        return search_best
    
    
    def _generate_sample_indices(self, data_train, target, idx_learner):
        """ Private function used to _parallel_build_learners function. """

        if self.wrab:
            random_instance = check_random_state(self.random_state + idx_learner)
            prop_class_ini = round(random_instance.uniform(0.05, 0.95), 2)
            num_samples_class_ini = max(int(prop_class_ini* self.num_samples_boostrap), 1)
            num_samples = [num_samples_class_ini, self.num_samples_boostrap-(num_samples_class_ini)]
            classes_index = [np.where(target == category)[0] for category in np.unique(target)]
            
            train_index =[]
            for i in range(len(np.unique(target))):
                train_index.extend(random_instance.choice(classes_index[i], int(num_samples[i])))
        else:
            for i in range(1000):
                random_instance = check_random_state(self.random_state + idx_learner + i)
                train_index = random_instance.randint(0, data_train.shape[0], self.num_samples_boostrap)
                if len(np.unique(target[train_index]))>1:
                    break                    
            else:
                print('No valid sample in 1000 iters')                    
        oob_index = list(set(range(data_train.shape[0])) - set(np.unique(train_index)))
        
        if self.max_features == self.n_features_:
            selected_features = range(self.n_features_)
        else:
            selected_features = random_instance.choice(range(self.n_features_), self.max_features_, replace=False)
                    
        return train_index, oob_index, np.sort(selected_features)

    
    def _parallel_build_learners(self, data_train, target, idx_learner, verbose):
        """Private function to construct each limited learner parallely

        Args:
            data_train (array): Partial learning set
            target (array): labels
            idx_learner (array): learner id 
            verbose (string): Logging output

        Returns:
            [dict]: Information of each single learner.
        """
        
        train_index, oob_index, selected_features = self._generate_sample_indices(data_train, target, idx_learner)
  
        X_train = data_train[np.ix_(train_index, selected_features)]
        y_train = target[train_index]
        
        X_oob = data_train[np.ix_(oob_index, selected_features)]
        y_oob = target[oob_index]

        best_learner = None
        best_score = float("inf")
        best_learner_train_error = None
        best_learner_test_error = None
        
        best_oob_error = None
        
        for learner_params in self.param_grid:
            
            try:
                learner = self.method(**learner_params, random_state=self.random_state)
            except:
                learner = self.method(**learner_params)
                
            try:
                learner.fit(X_train, y_train)
                learner_score, train_error, oob_error = self.model_score(learner, X_train, y_train, X_oob, y_oob)
            except:
                pass
            
            if  (learner_score < best_score): 
                best_learner = learner
                best_score = learner_score
                best_learner_train_error = train_error
                best_oob_error = oob_error
            
            if verbose:
                logging.error(f'Model:{learner}, score: {learner_score}, train_error: {train_error}, oob_error: {oob_error}')
                
        if verbose:
            logging.error(f'Best Model:{best_learner}, score: {best_score}')

        return {
                'data': {
                    'train_indexes': train_index,
                    'oob_indexes': oob_index,
                    'selected_features': selected_features
                 },
                'learner': best_learner, 
                'scores':{
                    'best_score': best_score,
                    'train_error': best_learner_train_error,
                    'oob_error':  best_oob_error
                }
               }
    
    def fit(self, data_train, target, n_jobs=-1, verbose=False):
        """ Find the minimally overfitted learner to each drawn sample.

        Args:
            data_train (array): Learning set.
            target (array): Labels of the learning set.
            n_jobs (int, optional): Number of processes in parallel tasks. Defaults to -1.
            verbose (bool, optional): Log information. Defaults to False.
        """
      
        n_samples, self.n_features_ = data_train.shape
        
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                    num_features = self.n_features_
            elif self.max_features == "sqrt":
                num_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                num_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")

        self.max_features_ = num_features
        
        if self.sample_size is not None:
            self.num_samples_boostrap = self.sample_size
        else:
            self.num_samples_boostrap = int(data_train.shape[0]*self.prop_sample)

        self.learners = Parallel(n_jobs=n_jobs, verbose=verbose, **_joblib_parallel_args(prefer='threads'))(
        delayed(self._parallel_build_learners)(data_train, target, idx_learner, verbose)
        for idx_learner in range(self.n_learners))
        
            
    def model_score(self, model, X_train, y_train, X_oob, y_oob):
        """ Evaluation of the Overfitting Loss Function (OLF)

        Args:
            model (scikit-learn estimator): [description]
            X_train (array): Learning set of the single learner.
            y_train (array): Target variable.
            X_oob (array): Out of bag (OOB) set.
            y_oob (array): OOB target variable.

        Returns:
            list: 3-tupla consisting of OLF value, train error and OOB error
        """
        train_prediction = model.predict(X_train)
        train_error = 1-self.eval_metric(y_train, train_prediction)

        oob_prediciton = model.predict(X_oob)
        oob_error = 1-self.eval_metric(y_oob, oob_prediciton)
        
        if (oob_error - train_error > self.rof):
            
            model_score = train_error + self.lam*(oob_error - train_error)**2
            
        else:
            
            model_score = float("inf")
        
        return model_score, train_error, oob_error
    
    def _parallel_predict(self, X, learner):
        """ Private function to parallel prediction.

        Args:
            X (array): Set from which the predictions are required.
            learner ([type]): Single learner.

        Returns:
            [array]: Predictions.
        """
        
        try:
            preds = learner['learner'].predict(X[:, learner['data']['selected_features']])
        except:
            preds = None
        return preds
    
    
    def predict(self, X, learners_predictions_return=False):
        """ Function to obtain new predictions.

        Args:
            X (array): Set from which the predictions are required.
            learners_predictions_return (bool, optional): List of predictions from each single learner. Defaults to False.

        Returns:
            [array]: Predictions.
        """
        
        learners_predictions = Parallel(n_jobs=-1, **_joblib_parallel_args(prefer='threads'))(
        delayed(self._parallel_predict)(X, learner) for learner in self.learners)
        
        learners_predictions = [pred for pred in learners_predictions if pred is not None]
        
        if learners_predictions_return:
            predictions = [learners_predictions, np.apply_along_axis(statistics.mode, 0, learners_predictions)]
        else:
            predictions = np.apply_along_axis(statistics.mode, 0, learners_predictions)
        
        return predictions
