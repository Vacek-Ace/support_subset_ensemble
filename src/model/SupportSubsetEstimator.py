from collections import Counter
import numpy as np
from sklearn.svm import SVC

from src.utils import gridsearch, scaled_mcc

class SupportSubsetEstimator(gridsearch):
    def __init__(self, 
                 method=SVC,
                 params={'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}, 
                 scoring=scaled_mcc,
                 cv=10,
                 n_jobs=-1,
                 random_state=1234
                 ):
        super().__init__( 
                 method, 
                 params, 
                 scoring=scoring,
                 cv=cv,
                 n_jobs=n_jobs,
                 random_state=random_state)
        
        self.supportsubset = None
        
    def _support_subset_estimation(self, sample, target, clf, prop=1, n_min=0):


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
        samp_prop = [np.max([n_min, prop * nsv_class[i]]) for i in nsv_class]
        
        # Definición de subconjuntos positivo y negativo
        pos_values = np.where(decision_function_values>0)[0]
        neg_values = np.where(decision_function_values<0)[0]
        x_pos = pos_values[(decision_function_values[pos_values]).argsort().argsort()<samp_prop[1]]
        x_neg = neg_values[((-1)*decision_function_values[neg_values]).argsort().argsort()<samp_prop[-1]]
        
        x_pos_nosv = [ss_pos for ss_pos in x_pos if ss_pos not in alphas_index]
        x_neg_nosv = [ss_neg for ss_neg in x_neg if ss_neg not in alphas_index]
        support_subset = np.hstack([valid_list for valid_list in [x_pos_nosv, x_neg_nosv, alphas_index] if len(valid_list) > 0])
        
        return support_subset.astype(int)
    
    def _is_param_grid(self):
        """Private function for checking param_grid format. """
        
        search_best = any([isinstance(i, list) for i in self.grid_params.values()])

        return search_best
    
    def fit(self, data_train, target):
        if self._is_param_grid():
            gridsearch.fit(self, data_train, target)
            clf = self.best_estimator_
        else:   
            clf = self.method(**self.params)
            clf.fit(data_train, target)
            
        self.supportsubset = self._support_subset_estimation(data_train, target, clf, prop=1, n_min=0)