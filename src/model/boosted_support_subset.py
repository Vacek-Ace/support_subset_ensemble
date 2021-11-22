from collections import Counter
from joblib import Parallel, delayed
import statistics
import numpy as np
from sklearn.svm import SVC
from sklearn.utils.validation import check_random_state
from sklearn.utils.fixes import _joblib_parallel_args


class BoostedSupportSubset():
    def __init__(self, 
                 method=SVC, 
                 params={'C': 1, 'kernel': 'linear'}, 
                 sample_size=None, 
                #  max_features='auto', 
                 support_subset = True, 
                 prop_sample=0.1, 
                 n_learners=None,
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
        self.sample_size = sample_size
        # self.max_features = max_features
        self.support_subset = support_subset
        self.prop_sample = prop_sample
        self.n_learners = n_learners
        self.random_state = random_state
        self.learners = None

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
        
        support_subset = np.hstack([x_pos, x_neg, alphas_index])
            
        return support_subset 
    
    
    def fit(self, data_train, target):
        
        
        if self.sample_size is not None:
            self.num_samples = self.sample_size
        else:
            self.num_samples = int(data_train.shape[0]*self.prop_sample)

        excluded_idx = []
        idx_learner = 0
        learners = []
        active_idx = [i for i in range(len(data_train))]
        
        while len(active_idx) >= self.num_samples:
            
            print(idx_learner)
            print('active_idx: ', active_idx)
            print('excluded_idx: ', sorted(excluded_idx))
            print('\n')
            
            sample_idx = self._generate_sample_indexes(active_idx, idx_learner)
            fail_condition = self._fail_condition(sample_idx, target, idx_learner)
            
            if fail_condition:
                break
            else:
                x = data_train[sample_idx]
                y =  target[sample_idx]
                learner = self._fit_learner(x, y, sample_idx)
                excluded_idx.extend(learner['data']['support_subset_indexes'])
                learners.append(learner)
                idx_learner += 1
                active_idx = self._active_set(max(active_idx), excluded_idx)
        self.learners = learners
    
    def _active_set(self, max_idx, excluded_idx):
        
        if self.support_subset:
            active_idx = [i for i in range(max_idx) if i not in excluded_idx]
        else:
            active_idx = [i for i in range(max_idx)]
    
        return active_idx
    
    def _generate_sample_indexes(self, active_idx, idx_learner):
        
        random_instance = check_random_state(self.random_state + idx_learner)
        sample_idx = random_instance.choice(active_idx, size=self.num_samples, replace=False)

        return sample_idx
    
    
    def _fail_condition(self, sample_idx, target, idx_learner):
        
        n_learners = idx_learner > (self.n_learners - 2)
        n_classes = Counter(target[sample_idx])
        classes_condition = (sum([n_classes[i] > 2 for i in n_classes]) < 2)

        fail_condition = classes_condition | n_learners
    
        return fail_condition
        
        
        
    def _fit_learner(self, x, y, sample_idx):
        
        learner = self.method(**self.params, random_state=self.random_state)
        
        learner.fit(x, y)
        
        sample_ss_idx = self._support_subset_estimation(x, y, learner, prop=1, n_min=0)
        
        ss_idx = sample_idx[sample_ss_idx]
        print('sample_idx: ', sorted(list(sample_idx)))
        print('ss_indx: ', sorted(list(ss_idx)))
        print('\n')
        return {
        'data': {
            'train_indexes': sample_idx,
            'support_subset_indexes': ss_idx
            },
        'learner': learner
        }
        
    

    
    
    def _parallel_predict(self, X, learner):
        """ Private function to parallel prediction.

        Args:
            X (array): Set from which the predictions are required.
            learner ([type]): Single learner.

        Returns:
            [array]: Predictions.
        """
        
        try:
            preds = learner['learner'].predict(X)
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
