from collections import Counter
from joblib import Parallel, delayed
import statistics
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_random_state
from sklearn.utils.fixes import _joblib_parallel_args


class BoostedSupportSubset():
    def __init__(self, 
                 method=SVC, 
                 params={'C': 1, 'kernel': 'linear'}, 
                 sample_size=None, 
                 k=10,
                 max_features='auto', 
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
        self.n_clusters = k
        self.max_features = max_features
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
        
        x_pos_nosv = [ss_pos for ss_pos in x_pos if ss_pos not in alphas_index]
        x_neg_nosv = [ss_neg for ss_neg in x_neg if ss_neg not in alphas_index]
        support_subset = np.hstack([valid_list for valid_list in [x_pos_nosv, x_neg_nosv, alphas_index] if len(valid_list) > 0])
        
        # print(x_pos_nosv)
        # print(x_neg_nosv)
        # print(alphas_index)
        return support_subset.astype(int)
    
    
    def _kmeans_sample(self, data_train, target):
    
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(data_train)
        
        partition_idx = []
        
        for i in range(1000):
            
            sample_seed = self.random_state + i
            
            random_instance = check_random_state(sample_seed)
            
            sample_cluster = random_instance.choice(range(self.n_clusters), size=2, replace=True)
            
            mask = np.isin(kmeans.labels_, sample_cluster)
            
            classes_condition = len(np.unique(target[mask]))<2
            
            if not classes_condition:
                partition_idx.append(np.where(mask)[0])
                feature_space_condition = len({x for l in partition_idx for x in l})
                if not feature_space_condition < len(data_train):
                    break
                
        return partition_idx
    
    def _parallel_build_learners(self, learning_set, target, active_idx):
        excluded_idx = []
        idx_learner = 0
        learners = []
        
        while len(active_idx) >= self.num_samples:
            
            # print(idx_learner)
            # print('active_idx: ', active_idx)
            # print('excluded_idx: ', sorted(excluded_idx))
            # print('\n')
            
            for j in range(1000):
                sample_idx, features_idx, sample_seed = self._generate_sample_indexes(active_idx, idx_learner, j)
                if len(np.unique(target[sample_idx]))>1:
                        break
                # else:
                #     print('No valid sample in 1000 iters')
                
            fail_condition = self._fail_condition(sample_idx, target, idx_learner)
            
            if fail_condition:
                break
            else:
                x = learning_set[np.ix_(sample_idx, features_idx)]
                y =  target[sample_idx]
                learner = self._fit_learner(x, y, sample_idx, features_idx, active_idx, excluded_idx, sample_seed)
                excluded_idx.extend(learner['data']['support_subset_indexes'])
                learners.append(learner)
                idx_learner += 1
                active_idx = self._active_set(max(active_idx), excluded_idx)
     
        return learners
    
    
    def fit(self, data_train, target, n_jobs):
        
        if self.sample_size is not None:
            self.num_samples = self.sample_size
        else:
            self.num_samples = int(data_train.shape[0]*self.prop_sample)
            
        self.n_features_ = data_train.shape[1]
        
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
        
        region_active_idx = self._kmeans_sample(data_train, target)

        self.learners = Parallel(n_jobs=n_jobs, **_joblib_parallel_args(prefer='threads'))(
        delayed(self._parallel_build_learners)(data_train, target, active_idx)
        for active_idx in region_active_idx)

    
    def _active_set(self, max_idx, excluded_idx):
        
        if self.support_subset:
            active_idx = [i for i in range(max_idx) if i not in excluded_idx]
        else:
            active_idx = [i for i in range(max_idx)]
    
        return active_idx
    
    def _generate_sample_indexes(self, active_idx, idx_learner, j):
        
        sample_seed = self.random_state + idx_learner + j
        
        random_instance = check_random_state(sample_seed)
        sample_idx = random_instance.choice(active_idx, size=self.num_samples, replace=False)
        
        if self.max_features == self.n_features_:
            features_idx = range(self.n_features_)
        else:
            features_idx = random_instance.choice(range(self.n_features_), self.max_features_, replace=False)

        return sample_idx, np.sort(features_idx), sample_seed
    
    
    def _fail_condition(self, sample_idx, target, idx_learner):
        
        if self.n_learners != None:
            n_learners_condition = idx_learner > (self.n_learners - 1)
        else:
            n_learners_condition = False
        classes_condition = len(np.unique(target[sample_idx]))<2

        fail_condition = classes_condition | n_learners_condition
        
        # print('classes_condition ', classes_condition)
        # print('n_learners ', n_learners)
    
        return fail_condition
        
        
    def _fit_learner(self, x, y, sample_idx, features_idx, active_idx, excluded_idx, sample_seed):
        
        learner = self.method(**self.params, random_state=self.random_state)
        
        learner.fit(x, y)
        
        sample_ss_idx = self._support_subset_estimation(x, y, learner, prop=1, n_min=0)
        # print(sample_ss_idx)
        ss_idx = sample_idx[sample_ss_idx]
        # print('sample_idx: ', sorted(list(sample_idx)))
        # print('ss_indx: ', sorted(list(ss_idx)))
        # print('\n')
        return {
        'data': {
            'seed': sample_seed,
            'train_indexes': sample_idx,
            'features_indexes': features_idx,
            'support_subset_indexes': ss_idx,
            'active_indexes': np.array(active_idx),
            'excluded_indexes': np.array(excluded_idx),
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
            preds = learner['learner'].predict(X[:, learner['data']['features_indexes']])
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
