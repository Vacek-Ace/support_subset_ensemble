import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm 
import contextlib
import joblib

from sklearn.model_selection import ParameterGrid
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def StratifiedKFoldTestPredictions(X, y, clf, k=10, n_jobs=-1):
    skf = StratifiedKFold(n_splits=k)
    preds = np.empty_like(y)
    i = 0
    for train_index, test_index in skf.split(X, y):
        # print(i)
        # print(train_index, test_index)
        try:
            clf.fit(X[train_index], y[train_index], n_jobs=n_jobs)
        except:
            clf.fit(X[train_index], y[train_index])
        preds[test_index] = clf.predict(X[test_index])
        i += 1
    return preds


def confusion_matrix(preds, y, normalize=True):
    confusion_matrix = pd.crosstab(
        preds, y, margins=True, margins_name='total', normalize=normalize)
    confusion_matrix.columns = pd.Index(
        [0, 1, 'total'], dtype='object', name='real')
    confusion_matrix.index = pd.Index(
        [0, 1, 'total'], dtype='object', name='pred')
    return confusion_matrix.round(2)


def scaled_mcc(y_true, y_pred):
    matthews_corrcoef_scaled = (matthews_corrcoef(y_true, y_pred) + 1)/2
    return matthews_corrcoef_scaled


class gridsearch():

    def __init__(self,
                 method,
                 grid_params,
                 scoring=scaled_mcc,
                 cv=10,
                 n_jobs=-1,
                 random_state=1234,
                 kwargs=None,
                 verbose=True):

        self.method = method
        self.grid_params = grid_params
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs
        self.verbose = verbose

    def _train_test_split(self, X, y):

        skf = StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        k_fold_splits = list(skf.split(X, y))
        return k_fold_splits

    def _eval(self, X, y, params):

        scores = []

        if self.kwargs is None:
            clf = self.method(**params)
        else:
            clf = self.method(**params, **self.kwargs)

        for train_index, test_index in self.k_fold_splits:

            clf.fit(X[train_index], y[train_index])

            scores.append(self.scoring(
                y[test_index], clf.predict(X[test_index])))

        return [clf, params, scores, np.mean(scores)]

    def fit(self, X, y):

        self.k_fold_splits = self._train_test_split(X, y)
        
        grid_params =list(ParameterGrid(self.grid_params))

        if self.verbose:
            with tqdm_joblib(tqdm(desc="Searching best hyperparameters", total=len(grid_params))) as progress_bar:
                self.scoring_results_ = Parallel(n_jobs=self.n_jobs, **_joblib_parallel_args(prefer='threads'))(
                    delayed(self._eval)(X, y, params)for params in grid_params)
        else:
            self.scoring_results_ = Parallel(n_jobs=self.n_jobs, **_joblib_parallel_args(prefer='threads'))(
                    delayed(self._eval)(X, y, params)for params in grid_params)

        self.best_index_ = np.argmax([result[3]
                                      for result in self.scoring_results_])

        self.best_estimator_ = self.scoring_results_[self.best_index_][0]
        self.best_params_ = self.scoring_results_[self.best_index_][1]
        self.best_score_ = np.mean(self.scoring_results_[self.best_index_][2]), np.std(
            self.scoring_results_[self.best_index_][2])


class GridSearch_moess():

    def __init__(self,
                 method,
                 grid_params,
                 scoring=scaled_mcc,
                 folds=None,
                 n_jobs=-1,
                 random_state=1234,
                 kwargs=None,
                 verbose=True):

        self.method = method
        self.grid_params = grid_params
        self.scoring = scoring
        self.k_fold_splits = folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs
        self.verbose = verbose


    def _eval(self, X, y, params):

        scores = []

        if self.kwargs is None:
            clf = self.method(**params)
        else:
            clf = self.method(**params, **self.kwargs)

        ii = 0
        for train_index, test_index, ss_index in self.k_fold_splits:
            
            try:
                clf.fit(X[train_index], y[train_index], hot_indexes=ss_index)
            except Exception as e:
                print(params, ii, e)
                raise e

            ii += 1
            scores.append(self.scoring(
                y[test_index], clf.predict(X[test_index])))

        return [clf, params, scores, np.mean(scores)]

    def fit(self, X, y):
        
        grid_params =list(ParameterGrid(self.grid_params))

        if self.verbose:
            with tqdm_joblib(tqdm(desc="Searching best hyperparameters", total=len(grid_params))) as progress_bar:
                self.scoring_results_ = Parallel(n_jobs=self.n_jobs, **_joblib_parallel_args(prefer='threads'))(
                    delayed(self._eval)(X, y, params)for params in grid_params)
        else:
            self.scoring_results_ = Parallel(n_jobs=self.n_jobs, **_joblib_parallel_args(prefer='threads'))(
                    delayed(self._eval)(X, y, params)for params in grid_params)

        self.best_index_ = np.argmax([result[3]
                                      for result in self.scoring_results_])

        self.best_estimator_ = self.scoring_results_[self.best_index_][0]
        self.best_params_ = self.scoring_results_[self.best_index_][1]
        self.best_score_ = np.mean(self.scoring_results_[self.best_index_][2]), np.std(
            self.scoring_results_[self.best_index_][2])
