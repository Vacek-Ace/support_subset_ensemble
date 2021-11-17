import numpy as np
import time
from sklearn import metrics
from src.hot_libsvm import SVC
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from random import seed
from fast_svm_predict import predict_fn


def f(x):
    return np.int(x)


f2 = np.vectorize(f)


def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def rbf_kernel(x1, x2):
    return pairwise.rbf_kernel(x1.reshape(-1, 1), x2.reshape(-1, 1)).reshape(-1)[0]


def bivariate_normal(mu, sigma, n, category):
    X = np.random.normal(mu, sigma, size=[n, 2])
    y = category * np.ones(n)

    return X, y


class Timer:

    def __init__(self, msg='', no_msg=False):
        self.msg = msg

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        if self.msg == '':
            print(f'[{self.msg}] Elapsed time: {self.interval.seconds}s')


def initial_sampling(df, clf, prop=1):

    x = df.copy()

    prop = prop + 1

    # Identificación de puntos influyentes
    x['clf_ini_values'] = clf.decision_function(df.drop(columns='y').values)
    # fast_predict = predict_fn(clf, output_type='decision_function') # output_type can be class|proba|decision_function.

    # x['clf_ini_values'] = fast_predict(df.drop(columns='y').values)

    alphas_index = clf.support_

    nsv_class = x.iloc[clf.support_]['y'].value_counts().sort_index()

    samp_pos = np.max([nsv_class.iloc[1] + 50, prop * nsv_class.iloc[1]])
    samp_neg = np.max([nsv_class.iloc[0] + 50, prop * nsv_class.iloc[0]])

    x_positive_sorted = x[x['y'] > 0].sort_values(by='clf_ini_values', ascending=True)

    x_negative_sorted = x[x['y'] < 0].sort_values(by='clf_ini_values', ascending=False)

    x_positive = x_positive_sorted.iloc[
                 nsv_class.iloc[1]:samp_pos].index  # .append(x_positive_sorted.tail(samp_pos)).index

    x_negative = x_negative_sorted.iloc[
                 nsv_class.iloc[0]:samp_neg].index  # .append(x_negative_sorted.tail(samp_neg)).index

    lb_sv = x['clf_ini_values'].iloc[alphas_index].min()

    ub_sv = x['clf_ini_values'].iloc[alphas_index].max()

    return x_positive, x_negative, lb_sv, ub_sv, nsv_class


def final_sampling_iter(df, clf, n_sample, lb_sv, ub_sv, rng_seed=42):
    df_new = df.copy()

    n_sample = min(n_sample, len(df_new))

    df_new['decision_value'] = clf.decision_function(df_new.drop(columns='y').values)
    
    # fast_predict = predict_fn(clf, output_type='decision_function') 

    # df_new['decision_value'] = fast_predict(df_new.drop(columns='y').values)
    
    df_new['missclassified'] = df_new['decision_value'] * df_new['y']

    mask_bounded_new = (df_new['decision_value'] <= ub_sv) & (df_new['decision_value'] >= lb_sv)

    bounded_new = df_new.loc[mask_bounded_new].index.astype(np.int32)

    mask_miss = df_new['missclassified'] < 0 & ~mask_bounded_new

    if (np.sum(mask_miss) > 0) and (np.sum(mask_bounded_new) > 0):

        samples = df_new.sample(n=n_sample, random_state=rng_seed).index.astype(np.int32)

    elif (np.sum(mask_miss) == 0) and (np.sum(mask_bounded_new) > 0):

        samples = bounded_new
    else:

        samples = []

    return samples


def final_sampling(df, new_points, clf, n_sample, lb_sv, ub_sv, rng_seed=42):

    df_new = df.loc[new_points].copy()
    n_sample = min(n_sample, len(df_new))

    df_new['decision_value'] = clf.decision_function(df_new.drop(columns='y').values)
    df_new['missclassified'] = df_new['decision_value'] * df_new['y']

    mask_bounded_new = (df_new['decision_value'] <= ub_sv) & (df_new['decision_value'] >= lb_sv)

    bounded_new = df_new.loc[mask_bounded_new].index.astype(np.int32)

    mask_miss = df_new['missclassified'] < 0 & ~mask_bounded_new

    if (np.sum(mask_miss) > 0) and (np.sum(mask_bounded_new) > 0):

        samples = df_new.sample(n=n_sample, random_state=rng_seed).index.astype(np.int32)

    elif (np.sum(mask_miss) == 0) and (np.sum(mask_bounded_new) > 0):

        samples = bounded_new
    else:

        samples = []

    return samples


def svm_metrics(df, clf, verbose=False):
    preds = clf.predict(df.drop(columns='y').values)
    f1 = metrics.f1_score(df['y'].values, preds)
    precision = metrics.precision_score(df['y'].values, preds)
    accuracy = metrics.accuracy_score(df['y'].values, preds)
    recall = metrics.recall_score(df['y'].values, preds)
    if verbose:
        print('Precision: ', round(precision, 4))
        print('Accuracy: ', round(accuracy, 4))
        print('Recall: ', round(recall, 4))
        print('F1: ', round(f1, 4))
    return f1, precision, accuracy, recall

def dict_metrics(df, clf, verbose=False):
    
    if df is not None:
    
        f1, precision, accuracy, recall = svm_metrics(df, clf, verbose=False)
        
        metrics = {'f1': f1,
                'precision': precision,
                'accuracy': accuracy,
                'recall': recall}
    else:
        metrics = {}    
    return metrics


def svm_plot(df, clf, title, xlim, ylim, extra_points=None):
    plt.figure(figsize=(8, 8))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plot_decision_regions(df.drop(columns='y').values, f2(df['y'].values), clf=clf, legend=2,
                          X_highlight=clf.support_vectors_, scatter_highlight_kwargs={'c': 'black', 's': 100})

    if extra_points is not None:
        plt.plot(df.X1.loc[extra_points], df.X2.loc[extra_points], 'o', marker='^', c='red', alpha=0.5, markersize=10)

    plt.show()
