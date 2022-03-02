'''
 created by jji, on 2022/2/22
 nonparametric method: get_ci_auc_bootstrap
 jji modified based on https://sites.google.com/site/lisaywtang/tech/python/scikit/auc-conf-interval

'''

import numpy as np
from sklearn.metrics import roc_auc_score


def get_ci_auc_bootstrap(y_true, y_pred, ci=0.95, sided='two', n_bootstraps=1000, seed=1234):
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))

        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    if sided == 'two':
        # 95% c.i.
        # confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        # confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

        confidence_lower = sorted_scores[int((1-ci) / 2 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(1 - (1-ci) / 2 * len(sorted_scores))]

        return confidence_lower, confidence_upper  # [confidence_lower, confidence_upper]

    if sided == 'lower':
        confidence_lower = sorted_scores[int((1 - ci) * len(sorted_scores))]
        return confidence_lower #[confidence_lower, Inf]

    if sided == 'upper':
        confidence_upper = sorted_scores[int(ci * len(sorted_scores))]

        return confidence_upper #[-Inf, confidence_upper]


