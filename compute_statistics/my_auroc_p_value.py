from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from libs.statistics.my_auroc_ci_normal_distrib import get_ci_auc_normal_distrib
from libs.statistics.my_auroc_ci_bootstrap import get_ci_auc_bootstrap
from libs.statistics.my_auroc_ci_delong import get_ci_auc_delong



csv_test = Path(__file__).parent.absolute().joinpath('2021_4_20_results', 'test.csv')
list_labels = pd.read_csv(csv_test)['labels'].tolist()

path_pkl_multi_labels = Path(__file__).parent.absolute().joinpath('2021_4_20_results', 'test', 'probs.pkl')
with path_pkl_multi_labels.open(mode='rb') as f:
    probs_multi_labels = pickle.load(f)


for label_no in [0, 1, 2, 3]:
# for label_no in [0]:
    y_true = []
    for labels in list_labels:
        y_true.append(int(labels.split('_')[label_no]))
    y_true = np.array(y_true)

    y_score_ml = probs_multi_labels[0][:, label_no]
    auroc_ml = roc_auc_score(y_true, y_score_ml)
    #different methods to calcuate confidence interval make no significant significantdifference
    lower_ml_1, upper_ml_1 = get_ci_auc_normal_distrib(y_true, y_score_ml)
    se_ml_1 = (upper_ml_1 - lower_ml_1) / 3.92
    lower_ml_2, upper_ml_2 = get_ci_auc_bootstrap(y_true, y_score_ml, seed=1111)
    se_ml_2 = (upper_ml_2 - lower_ml_2) / 3.92
    lower_ml_3, upper_ml_3 = get_ci_auc_delong(y_true, y_score_ml)
    se_ml_3 = (upper_ml_3 - lower_ml_3) / 3.92

    path_pkl_indepedent_classifier = Path(__file__).parent.absolute().joinpath('2022_02_07_results', 'test', str(label_no), 'probs.pkl')
    with path_pkl_indepedent_classifier.open(mode='rb') as f:
        y_score_ic = pickle.load(f)[0][:,0]
        auroc_ic = roc_auc_score(y_true, y_score_ic)

        lower_ic_1, upper_ic_1 = get_ci_auc_normal_distrib(y_true, y_score_ic)
        se_ic_1 = (upper_ic_1 - lower_ic_1) / 3.92
        lower_ic_2, upper_ic_2 = get_ci_auc_bootstrap(y_true, y_score_ic, seed=1111)
        se_ic_2 = (upper_ic_2 - lower_ic_2) / 3.92
        lower_ic_3, upper_ic_3 = get_ci_auc_delong(y_true, y_score_ic)
        se_ic_3 = (upper_ic_3 - lower_ic_3) / 3.92





    #AUC difference, standard error, Z-statistics, p_value
    # z score for a sample mean. difference / SE
    # SE = (confidence interval (upper bound - lower bound)/3.92
    difference = (auroc_ml - auroc_ic)
    se = (upper_ml_1 - lower_ml_1) / 3.92
    z = difference / se

    #class no 0 z:-0.52, p value 0.6
    #class no 1  z:-3.6
    #class no 2 z:-1.45, p value 0.14
    #class no 3 z:4.69,  p value < .00001.
    #https://www.socscistatistics.com/pvalues/normaldistribution.aspx

    print(f'z score:{z}')


'''
#results
z score:-0.4936344925934139,  p value:.6241
z score:-1.4158100924755104,  p value:.15706
z score:-1.2824754737722528,  p value:.199843
z score:19.40081063918061,    p value:< .00001.
'''



