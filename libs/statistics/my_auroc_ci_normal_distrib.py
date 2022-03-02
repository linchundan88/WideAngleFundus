'''
 created by jji, on 2022/2/22

 parametric method: normal distribution assumption, get_ci_auc_normal_distrib
 jji modified based on https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c

'''

from sklearn.metrics import roc_auc_score
from math import sqrt



def get_ci_auc_normal_distrib(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)



#sided: two, lower, upper
