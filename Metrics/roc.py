"""
References:
    1. https://www.zhihu.com/question/40748327
    2. http://nphard.me/2017/08/17/roc-auc/
    3. http://alexkong.net/2013/06/introduction-to-auc-and-roc/
"""

import numpy as np
import sklearn.metrics as metrics


def binary_clf_curve(y_true, y_score, pos_label=None):
    if pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores
    desc_score_indices = np.argsort(y_score)[::-1]
    thresholds = y_score[desc_score_indices]

    fps = []
    tps = []
    for threshold in thresholds:
        # 大于等于阈值判定为 1 (正类)，否则为 0 (负类)
        y_predict = [1 if i >= threshold else 0 for i in y_score]
        # 预测值是否等于真实值
        result = [i == j for i, j in zip(y_true, y_predict)]
        # 预测值是否为 1 (正类)
        positive = [i == 1 for i in y_predict]

        # 预测为正类且预测错误
        fp = [(not i) and j for i, j in zip(result, positive)].count(True)
        # 预测为正类且预测正确
        tp = [i and j for i, j in zip(result, positive)].count(True)

        fps.append(fp)
        tps.append(tp)
    fps = np.array(fps)
    tps = np.array(tps)

    return fps, tps, thresholds


def roc_curve(y_true, y_score, pos_label=None):
    fps, tps, thresholds = binary_clf_curve(y_true, y_score, pos_label)

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    return fpr, tpr, thresholds


if __name__ == '__main__':
    y_true = [0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    y_score = [0.31689620142873609, 0.32367439192936548,
               0.42600526758001989, 0.38769987193780364,
               0.3667541015524296, 0.39760831479768338,
               0.42017521636505745, 0.41936155918127238,
               0.33803961944475219, 0.33998332945141224]
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    fpr1, tpr1, thresholds1 = roc_curve(y_true, y_score)
    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_true, y_score,
                                                drop_intermediate=False)
    print('FPR 1: {}'.format(fpr1))
    print('FPR 2: {}'.format(fpr2))
    print()
    print('TPR 1: {}'.format(tpr1))
    print('TPR 2: {}'.format(tpr2))
    print()
    print('Threshold 1: {}'.format(thresholds1))
    print('Threshold 2: {}'.format(thresholds2))
