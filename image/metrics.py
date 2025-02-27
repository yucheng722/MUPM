import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import numpy as np
import string
import re

def normalize(s):
    # 移除空格和标点符号
    return ''.join(c for c in s if c not in string.whitespace + string.punctuation)


def exact_match_score(y_true, y_pred):
    """
    计算 Exact Match (EM) 指标。

    参数：
    - y_true: list of str, 标准答案
    - y_pred: list of str, 模型预测答案

    返回：
    - em_score: float, EM 指标值
    """
    matches = [int(t == p) for t, p in zip(y_true, y_pred)]
    em_score = sum(matches) / len(y_true)
    return em_score


def precision_recall_f1_score(y_true, y_pred):
    """
    计算 Precision, Recall 和 F1 Score。

    参数：
    - y_true: list of list, 每个问题的正确答案（支持多选）
    - y_pred: list of list, 每个问题的模型预测答案（支持多选）

    返回：
    - metrics: dict, 包含 Precision, Recall 和 F1 的字典
    """
    # 将多选答案展开为平面列表
    y_true_flat = []
    y_pred_flat = []
    for t, p in zip(y_true, y_pred):
        y_true_flat.extend(t)
        y_pred_flat.extend(p)

    precision = precision_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='micro', zero_division=0)

    return {'Precision': precision, 'Recall': recall, 'F1': f1}


def top_k_accuracy(y_true, y_pred, k=1):
    """
    计算 Top-k Accuracy。

    参数：
    - y_true: list of str, 每个问题的正确答案
    - y_pred: list of list of str, 每个问题的模型预测答案的候选列表
    - k: int, 考虑的候选答案数量

    返回：
    - top_k_score: float, Top-k 准确率
    """
    correct = 0
    for t, preds in zip(y_true, y_pred):
        if t in preds[:k]:
            correct += 1
    top_k_score = correct / len(y_true)
    return top_k_score

def has_repeating_substring(n, p, length):
    normalized_p = normalize(p)
    for i in range(len(n) - length + 1):
        substring = n[i:i+length+1]
        if substring in normalized_p:
           return True
    #return any(match in normalized_p for match in matches)

dir = r"D:\NewPythonProject\Yucheng\results_image_only.xlsx"
dir = r"D:\NewPythonProject\Yucheng\results_2.xlsx"
df = pd.read_excel(dir)
diseases = [
    "Hypertensive", "Ischaemic",
    "arrhythmias", "Conduction disorders",
    "Complications", "Valve",
    "Heart failure", "Pulmonary",
    "Chronic rheumatic", "Pericarditis",
    "Cardiomyopathy"
]

count = 0
total = 0
for i in np.arange(50):
  Ans = df['answer']
  Pre = df['pre_{}'.format(i)]
  for a,p in zip(Ans,Pre):
    #for disease in diseases:
     # n = a.split('.')[1]
      n = normalize(a)
      total += 1
      if has_repeating_substring(n, str(p), length=3):
          count+=1

rate = count/total
print(rate)

x1 = np.array([9.625, 2.734375, -1.34375, 9.5625, 5.78125, -2.9375, -5.59375, 5.1875, 15.25, 7.0625, 2.03125, 4.34375, 1.3203125, 16.375, 0.953125, 1.1953125, -7.96875, 10.5, 0.5703125, 8.75, 6.28125, 8.0625, 5.3125, 8.25, 3.32])
x2 = np.array([11.0625, 3.03125, 5.5625, -5.25, 1.4140625])
x3 = np.array([[7.34375, -7.4375, -2.109375, -1.0859375, 0.73828125, 1.0546875, 7.03125, 18.125, 22.875, 1.0625, 6.6875, -6.375, -3.875, 3.109375, 0.1591796875, 8.75, 0.79296875, 10.125, -7.75, 28.5, 1.4140625, -8.9375, -0.765625, 4.25, 7.5625, -5.3125, 2.34375, 0.2412109375, 1.5625, 6.375, 2.296875, 6.0625, 1.0546875, 1.7734375, 4.5, 5.65625, 2.140625, 4.46875, 7.5625, 2.109375, -0.369140625, 7.0625, 10.5625, 7.0, 17.125, 10.4375, 9.0, 6.125, 8.0, 6.09375, 11.125, 13.9375, 14.8125, 2.78125, 6.40625, 4.28125, 8.0, 8.0, 1.0625, 10.6875, 12.3125, 5.375, 10.625, 3.734375, 7.90625, 2.328125, 8.5625, 6.8125, 4.5625, 6.1875, 8.1875, 3.859375, 5.78125, 9.75, -1.1328125, 0.7109375, -0.08935546875, 6.53125, 8.3125, 4.59375, 7.65625, 7.40625, 1.3671875, 5.03125, 7.96875, 7.1875, 9.3125, 5.9375, -2.84375, 10.3125, 7.6875, 4.71875, 7.25, 17.875, 17.25, 1.640625, -2.140625, 4.5625, 13.9375, 5.84375, 6.46875, 17.875, 12.375, 9.5625, 6.71875, 8.375, 1.5, 2.34375, 4.0625, 5.4375, 3.734375, 4.1875, 9.4375, 6.875, 6.0, 10.375, 7.5, 10.6875, 10.25, 6.5, -1.15625, -6.65625, 5.71875, 4.4375, 2.40625, 7.96875]])
x4 = np.array([3.890625, 11.625, -0.498046875, 4.15625, 8.6875, -0.8046875, 12.75, 6.3125, 8.3125, 1.6171875, 10.0625, -2.75, 7.21875, 2.703125, 6.96875, 9.8125, 5.78125, 6.0, 2.71875, 3.25, 3.59375, 4.34375, -10.125, -6.34375, 0.19140625, -0.466796875, 0.134765625, 5.1875, -10.5, 5.0625, 11.0, 3.703125, 0.60546875, -3.828125, 3.5625, 8.625, -1.8515625, -9.8125, -11.0625, -2.8125, 4.125, -9.5625, -3.0625, 8.9375, 2.609375, 8.6875, 3.65625, -0.57421875, -1.6796875, 5.71875, 5.96875, 10.625, 0.5625, 13.5625, 7.59375, 11.9375, -0.59375, 11.25, 8.5625, 10.0, 10.25, 7.46875, 3.75, 5.5, 8.1875, 4.6875, -6.28125, 4.09375, 0.6484375, 1.078125, 5.53125, 17.375, 6.40625, -1.6640625, 3.953125, 5.59375, 6.125, 4.71875, 16.25, 0.9296875, 2.90625, 1.09375, 0.06201171875, 6.4375, 3.15625, 12.3125, 13.875, 2.828125, 3.265625, 9.6875, 10.5, 2.75, 12.0625, 9.0, 0.412109375, 2.6875, 2.234375, 8.9375, 1.8046875, 19.75, 3.65625, 16.125, 3.90625, 5.0625, 3.109375, 15.1875, 5.0625, 8.5625, 5.9375, 10.0, 16.0, 5.125, 18.125, 8.1875, 7.9375, 5.84375, 7.0, 7.96875, 6.5625, 17.25, 5.125, 11.375, 4.875, 2.40625, 3.203125, 10.1875])

from sklearn.calibration import calibration_curve
import numpy as np
from sklearn.metrics import log_loss
def compute_ece(y_true, y_prob, n_bins=10):
    # 将预测概率划分为 n_bins 个区间
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        # 获取当前区间内的样本索引
        in_bin = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if np.sum(in_bin) == 0:
            continue
        # 计算当前区间内的准确率和平均预测概率
        accuracy = np.mean(y_true[in_bin] == np.argmax(y_prob[in_bin], axis=1))
        confidence = np.mean(np.max(y_prob[in_bin], axis=1))
        # 计算当前区间的加权误差
        ece += np.abs(confidence - accuracy) * np.sum(in_bin) / len(y_true)
    return ece

def compute_nll(y_true, y_prob):
    # 计算负对数似然
    return log_loss(y_true, y_prob)

def compute_ece_for_multiple_samples(y_true_list, y_prob_list, n_bins=10):
    ece_list = []
    for y_true, y_prob in zip(y_true_list, y_prob_list):
        ece = compute_ece(y_true, y_prob, n_bins)
        ece_list.append(ece)
    return np.mean(ece_list)

def compute_nll_for_multiple_samples(y_true_list, y_prob_list):
    nll_list = []
    for y_true, y_prob in zip(y_true_list, y_prob_list):
        nll = compute_nll(y_true, y_prob)
        nll_list.append(nll)
    return np.mean(nll_list)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_variance_grouped_percentiles(y_true, y_prob, variance, num_percentiles=10):
    # 将 y_true 和 y_prob 转换为 NumPy 数组
    stack_t = np.stack(y_true, axis=0)
    stack_p = np.stack(y_prob, axis=0)
    stack_var = np.stack(variance, axis=0)
    y_true,y_prob,variance = stack_t, stack_p, stack_var
    #y_true = np.array(y_true)
    #y_prob = np.array(y_prob)
    variance = np.linalg.norm(variance, axis=1)

    # 计算方差的百分位数
    percentiles = np.percentile(variance,
    np.linspace(0, 100, num_percentiles + 1))

    # 初始化结果列表
    results = []
    ece = 0.0

    # 遍历每个百分位数区间
    for i in range(num_percentiles):
        # 获取当前区间的索引
        indices = np.where((variance > percentiles[i]) & (variance < percentiles[i + 1]))[0]

        if len(indices) > 0:
            # 计算当前区间的准确度
            acc = np.mean(np.argmax(y_true[indices], axis=1) == np.argmax(y_prob[indices], axis=1))
            results.append(acc)
            #print(acc)
            # 计算confidence
            avg_confidence = np.mean(sigmoid(np.log(variance[indices])))
            #print(avg_confidence)
            ece += np.abs(acc - avg_confidence) * len(indices) / len(y_true)
            #print(ece)
        else:
            #results.append(np.nan)
            ece += 0

    return ece

Re = compute_variance_grouped_percentiles(R, MV, V)
Re1 = compute_variance_grouped_percentiles(RI, MVI, VI)
Re2 = compute_variance_grouped_percentiles(RT, MVT, VT)
print((Re+Re1+Re2)/3)