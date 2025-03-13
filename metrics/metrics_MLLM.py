import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import string
import re
from sklearn.metrics import log_loss

def get_prob_logits(logits_list_1, logits_list_2, logits_list_3):
    max_seq_len_1 = max([logits.shape[1] for logits in logits_list_1])
    max_seq_len_2 = max([logits.shape[1] for logits in logits_list_2])
    max_seq_len_3 = max([logits.shape[1] for logits in logits_list_3])
    probs_list_1, probs_list_2, probs_list_3 = [], [], []
    for logits in logits_list_1:
        probs = F.softmax(logits, dim=-1)
        padding_size = max_seq_len_1 - probs.shape[1]
        padded_probs = F.pad(probs, (0, 0, 0, padding_size))  # 填充第二维
        probs_list_1.append(padded_probs)

    for logits in logits_list_2:
        probs = F.softmax(logits, dim=-1)
        padding_size = max_seq_len_2 - probs.shape[1]
        padded_probs = F.pad(probs, (0, 0, 0, padding_size))  # 填充第二维
        probs_list_2.append(padded_probs)

    for logits in logits_list_3:
        probs = F.softmax(logits, dim=-1)
        padding_size = max_seq_len_3 - probs.shape[1]
        padded_probs = F.pad(probs, (0, 0, 0, padding_size))  # 填充第二维
        probs_list_3.append(padded_probs)

    return probs_list_1, probs_list_2, probs_list_3

def get_token_embeddings(embeddings_list_1, embeddings_list_2, embeddings_list_3):
    max_seq_len = max(
        max([embeddings.shape[1] for embeddings in embeddings_list_1]),
        max([embeddings.shape[1] for embeddings in embeddings_list_2]),
        max([embeddings.shape[1] for embeddings in embeddings_list_3])
    )

    def pad_embeddings(embeddings_list):
        padded_list = []
        for embeddings in embeddings_list:
            seq_padding = max_seq_len - embeddings.shape[1]
            padded_embeddings = F.pad(embeddings, (0, 0, 0, seq_padding))
            padded_list.append(padded_embeddings)
        return padded_list

    embd_1 = pad_embeddings(embeddings_list_1)
    embd_2 = pad_embeddings(embeddings_list_2)
    embd_3 = pad_embeddings(embeddings_list_3)

    return embd_1, embd_2, embd_3

def compute_variance(embd):

    logits_stack = torch.stack(embd, dim=0)  # (num_tta, batch_size, seq_len, vocab_size)
    logits_variance = torch.var(logits_stack, dim=0, unbiased=True)  # 按照 TTA 维度计算方差
    return logits_variance

def compute_covariance(embd_1, embd_2):
    stack_1 = torch.stack(embd_1, dim=0)  # (num_tta, batch_size, seq_len, embedding_dim)
    stack_2 = torch.stack(embd_2, dim=0)  # (num_tta, batch_size, seq_len, embedding_dim)

    num_tta, batch_size, seq_len, embedding_dim = stack_1.shape
    covariance_result = torch.zeros(batch_size, seq_len, embedding_dim, device=stack_1.device)

    # 计算协方差
    for b in range(batch_size):
        for s in range(seq_len):
            embd_1_token = stack_1[:, b, s, :]  # 取出第一个 embeddings 的第 s 个 token，形状为 (num_tta, embedding_dim)
            embd_2_token = stack_2[:, b, s, :]  # 取出第二个 embeddings 的第 s 个 token，形状为 (num_tta, embedding_dim)

            # 计算均值
            mean_embd_1 = embd_1_token.mean(dim=0)  # 计算第一个 embeddings 的均值，形状为 (embedding_dim,)
            mean_embd_2 = embd_2_token.mean(dim=0)  # 计算第二个 embeddings 的均值，形状为 (embedding_dim,)

            # 中心化
            centered_embd_1 = embd_1_token - mean_embd_1  # 第一个 embeddings 中心化
            centered_embd_2 = embd_2_token - mean_embd_2  # 第二个 embeddings 中心化

            # 对每个embedding_dim维度计算协方差
            for e in range(embedding_dim):
                # 计算协方差：针对num_tta维度进行协方差计算
                cov = (centered_embd_1[:, e] @ centered_embd_2[:, e]) / (num_tta - 1)  # 协方差公式
                covariance_result[b, s, e] = cov  # 存储计算结果

    return covariance_result.float().detach().cpu().numpy()

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

def compute_tta_metrics(logits_list_1,logits_list_2,logits_list_3, input_ids
                        ,embde_1,embde_2,embde_3):
    probs_list_1,probs_list_2,probs_list_3 = (
        get_prob_logits(logits_list_1,logits_list_2,logits_list_3))
    embd_1,embd_2,embd_3 = get_token_embeddings(embde_1,embde_2,embde_3)

    metric_df = pd.DataFrame()
    i = 1
    for probs_list,embd in zip([probs_list_1,probs_list_2,probs_list_3],
                               [embd_1,embd_2,embd_3]):
      probs_mean = torch.mean(torch.stack(probs_list), dim=0)  # (batch_size, seq_len, vocab_size)

      predictive_entropy = compute_entropy(probs_mean)  # (batch_size, seq_len)
      metric_df['Predictive_Etropy_{}'.format(i)] = [predictive_entropy.mean().item()]  # 求平均

      conditional_entropies = torch.stack([
        compute_entropy(probs) for probs in probs_list
      ])  # (num_tta, batch_size, seq_len)
      avg_conditional_entropy = conditional_entropies.mean(dim=0)  # (batch_size, seq_len)
      metric_df['Conditional_Etropy_{}'.format(i)] = [avg_conditional_entropy.mean().item()]  # 求平均

      variance = compute_variance(embd)
      metric_df['Variance_{}'.format(i)] = [(variance.float().detach().cpu().numpy())*10e3]

      i += 1

    metric_df['Co_Variance_12'] = [compute_covariance(embd_1, embd_2)*10e3]
    return metric_df



def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if np.sum(in_bin) == 0:
            continue
        accuracy = np.mean(y_true[in_bin] == np.argmax(y_prob[in_bin], axis=1))
        confidence = np.mean(np.max(y_prob[in_bin], axis=1))
        ece += np.abs(confidence - accuracy) * np.sum(in_bin) / len(y_true)
    return ece

def compute_nll(y_true, y_prob):
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
    stack_t = np.stack(y_true, axis=0)
    stack_p = np.stack(y_prob, axis=0)
    stack_var = np.stack(variance, axis=0)
    y_true,y_prob,variance = stack_t, stack_p, stack_var
    variance = np.linalg.norm(variance, axis=1)

    percentiles = np.percentile(variance,
    np.linspace(0, 100, num_percentiles + 1))

    results = []
    ece = 0.0

    for i in range(num_percentiles):
        indices = np.where((variance > percentiles[i]) & (variance < percentiles[i + 1]))[0]

        if len(indices) > 0:
            acc = np.mean(np.argmax(y_true[indices], axis=1) == np.argmax(y_prob[indices], axis=1))
            results.append(acc)
            avg_confidence = np.mean(sigmoid(np.log(variance[indices])))
            ece += np.abs(acc - avg_confidence) * len(indices) / len(y_true)
        else:
            ece += 0

    return ece

#等会移到MUPM.py里
Re = compute_variance_grouped_percentiles(R, MV, V)
Re1 = compute_variance_grouped_percentiles(RI, MVI, VI)
Re2 = compute_variance_grouped_percentiles(RT, MVT, VT)
print((Re+Re1+Re2)/3)