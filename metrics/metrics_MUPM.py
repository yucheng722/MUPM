import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

def compute_variance(list):
    stack = np.stack(list, axis=0)
    variance = np.var(stack, axis=0, ddof=1)
    return variance

def compute_covariance(embd_1, embd_2):
    embd_1 = np.stack(embd_1, axis=0)  # (num_tta, 11)
    embd_2 = np.stack(embd_2, axis=0)  # (num_tta, 11)

    num_tta, embedding_dim = embd_1.shape
    covariance_result = np.zeros(embedding_dim)

    # 计算均值
    mean_embd_1 = np.mean(embd_1, axis=0)  # (11,)
    mean_embd_2 = np.mean(embd_2, axis=0)  # (11,)

    # 中心化
    centered_embd_1 = embd_1 - mean_embd_1  # (num_tta, 11)
    centered_embd_2 = embd_2 - mean_embd_2  # (num_tta, 11)

    # 计算协方差（逐元素计算 11 维度的协方差）
    for e in range(embedding_dim):
        cov = np.dot(centered_embd_1[:, e], centered_embd_2[:, e]) / (num_tta - 1)
        covariance_result[e] = cov

    return covariance_result

def variances_analysis(results_dir):
    if results_dir[-1] == 'x':
      df = pd.read_excel(results_dir)
    else:
      df = pd.read_json(results_dir)
    Var_1 = df['Variance_1']
    Var_2 = df['Variance_2']
    Var_3 = df['Variance_3']
    Co = df['Co_Variance_12']
    De, VI, VT, V, C = [], [], [], [], []
    for i in range(len(Var_1)):
      if i == len(Var_1) - 1:
          break
      v1 = np.array(Var_1[i]).reshape(-1)
      v2 = np.array(Var_2[i]).reshape(-1)
      v3 = np.array(Var_3[i]).reshape(-1)
      co = np.array(Co[i]).reshape(-1)
      vv = v3 - v1 - v2
      De.append(vv)
      VI.append(v1)
      VT.append(v2)
      V.append(v3)
      C.append(co)

    return De,VI,VT,V,C

def onhot_analysis(results_dir,n):
    df = pd.read_excel(results_dir)
    An = df['answer']
    df['Variance_Onehot_Image'] = np.nan
    df['Variance_Onehot_Text'] = np.nan
    df['Variance_Onehot'] = np.nan
    df['Cov_Onehot'] = np.nan
    diseases = [
    "Cardiac", #Cardiac arrhythmias
    "Cardiomyopathy", #Cardiomyopathy
    "Chronic", #Chronic rheumatic heart diseases
    "Complications", #Complications of heart disease
    "Conduction", #Conduction disorders
    "Failure", #Heart failure
    "Hypertensive", #Hypertensive diseases
    "Ischaemic", #Ischaemic heart diseases
    "Pericarditis", #Pericarditis
    "Pulmonary", #Pulmonary heart disease
    "Valve" #Valve disorders
    ]
    VI,VT,V,C = [],[],[],[]
    for i in range(len(An)):
        print(i)
        One_hot_image,One_hot_text,One_hot = [],[],[]
        ranges = [(0, n), (n+1, (n+1)*2-1), ((n+1)*2, (n+1)*3-1)]
        samples = [np.random.choice(np.arange(start, end + 1), n, replace=False) for start, end in ranges]
        Ra = np.concatenate(samples)
        for item in Ra:
            Pre = df['pre_{}'.format(item)][i]
            Ans = df['answer'][i]
            ind = 0
            for disease in diseases:
              if disease.lower() in str(Ans).lower():
                  onehot_an = np.zeros(11)
                  onehot_an[ind] = 1
              else:
                  ind+=1
            ind = 0
            for disease in diseases:
              if disease.lower() in str(Pre).lower():
                  onehot = np.zeros(11)
                  if ind != 11:
                     onehot[ind] = 1
                  if item <= n:
                      One_hot_image.append(onehot)
                  if n < item < (n+1)*2:
                      One_hot_text.append(onehot)
                  if (n+1)*2  <= item < (n+1)*3:
                      One_hot.append(onehot)
                  break
              else:
                  ind += 1

        Var_Onehot_Image = compute_variance(One_hot_image)
        Var_Onehot_Text = compute_variance(One_hot_text)
        Var_Onehot = compute_variance(One_hot)
        Cov_Onehot = np.sqrt(Var_Onehot_Image) * np.sqrt(Var_Onehot_Text)
        VI.append(Var_Onehot_Image)
        VT.append(Var_Onehot_Text)
        V.append(Var_Onehot)
        C.append(Cov_Onehot)

        #stack = np.stack(One_hot_image, axis=0)
        #MVI.append(np.mean(stack, axis=0))
        #stack = np.stack(One_hot_text, axis=0)
        #MVT.append(np.mean(stack, axis=0))
        #stack = np.stack(One_hot, axis=0)
        #MV.append(np.mean(stack, axis=0))
        #RI.append(onehot_an)
        #RT.append(onehot_an)
        #R.append(onehot_an)
    return VI,VT,V,C

def entropy_analysis(results_dir):
    df = pd.read_excel(results_dir)
    pe1 = df['Predictive_Etropy_1']
    pe2 = df['Predictive_Etropy_2']
    pe3 = df['Predictive_Etropy_3']
    con1 = df['Conditional_Etropy_1']
    con2 = df['Conditional_Etropy_2']
    con3 = df['Conditional_Etropy_3']
    return (np.array(pe1),np.array(pe2),np.array(pe3),
            np.array(con1),np.array(con2),np.array(con3))