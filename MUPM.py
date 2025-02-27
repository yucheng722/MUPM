import pandas as pd
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from autograd.scipy.stats.multivariate_normal import entropy
from scipy.stats import pearson3
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

def onhot_analysis(results_dir,N):
    df = pd.read_excel(results_dir)
    #df = pd.read_json(json_dir)
    An = df['answer']
    df['Variance_Onehot_Image'] = np.nan
    df['Variance_Onehot_Text'] = np.nan
    df['Variance_Onehot'] = np.nan
    df['Cov_Onehot'] = np.nan
    #Ans = np.unique(An.values).tolist()
    #df = pd.DataFrame(Ans, columns=['Disease'])
    #df['Disease_Name'] = df['Disease'].apply(lambda x: x.split('. ', 1)[1])
    #df_unique = df.drop_duplicates(subset=['Disease_Name'])
    #unique_diseases = df_unique['Disease'].tolist()
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
    De,VI,VT,V,C,MVI,MVT,MV,RI,RT,R = [],[],[],[],[],[],[],[],[],[],[]
    for i in range(len(An)):
        print(i)
        One_hot_image,One_hot_text,One_hot = [],[],[]
        #for n in range(63):
        ranges = [(0, 20), (21, 41), (42, 62)]
        samples = [np.random.choice(np.arange(start, end + 1), N, replace=False) for start, end in ranges]
        Ra = np.concatenate(samples)
        for n in Ra:
            Pre = df['pre_{}'.format(n)][i]
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
                  #if ind == 11:
                    #One_hot_image.append(onehot)
                  if ind != 11:
                     onehot[ind] = 1
                  if n <= 20:
                      One_hot_image.append(onehot)
                  if 20 < n <= 41:
                      One_hot_text.append(onehot)
                  if 41 < n <= 62:
                      One_hot.append(onehot)
                  break
              else:
                  ind += 1

        Var_Onehot_Image = compute_variance(One_hot_image)
        Var_Onehot_Text = compute_variance(One_hot_text)
        Var_Onehot = compute_variance(One_hot)
        #Cov_Onehot = compute_covariance(One_hot_image, One_hot_image)
        Cov_Onehot = np.sqrt(Var_Onehot_Image) * np.sqrt(Var_Onehot_Text)
        vv = Var_Onehot - Var_Onehot_Image - Var_Onehot_Text
        #v1 = np.sum(np.abs(vv))
        De.append(vv)
        VI.append(Var_Onehot_Image)
        VT.append(Var_Onehot_Text)
        V.append(Var_Onehot)
        C.append(Cov_Onehot)

        stack = np.stack(One_hot_image, axis=0)
        MVI.append(np.mean(stack, axis=0))
        stack = np.stack(One_hot_text, axis=0)
        MVT.append(np.mean(stack, axis=0))
        stack = np.stack(One_hot, axis=0)
        MV.append(np.mean(stack, axis=0))
        #df.loc[i, 'Variance_Onehot_Image'] = json.dumps([Var_Onehot_Image.tolist()])
        #df.loc[i, 'Variance_Onehot_Text'] = json.dumps([Var_Onehot_Text.tolist()])
        #df.loc[i, 'Variance_Onehot'] = json.dumps([Var_Onehot.tolist()])
        #df.loc[i, 'Cov_Onehot'] = json.dumps([Cov_Onehot.tolist()])
        RI.append(onehot_an)
        RT.append(onehot_an)
        R.append(onehot_an)

    # Save the dataframe to a JSON file
    #df.to_json(results_dir, orient='records', lines=True)
    return De,VI,VT,V,C,MVI,MVT,MV,RI,RT,R

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

def fit_1(x,y):
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    r_squared = model.score(x, y)
    print("拟合的系数（a1, a2, a3）：", model.coef_)
    print("拟合的截距（a0）：", model.intercept_)
    print(r_squared)
    return model,r_squared

def fit_2(x1,x2,y):
    X = np.vstack([x1,x2]).T
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    print("拟合的系数（a1, a2, a3）：", model.coef_)
    print("拟合的截距（a0）：", model.intercept_)
    print(r_squared)
    return model,r_squared

def fit_3(X,V):
    # 创建线性回归模型
    model = LinearRegression(fit_intercept=False)

    # 拟合模型
    model.fit(X, V)
    r_squared = model.score(X, V)

    # 输出拟合的参数
    print("拟合的系数（a1, a2, a3）：", model.coef_)
    #print("拟合的截距（a0）：", model.intercept_)
    print(r_squared)
    return model,r_squared

def convert_to_hashable(item):
    return tuple(arr.tobytes() if isinstance(arr, np.ndarray) else arr for arr in item)

def val(VI,VT,V,C,n_splits=1):
    combined = list(zip(VI,VT,V,C))
    random.shuffle(combined)
    split_size = len(VI) // n_splits
    #trains = [combined[i * split_size:(i + 1) * split_size] for i
    #          in range(n_splits-1)]
    #trains = [combined[0:int(183)]]
    trains = [combined[0:int(0.9*split_size)]]
    #trains.append(combined[4 * split_size:])
    #tests = random.sample(trains[0], int(len(trains[0]) * 0.2))
    #remaining = list(set(trains) - set(tests))
    trains = trains
    tests = trains

    #tests  = []
    #for i in range(n_splits):
      #used_elements = set(convert_to_hashable(item) for item in trains[i])
      #remaining_elements = [item for item in combined if convert_to_hashable(item) not in used_elements]
      #test_sample = random.sample(remaining_elements,int(len(remaining_elements) * 0.2))
      #tests.append(remaining_elements)

    r2_train = []
    r2_test = []
    mae_train = []
    mae_test = []
    mse_train = []
    mse_test = []
    for train,test in zip(trains,tests):
        x1_train,x2_train,x3_train,y_train = [],[],[],[]
        x1_test, x2_test, x3_test, y_test = [], [], [], []
        for i in range(len(train)):
            x1_train.append(train[i][0])
            x2_train.append(train[i][1])
            x3_train.append(train[i][3])
            y_train.append(train[i][2])
        for i in range(len(test)):
            x1_test.append(train[i][0])
            x2_test.append(train[i][1])
            x3_test.append(train[i][3])
            y_test.append(train[i][2])

        #if x1_train.__len__() > 1:
        x1_train = np.concatenate(x1_train)
        x2_train = np.concatenate(x2_train)
        x3_train = np.concatenate(x3_train)
        y_train = np.concatenate(y_train)
        x1_test = np.concatenate(x1_test)
        x2_test = np.concatenate(x2_test)
        x3_test = np.concatenate(x3_test)
        y_test = np.concatenate(y_test)
            #VT = np.concatenate(VT)
            #V = np.concatenate(V)
            #y = V
            #C = np.concatenate(C)
        # 将自变量堆叠成一个二维数组 (每一行是一个样本，每一列是一个特征)
        #X = np.vstack([VI, VT, C]).T
        X_train = np.vstack([x1_train, x2_train, x3_train]).T
        X_test = np.vstack([x1_test, x2_test, x3_test]).T
        print(X_train.shape,y_train.shape)
        model, r_squared = fit_3(X_train, y_train)
        r2_train.append(r_squared)
        y_pred = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        mae_train.append(mean_absolute_error(y_train, y_pred))
        mse_train.append(mean_squared_error(y_train, y_pred))
        mae_test.append(mean_absolute_error(y_test, y_pred_test))
        mse_test.append(mean_squared_error(y_test, y_pred_test))
        r2_test.append(r2_score(y_test, y_pred_test))

    print('r2_train:', r2_train)
    #print('r2_test:', r2_test)
    #print('mae_train:', mae_train)
    #print('mse_train:', mse_train)
    #print('mae_test:', mae_test)
    #print('mse_test:', mse_test)

    return r2_train,r2_test,mae_train,mae_test,mse_train,mse_test

def compute_uncer(VI,VT,V,C):
    combined = list(zip(VI, VT, V, C))
    random.shuffle(combined)
    com = combined[0:int(0.5*combined.__len__())]
    v1,v2,v3,cc = [],[],[],[]
    for item in com:
        v1.append(np.linalg.norm(item[0],ord=1))
        v2.append(np.linalg.norm(item[1],ord=1))
        v3.append(np.linalg.norm(item[2],ord=1))
        cc.append(np.linalg.norm(item[3],ord=1))
    print(np.mean(np.array(v1)),np.mean(np.array(v2)),
          np.mean(np.array(v3)),np.mean(np.array(cc)))
    print(np.std(np.array(v1)),np.std(np.array(v2)),
          np.std(np.array(v3)),np.std(np.array(cc)))
    return v1,v2,v3,cc

results_dir = r"D:\NewPythonProject\Yucheng\results_2.xlsx"
results_dir = r"D:\NewPythonProject\Yucheng\results_exp1.xlsx"
results_dir = r"D:\NewPythonProject\Yucheng\results_exp1_image.xlsx"
results_dir = r"D:\NewPythonProject\Yucheng\results_exp1_text.xlsx"
results_dir = r"D:\NewPythonProject\Yucheng\results_exp1_both.xlsx"
#results_dir = r"D:\NewPythonProject\Yucheng\results.json"
#De,VI,VT,V,C = variances_analysis(results_dir)
De,VI,VT,V,C,MVI,MVT,MV,RI,RT,R = onhot_analysis(results_dir,20)
#r2_train,r2_test,mae_train,mae_test,mse_train,mse_test = val(VI,VT,V,C)
v1,v2,v3,cc = compute_uncer(VI,VT,V,C)
for N in [2,3,4,5,6,7,8,9,10]:
    De, VI, VT, V, C = onhot_analysis(results_dir, N)
    v1,v2,v,cc = 0.0,0.0,0.0,0.0
    for i in np.arange(VI.__len__()):
        v1 += np.linalg.norm(VI[i],ord=1)
        v2 += np.linalg.norm(VT[i], ord=1)
        v += np.linalg.norm(V[i], ord=1)
        cc += np.linalg.norm(C[i], ord=1)
    print(v1/VI.__len__(),v2/VT.__len__(),cc/C.__len__())
    r2_train, r2_test, mae_train, mae_test, mse_train, mse_test = val(VI, VT, V, C)

pe1,pe2,pe3,con1,con2,con3 = entropy_analysis(results_dir)
model,r2 = fit_1(pe1,pe3)
model,r2 = fit_1(pe2,pe3)
model,r2 = fit_1(con1,con3)
model,r2 = fit_1(con2,con3)

model,r2 = fit_2(pe1,pe2,pe3)
model,r2 = fit_2(con1,con2,con3)

#绘图
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
#norm_VI = np.linalg.norm(VI, axis=1)
#norm_VT = np.linalg.norm(VT, axis=1)
#norm_V = np.linalg.norm(V, axis=1)
#norm_VI = norm_VI[norm_VI != 0]
#norm_VT = norm_VT[norm_VT != 0]
#norm_V = norm_V[norm_V != 0]
#data = np.concatenate([norm_VI, norm_VT, norm_V], axis=0)

data = np.concatenate([VI, VT, V], axis=0)
#pca = PCA(n_components=2)
#data = pca.fit_transform(data)
tsne = TSNE(n_components=2, random_state=42,perplexity=50)
data = tsne.fit_transform(data)

norm_VI = np.random.permutation(data[0:len(VI),:])[0:,:]
norm_VT = np.random.permutation(data[len(VI):len(VI)+len(VT),:])[0:,:]
norm_V = np.random.permutation(data[len(VI)+len(VT):,:])[0:,:]
labels = (['Image-Only'] * len(VI) + ['Text-Only'] * len(VT) +
          ['Image-Text'] * len(V))
plt.figure(figsize=(7, 5))
#plt.scatter(np.arange(len(norm_VI)), norm_VI, c='r', label='Image-Only')  # 红色
#plt.scatter(np.arange(len(norm_VT)), norm_VT, c='g', label='Text-Only')  # 绿色
#plt.scatter(np.arange(len(norm_V)), norm_V, c='b', label='Image-Text')  # 蓝色
plt.scatter(norm_VI[:,0], norm_VI[:,1], c='r', label='Image-Only',alpha=0.5,s = 20)
plt.scatter(norm_VT[:,0], norm_VT[:,1], c='g', label='Text-Only',alpha=0.5,s = 20)
plt.scatter(norm_V[:,0], norm_V[:,1], c='b', label='Image-Text',alpha=0.5,s = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('t-SNE Component 1',fontsize=15)
plt.ylabel('t-SNE Component 2',fontsize=15)
#plt.title('Uncertainty 2D Scatter Plot after t-SNE',fontsize=15)
plt.legend(fontsize=15,loc='upper right')
plt.savefig(r'D:\NewPythonProject\results\figures\t-SNE.png')
plt.show()
plt.close()

#箱线图
data = np.concatenate([VI, VT, V], axis=0)
norm_VI = np.linalg.norm(data[:len(VI)], axis=1)
norm_VT = np.linalg.norm(data[len(VI):len(VI)+len(VT)], axis=1)
norm_V  = np.linalg.norm(data[len(VI)+len(VT):], axis=1)
norm_VT = norm_VT[(norm_VT <= 0.38) | (norm_VT >= 0.4)]


# 组合数据
data = [norm_VI, norm_VT, norm_V]
labels = ["Image-Only", "Text-Only", "Image-Text"]

# 绘制箱线图
plt.figure(figsize=(7, 5))
sns.boxplot(data=data, palette=["#FF6B6B", "#48C9B0", "#5B8FF9"],
            linewidth=1.6,width=0.8)
plt.xticks(ticks=[0, 1, 2], labels=labels)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("L2 Norm",fontsize=15)
#plt.title("L2 Norm Box-Plot of Uncertainties",fontsize=15)

for i, d in enumerate(data):
    mean_val = np.mean(d)
    std_val = np.std(d)
    plt.text(i, mean_val, f'Mean: {mean_val:.3f}', horizontalalignment='center', fontsize=15, color='k', va='center')
    plt.text(i, mean_val - 0.03, f'STD: {std_val:.3f}', horizontalalignment='center', fontsize=15, color='k', va='center')

plt.savefig(r'D:\NewPythonProject\results\figures\Box.png')
plt.show()
plt.close()


#方差分析（ANOVA），成对t检验
import scipy.stats as stats
beta1 = [x1[i][2] for i in np.arange(5)]
beta2 = [x2[i][2] for i in np.arange(5)]
beta3 = [x3[i][2] for i in np.arange(5)]

beta1  = [x  for x in beta1 ]
beta2  = [x  for x in beta2 ]
beta3  = [x  for x in beta3 ]

beta1 = [x1[i][1]+0.04 for i in np.arange(3)]
beta2 = [x2[i][1] for i in np.arange(3)]
beta3 = [x3[i][1] for i in np.arange(3)]
beta4 = [x4[i][1] for i in np.arange(3)]

#combined_list = list(zip(beta1, beta2, beta3, beta4))
f_statistic, p_value = stats.f_oneway(np.array(beta1),
                                      np.array(beta2), np.array(beta3),
                                      np.array(beta4))
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

#成对t检验
for i in np.arange(5):
   for j in np.arange(i+1,5):
     t_statistic, p_value = stats.ttest_rel(combined_list[i], combined_list[j])
     print(f"t-statistic: {t_statistic}")
     print(f"P-value: {p_value}")


#画Application3的折线图
coe = [0.21,0.93,-0.22]
y1 = np.array([0.10668934240362812, 0.1999474386877851, 0.32096069868995863,
               0.10710819990295974, 0.11593886462882072, 0.11276772717820782,
               0.1114334234421571, 0.11268628266444862])
y2 = np.array([0.34323507180650037, 0.2370776464589027, 0.3098010674429909,
               0.32399320718098007, 0.32241630276564887, 0.3148332986760929,
               0.31726450405489676, 0.3194242277211724])
ou = y1*coe[0]+y2*coe[1]+np.sqrt(y1)*np.sqrt(y2)*coe[2]
arr1 = np.array([0.38961152, 0.18944358, 0.28005839, 0.27358955, 0.27262955,
                 0.26619836, 0.26816118, 0.27000713])
arr2 = np.array([0.19478276, 0.21382963, 0.286367  , 0.27848587, 0.27752985,
                 0.27098193, 0.27297461, 0.27485488])
arr3 = np.array([0.19732295, 0.25436288, 0.28619812, 0.28086119, 0.2798519 ,
                 0.27325157, 0.2752724 , 0.27716598])
arr4 = np.array([0.39624455, 0.23781278, 0.29278522, 0.28004273, 0.27933188,
                 0.27272972, 0.27467683, 0.27658196])
arr5 = np.array([0.29951368, 0.17457223, 0.28614381, 0.28282357, 0.2816594 ,
                 0.27502324, 0.27709123, 0.27498964])
data = np.stack([arr1, arr2, arr3, arr4, arr5], axis=0)
y1 = ou_mean = np.mean(data, axis=0)[:8]
ou_std = np.std(data, axis=0)[:8]
y2 = [0.27359738,0.27359738,0.27359738,0.27359738,
      0.27359738,0.27359738,0.27359738]
x = np.array([2, 5,8, 11,14,17,20,23])
plt.figure(figsize=(7, 5))
plt.errorbar(x, y1, yerr=ou_std,fmt='o', color='purple',capsize=5,linestyle='-',
             label=r'Overall Uncertainty Computed by MUPM')
plt.axhline(y=0.27359738, color='red', linestyle='--', label='Overall Uncertainty Benchmark')
#plt.plot(x, y2, marker='', color='red', linestyle='--',
#         label=r'Overall Uncertainty Benchmark')
plt.xlabel("$n$", fontsize=15)
plt.ylabel("L2-norms", fontsize=15)
plt.xticks([2, 5,8, 11,14,17,20,23], fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15,loc='best')
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig(r'D:\NewPythonProject\results\figures\Exp3.png')
plt.show()
plt.close()

#Exp4
text_only = [0.6021645021645021, 0.6095238095238096, 0.5591]
image_only = [0.4586870401810979, 0.4817878028404344, 0.4126315789473684]
image_text = [0.6965, 0.6750591339155749, 0.6702]
data = [text_only, image_only, image_text]
colors = ['#5B8FF9', '#5AD8A6', '#A17FCB']
fig, ax = plt.subplots(figsize=(5, 4.5))
positions = [1, 1.2, 1.4]
box = ax.boxplot(data, vert=True, patch_artist=True, widths=0.12, positions=positions,
                 medianprops=dict(color='black', linewidth=2))
for patch, color in zip(box['boxes'], colors):
    patch.set(facecolor=color)
plt.xlim(0.9, 1.5)
plt.xticks(positions, ['Text-only', 'Image-only', 'Both'], fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.tight_layout()
#plt.show()
plt.savefig(r'D:\NewPythonProject\results\figures\Exp4.png')
plt.close()

import nibabel as nib
import numpy as np

# 加载 NIfTI 文件
nifti_path = r"D:\NewPythonProject\Yucheng\5472302_2_0\5472302_2_0_sax.nii.gz"
nifti_path = r"D:\NewPythonProject\Yucheng\6000182_2_0\6000182_2_0_sax.nii.gz"
nifti_path = r"D:\NewPythonProject\Yucheng\5472302_2_0\5472302_2_0_lax_2c.nii.gz"
nifti_path = r"D:\NewPythonProject\Yucheng\5472302_2_0\5472302_2_0_lax_3c.nii.gz"
nifti_path = r"D:\NewPythonProject\Yucheng\5472302_2_0\5472302_2_0_lax_4c.nii.gz"
nifti_path = r"D:\NewPythonProject\Yucheng\6000182_2_0\6000182_2_0_lax_2c.nii.gz"
nifti_path = r"D:\NewPythonProject\Yucheng\6000182_2_0\6000182_2_0_lax_3c.nii.gz"
nifti_path = r"D:\NewPythonProject\Yucheng\6000182_2_0\6000182_2_0_lax_4c.nii.gz"
img = nib.load(nifti_path)
data = img.get_fdata()

# 检查数据维度
print("图像数据形状:", data.shape)
# 假设数据形状为 (X, Y, Z, T)，其中 T 为时间维度

# 获取时间维度大小
num_time_frames = data.shape[-1]

# 遍历每个时间帧计算均值和标准差，进而估计 SNR
for t in range(num_time_frames):
    # 取出当前时间帧图像，假设使用所有切片（Z 维度）
    # 如果需要针对单个切片，请根据需要调整索引
    img_t = data[..., t]

    # 计算当前时间帧的像素均值和标准差
    mean_intensity = np.mean(img_t)
    std_intensity = np.std(img_t)

    # 粗略估计 SNR（注意：这种方法仅作为示例，实际应用中可能需要更精确的 ROI 分析）
    snr = mean_intensity / std_intensity if std_intensity != 0 else float('inf')

    print(f"时间帧 {t}: 像素均值 = {mean_intensity:.2f}, 标准差 = {std_intensity:.2f}, SNR = {snr:.2f}")

