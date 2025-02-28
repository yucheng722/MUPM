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

    mean_embd_1 = np.mean(embd_1, axis=0)  # (11,)
    mean_embd_2 = np.mean(embd_2, axis=0)  # (11,)

    centered_embd_1 = embd_1 - mean_embd_1  # (num_tta, 11)
    centered_embd_2 = embd_2 - mean_embd_2  # (num_tta, 11)

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
    print("(b1, b2, b3)：", model.coef_)
    print("(b0)：", model.intercept_)
    print(r_squared)
    return model,r_squared

def fit_2(x1,x2,y):
    X = np.vstack([x1,x2]).T
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    print("(b1, b2, b3)：", model.coef_)
    print("(b0)：", model.intercept_)
    print(r_squared)
    return model,r_squared

def fit_3(X,V):
    model = LinearRegression(fit_intercept=False)

    model.fit(X, V)
    r_squared = model.score(X, V)

    print("(b1, b2, b3)：", model.coef_)
    #print("(b0)：", model.intercept_)
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

#Plot
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

data = np.concatenate([VI, VT, V], axis=0)
norm_VI = np.linalg.norm(data[:len(VI)], axis=1)
norm_VT = np.linalg.norm(data[len(VI):len(VI)+len(VT)], axis=1)
norm_V  = np.linalg.norm(data[len(VI)+len(VT):], axis=1)
norm_VT = norm_VT[(norm_VT <= 0.38) | (norm_VT >= 0.4)]


data = [norm_VI, norm_VT, norm_V]
labels = ["Image-Only", "Text-Only", "Image-Text"]


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



