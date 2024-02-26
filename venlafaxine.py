# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 08:58:26 2022

@author: yuan 抑郁症 20221011 文拉法辛预测模型
"""

from sklearn.metrics import r2_score,average_precision_score,precision_recall_curve
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,accuracy_score
from pytorch_tabnet.tab_model import TabNetRegressor

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(0)
import xgboost as xgb


import os
# import wget
from pathlib import Path
#%%
# 读入数据
df4 = pd.read_excel(r'df3.xlsx')


col_seq = [ 'AGE', '体重', '身高', 'α-L-岩藻糖苷酶', 'α-羟丁酸脱氢酶', 'γ-谷氨酰转肽酶', '丙氨酸氨基转移酶', '中性/淋巴',
        '中性粒细胞百分数', '中性粒细胞绝对值', '乳酸脱氢酶', '二氧化碳结合力', '单核细胞百分数', '单核细胞绝对值',
        '嗜碱性粒细胞百分数', '嗜碱性粒细胞绝对值', '嗜酸性粒细胞百分数', '嗜酸性粒细胞绝对值', '大的不成熟细胞%',
        '天门冬氨酸氨基转移酶', '天门冬氨酸氨基转移酶/丙氨酸氨基转移酶', '尿素', '尿酸', '平均红细胞体积', '平均血小板体积',
        '平均血红蛋白浓度', '平均血红蛋白量', '总胆汁酸', '总胆红素', '总蛋白', '有核红细胞百分数', '有核红细胞计数',
        '淋巴细胞百分数', '淋巴细胞绝对值', '球蛋白', '白球比例', '白细胞计数', '白蛋白(溴甲酚绿）', '直接胆红素',
        '红细胞体积分布宽度CV', '红细胞体积分布宽度SD', '红细胞压积', '红细胞计数', '肌酐（酶法）', '肌酸激酶',
        '胆碱酯酶', '腺苷脱氨酶', '血小板/淋巴', '血小板体积分布宽度', '血小板压积', '血小板计数', '血红蛋白',
        '间接胆红素']

col_class = ['SEX', '呼吸系统疾病', '神经系统疾病', '危重症疾病', '心血管疾病', '泌尿疾病', '内分泌与代谢疾病',
        '消化系统疾病', '高血压', '冠心病', '心力衰竭', '心律失常', '肾功能异常', '尿路感染', '糖尿病', '高脂血症',
        '低血钾', 'CYP2C19酶抑制剂', 'CYP2C19酶竞争性底物', 'CYP2C19酶诱导剂', 'CYP2D6酶抑制剂',
        'CYP2D6酶竞争性底物', 'β-受体阻滞剂', '三环类抗抑郁药', '抗心律失常药', '单胺氧化酶抑制剂']

# 分类变量计数
for i in col_class:
    print(df4[i].value_counts())
#%%
def fun1(num):
    if num==75:
        return 0
    elif num==150:
        return 1
    else:
        return 2
df4['DOSAGE1'] = df4['DOSAGE'].apply(fun1)
#%%
col_class_balance = ['SEX', '高血压', '内分泌与代谢疾病', '高脂血症', '心血管疾病', 'DOSAGE1', '糖尿病'
                     ,'CYP2C19酶抑制剂', 'CYP2C19酶竞争性底物',  'CYP2D6酶抑制剂',
        'CYP2D6酶竞争性底物', 'β-受体阻滞剂'] 

# 显著性检验

from scipy import stats
import scipy.stats as st
from scipy.stats import chi2_contingency

##检验是否正态
def norm_test(data):
    t,p =  stats.shapiro(data)
    #print(t,p)
    if p>=0.05:
        return True
    else:
        return False
    
def test2(data_b, data_p):
    if norm_test(data_b) and norm_test(data_p):
        x = 1
        y = '独立样本T检验'
        t, p = st.ttest_ind(list(data_b),list(data_p), nan_policy='omit')
    else:
        x = 0
        y = 'Mann-Whitney U检验'
        t,p = st.mannwhitneyu(list(data_b),list(data_p))
    return x,y,t,p

# train = pd.concat([X_train, y_train], axis=1)
for i in col_class_balance:
    data_b = df4[df4[i]==1]['RESULT']
    data_p = df4[df4[i]==0]['RESULT']
    x,y,t,p = test2(data_b, data_p)
    print(i, p)

#%%
from scipy import stats
x_num = 0
n_list = []
t_list = []
p_list = []
q_list = []
for i in col_seq:             
    x = df4[df4[i].notna()]['RESULT']
    y = df4[df4[i].notna()][i]

    t,p = stats.spearmanr(x, y)
    t = round(t, 2)
    p = round(p, 3)
    q = '斯皮尔曼'
    print(i, t, p)
    print('====================================================')
        
    n_list.append(i)
    t_list.append(t)
    p_list.append(p)
    q_list.append(q )
    x_num += 1

df_check_c = pd.DataFrame(data={'检验项目':list(col_seq),
# df_xiangguan = pd.DataFrame(data={'其他用药':drug_list,
                                  't值':t_list,
                                  'p值':p_list,
                                  '方法':q_list})
df_check_c1 = df_check_c[df_check_c['p值']<=0.05]
df_check_c2 = df_check_c[df_check_c['p值']>=0.05]    

df_check_c1.检验项目.unique()

#%%

col_xian = ['DOSAGE', 'SEX', '高脂血症', 'AGE',   'α-羟丁酸脱氢酶', 'γ-谷氨酰转肽酶', '丙氨酸氨基转移酶', '中性/淋巴',
        '中性粒细胞百分数', '中性粒细胞绝对值', '乳酸脱氢酶', 
        '嗜碱性粒细胞百分数','嗜酸性粒细胞百分数', 
        '天门冬氨酸氨基转移酶', '天门冬氨酸氨基转移酶/丙氨酸氨基转移酶', '尿素', '尿酸', '平均红细胞体积', '平均血小板体积',
        '平均血红蛋白浓度', '平均血红蛋白量', '总胆汁酸', '总胆红素', '总蛋白', '有核红细胞百分数', '有核红细胞计数',
        '淋巴细胞百分数',  '球蛋白', '白球比例', '白细胞计数', '白蛋白(溴甲酚绿）', '直接胆红素',
        '红细胞体积分布宽度CV', '红细胞压积', '红细胞计数', '肌酐（酶法）', '肌酸激酶',
        '胆碱酯酶', '腺苷脱氨酶', '血小板/淋巴', '血小板体积分布宽度', '血小板压积', '血小板计数', '血红蛋白',
        '间接胆红素']

col_xian = ['DOSAGE', 'SEX', '高脂血症','AGE', 'γ-谷氨酰转肽酶', '丙氨酸氨基转移酶', '球蛋白', '白球比例', '肌酐（酶法）', '胆碱酯酶',
       '腺苷脱氨酶']

#%%
#均值替换异常值
# df2['CREA(umol/L)'].replace(2723, 48, inplace=True)

# df4 = df.copy()
col_1 = ['AGE',   'α-羟丁酸脱氢酶', 'γ-谷氨酰转肽酶', '丙氨酸氨基转移酶', '中性/淋巴',
        '中性粒细胞百分数', '中性粒细胞绝对值', '乳酸脱氢酶', 
        '嗜碱性粒细胞百分数','嗜酸性粒细胞百分数', 
        '天门冬氨酸氨基转移酶', '天门冬氨酸氨基转移酶/丙氨酸氨基转移酶', '尿素', '尿酸', '平均红细胞体积', '平均血小板体积',
        '平均血红蛋白浓度', '平均血红蛋白量', '总胆汁酸', '总胆红素', '总蛋白', '有核红细胞百分数', '有核红细胞计数',
        '淋巴细胞百分数',  '球蛋白', '白球比例', '白细胞计数', '白蛋白(溴甲酚绿）', '直接胆红素',
        '红细胞体积分布宽度CV', '红细胞压积', '红细胞计数', '肌酐（酶法）', '肌酸激酶',
        '胆碱酯酶', '腺苷脱氨酶', '血小板/淋巴', '血小板体积分布宽度', '血小板压积', '血小板计数', '血红蛋白',
        '间接胆红素']

col_1 = ['AGE', 'γ-谷氨酰转肽酶', '丙氨酸氨基转移酶', '球蛋白', '白球比例', '肌酐（酶法）', '胆碱酯酶',
       '腺苷脱氨酶']
for i in col_1:
    df4[i] = df4[i].apply(lambda x: np.log(x) if x > 0 else np.nan if x!=x else 0)

#%%
#选择最优的种子
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, r2_score
import catboost
import math


#%%
# SFS

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import xgboost as xgb
selected_feat=[]
selected_feat1=[]
all_results = []
f1=[]
f2=[]
train_x, test_x, train_y, test_y = train_test_split(df4[col_xian],
                                                    df4['RESULT'],
                                                    test_size=0.2,
                                                    random_state=340)
for i in range(1,45):
    print(i)
    pra = {'depth': 5, 'iterations': 600, 'l2_leaf_reg': 9, 'learning_rate': 0.03}
    sfs1 = SFS(xgb.XGBRegressor(max_depth=5,
                        learning_rate=0.05,
                        n_estimators=200,
                        min_child_weight=0.6,
                        eta=0.5,
                        gamma=0.5,
                        reg_lambda=5,
                        subsample=0.8,
                        colsample_bytree=0.6,
                        nthread=10,
                        scale_pos_weight=1
                        ), 
                   k_features=i, 
                   forward=True, 
                   floating=False, 
                   verbose=2,
                   scoring='r2',
                   cv=2,
                   n_jobs=-1)
    sfs1 = sfs1.fit(train_x, train_y)
    selected_feat= train_x.columns[list(sfs1.k_feature_idx_)]
    selected_feat1.append(selected_feat)
    f2.append(sfs1.k_score_)

#%%

from sklearn.metrics import r2_score,average_precision_score,precision_recall_curve
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,accuracy_score
from pytorch_tabnet.tab_model import TabNetRegressor

import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(0)


import os
# import wget
from pathlib import Path


col_in = ['DOSAGE', 'SEX', '高脂血症', 'AGE', '腺苷脱氨酶']


train_1 = df4.copy()
target = 'RESULT'
train_1 = train_1.reset_index().drop(columns='index')
train_2 = train_1[selected_feat1[4]]
for col in selected_feat1[4]:
    train_2[col].fillna(train_2[ col].mean(), inplace=True) 
train_x, test_x, train_y, test_y = train_test_split(train_2,
                                                        train_1['RESULT'],
                                                        test_size=0.2,
                                                        random_state=1536)
train1 = pd.concat([train_x, train_y],axis=1)
train1['Set'] = 1
train3 = pd.concat([test_x, test_y],axis=1)
train3['Set'] = 0

 
train=pd.concat([train1, train3])
unused_feat = ['Set']


train_indices = train[train.Set==1].index
test_indices = train[train.Set==0].index
# wg_set = pd.merge(train.reset_index(), df1.reset_index()[['index', 'ID', 'NM']], how='left', on='index')
  
col_class = [
        'SEX', 'DOSAGE', '高脂血症']
col_seq = [col for col in train.columns if col not in col_class+[target]+[unused_feat]]


# Simple preprocessing
# Label encode categorical features and fill empty cells.

categorical_columns = []
categorical_dims =  {}
# for col in train.columns[train.dtypes == object]:
for col in col_class:
    print(col, train[col].nunique())
    l_enc = LabelEncoder()
    train[col] = train[col].fillna("VV_likely")
    train[col] = l_enc.fit_transform(train[col].values)
    categorical_columns.append(col)
    categorical_dims[col] = len(l_enc.classes_)
# for col in train.columns[train.dtypes == 'int64']:
for col in col_seq:
    train.fillna(train.loc[train_indices, col].mean(), inplace=True) 
unused_feat = ['Set']

features = [ col for col in train.columns if col not in unused_feat+[target]] 

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    
    # define your embedding sizes : here just a random choice
cat_emb_dim = [2, 3]
tran_x = train[train['Set']==1][features]
tran_y = train[train['Set']==1][target]

test_x = train[train['Set']==0][features]
test_y = train[train['Set']==0][target]


                                                              
import xgboost
import xgboost as xgb
# XGBoost模型
xgb_model=xgb.XGBRegressor(max_depth=3,
                             learning_rate=0.06,#0.05 0.64 ;0.07/0.09--0.65
                             n_estimators=170,
                             min_child_weight=0.7,
                             eta=0.8,
                             gamma=0.6,
                             reg_lambda=10,
                             subsample=0.7,
                             colsample_bytree=0.8,
                             nthread=10,
                             scale_pos_weight=3)

xgb_model.fit(tran_x,tran_y)
xgb_predictions=xgb_model.predict(test_x)
print( round(r2_score(test_y, xgb_predictions),3))

import lightgbm
# LightGBM模型
# for i in range(100):
params = {
        'boosting_type': 'gbdt',
        'n_estimators': 300,
        'n_jobs': 6,
        "objective" : "regression",
        "metric" : "r2",
        "num_leaves" : 128,
        "learning_rate" : 0.04,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.8,
        "bagging_frequency" : 11,
        "bagging_seed" : 16,
        "verbosity" : -1,
        "seed": 1
    }
lgbm_model=lightgbm.LGBMRegressor(**params)
lgbm_model.fit(tran_x,tran_y)
lgbm_predictions=lgbm_model.predict(test_x)
print( round(r2_score(test_y, lgbm_predictions),3))

import catboost
# CatBoost模型
pra = {'depth': 4, 'iterations': 300, 'l2_leaf_reg': 9, 'learning_rate': 0.03}
cat_model=catboost.CatBoostRegressor(**pra)
cat_model.fit(tran_x,tran_y)
cat_predictions=cat_model.predict(test_x)
print(round(r2_score(test_y, cat_predictions),3))

# 随机森林
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# 列出参数列表
# tree_grid_parameter = {'n_estimators': list((10, 50, 100, 150, 200))}
# # 进行参数的搜索组合
# grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_grid_parameter, cv=3)
# 根据已有数据去拟合随机森林模型
# grid.fit(tran_x, tran_y)
# grid.best_params_['n_estimators']
rf_model = RandomForestRegressor(n_estimators=200,
                            max_depth=2,
                            random_state=1)
rf_model.fit(tran_x, tran_y)
# 预测缺失值
rf_predictions = rf_model.predict(test_x)
# print(round(r2_score(test_y, rf_predictions),3))

# GBDT
# 列出参数列表
gbdt_model = GradientBoostingRegressor(n_estimators=200,
                            learning_rate=0.1,
                            max_depth=4,
                            subsample=0.4,
                            random_state=3)
gbdt_model.fit(tran_x,tran_y)
# 预测缺失值
gbdt_predictions = gbdt_model.predict(test_x)

# SVR
from sklearn.svm import SVR,SVC
# 回归模型
# svr = SVR(kernel='linear', C=1.25)
# 分类模型
svr_model = SVR(kernel='linear', C=50)
svr_model.fit(tran_x,tran_y)
svr_predictions=svr_model.predict(test_x)


# Linear回归，Lasso回归，领回归，logistic回归
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,LogisticRegression
# lcv_model = LogisticRegression()

# LinearRegression
lcv_model = Ridge(alpha=12)
# lcv = Ridge()
lcv_model.fit(tran_x, tran_y)
lcv_predictions = lcv_model.predict(test_x)




# ANN
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix

ANN_model = MLPRegressor(alpha=10, max_iter=600,
                    hidden_layer_sizes=[300,], 
                    solver='adam', 
                    activation='relu', 
                    random_state=3)
ANN_model.fit(tran_x, tran_y)
ANN_predictions=ANN_model.predict(test_x)
print(round(r2_score(test_y, ANN_predictions),3))

# TabNet
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
TabNet_model = TabNetRegressor() # cat_dims=cat_dims, cat_emb_dim=cat_emb_dim, cat_idxs=cat_idxs
TabNet_model.fit(
        X_train=tran_x.to_numpy(), y_train=tran_y.to_numpy().reshape(-1,1),
        eval_set=[(tran_x.to_numpy(), tran_y.to_numpy().reshape(-1,1)), (test_x.to_numpy(), test_y.to_numpy().reshape(-1,1))],
        eval_name=['train', 'valid'],
        eval_metric=['mae', 'rmse'],
        max_epochs=100,
        patience=50,
        batch_size=256, virtual_batch_size=128,  # (68, 128,16, 0.61)
        num_workers=0,
        drop_last=False,
        # seed=k
        
    )

TabNet_predictions=TabNet_model.predict(test_x.to_numpy())
print(round(r2_score(test_y, TabNet_predictions),3))

# GBDT
# 列出参数列表
gbdt_model = GradientBoostingRegressor(n_estimators=500,
                            learning_rate=0.07,
                            max_depth=5,
                            subsample=0.4,
                            random_state=3)
gbdt_model.fit(tran_x,tran_y)
# 预测缺失值
gbdt_predictions = gbdt_model.predict(test_x)

# SVR
from sklearn.svm import SVR,SVC
# 回归模型
# svr = SVR(kernel='linear', C=1.25)
# 分类模型
svr_model = SVR(kernel='linear', C=50,epsilon=3)
svr_model.fit(tran_x,tran_y)
svr_predictions=svr_model.predict(test_x)
print(round(r2_score(test_y, svr_predictions),3))
#%%
df_model_result=pd.DataFrame(
columns=['model','R2','RMSE','MAE','MAPE', 'MPE','Accuracy within ± 10% range','Accuracy within ± 20% range','Accuracy within ± 30% range',
         'Accuracy within ± 40% range', 'Accuracy within ± 50% range'])
model_list=[xgb_model,lgbm_model,cat_model,rf_model,gbdt_model,svr_model,lcv_model,ANN_model,TabNet_model]#
model_name_list=['XGBoost','LGBM','CatBoost','RF','GBDT','SVR','LR','ANN','TabNet']
for model,name in zip(model_list,model_name_list):
#     print(name)
    # 计算R2、RMSE、MAE
    if name == 'TabNet':
        predictions=model.predict(test_x.to_numpy())
    else:
        predictions=model.predict(test_x)
    r2=r2_score(test_y,predictions)
    r2=round(r2,4)
    mae=mean_absolute_error(test_y,predictions)
    
    mae=round(mae,4)
    rmse=mean_squared_error(test_y,predictions) ** 0.5
    rmse=round(rmse,4)
    # mpe=round(mean_percentage_error(test_y,predictions),4)
    
    # mape=round(mean_absolute_percentage_error(test_y,predictions),4)
    
 
    # 计算'Accuracy within ± 10%, 20%, 30%, 40% range'
    accuracy_10_list = [ (i,j) for i,j in zip(test_y,predictions) if abs((i-j)/i)<=0.1]
    accuracy_10_perc = round(len(accuracy_10_list)/len(test_y),4)
    accuracy_10_perc="%.2f%%" % (accuracy_10_perc * 100)      # 百分数输出

    accuracy_20_list = [ (i,j) for i,j in zip(test_y,predictions) if abs((i-j)/i)<=0.2]
    accuracy_20_perc = round(len(accuracy_20_list)/len(test_y),4)
    accuracy_20_perc="%.2f%%" % (accuracy_20_perc * 100)      # 百分数输出
    
    accuracy_30_list = [ (i,j) for i,j in zip(test_y,predictions) if abs((i-j)/i)<=0.3]
    accuracy_30_perc = round(len(accuracy_30_list)/len(test_y),4)
    accuracy_30_perc="%.2f%%" % (accuracy_30_perc * 100)      # 百分数输出
    
    accuracy_40_list = [ (i,j) for i,j in zip(test_y,predictions) if abs((i-j)/i)<=0.4]
    accuracy_40_perc = round(len(accuracy_40_list)/len(test_y),4)
    accuracy_40_perc="%.2f%%" % (accuracy_40_perc * 100)      # 百分数输出
    
    accuracy_50_list = [ (i,j) for i,j in zip(test_y,predictions) if abs((i-j)/i)<=0.5]
    accuracy_50_perc = round(len(accuracy_50_list)/len(test_y),4)
    accuracy_50_perc="%.2f%%" % (accuracy_50_perc * 100)      # 百分数输出
    
    df_model_result.loc[df_model_result.shape[0],['model','R2','RMSE','MAE', 'Accuracy within ± 10% range', #'MAPE', 'MPE',
                                                  'Accuracy within ± 20% range','Accuracy within ± 30% range',
                                                 'Accuracy within ± 40% range', 'Accuracy within ± 50% range']]=\
                                                  [name, r2, rmse, mae,  accuracy_10_perc, accuracy_20_perc, #mape, mpe,
                                                   accuracy_30_perc, accuracy_40_perc, accuracy_50_perc]
    df_model_result=df_model_result.reset_index(drop=True)

df_model_result.to_excel('df_model_result_quetiapine.xlsx')
#%%

important_features = pd.DataFrame([xgb_model.feature_importances_] , columns = test_x.rename(columns={'高脂血症':'HLP', '腺苷脱氨酶':'ADA', 'DOSAGE':'Venlafaxine'}).columns, index=['Importance'])     
important_features =important_features.transpose().sort_values(by=['Importance'], ascending=False).head(11)
important_features

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文宋体

plt.rcParams['axes.unicode_minus']=False #显示负号
fig = plt.figure(figsize=(16, 8), facecolor='white')  #创建figure对象
ax = fig.add_subplot(1, 1, 1)  #获得Axes对象
# plt.bar( important_features.index[0:12],important_features['Importance'][0:12],color='lightsteelblue')
plt.bar( important_features.index,important_features['Importance'],color='lightsteelblue')
# plt.ylim(0.05, 0.13)
plt.ylabel('Importance Score')
plt.title('Feature Importance of Catboost Model')
plt.show() 

#%%

#shap图
import shap
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] ##绘图显示中文
mpl.rcParams['axes.unicode_minus'] = False

# load JS visualization code to notebook
shap.initjs()

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(tran_x)
shap.summary_plot(shap_values, tran_x.rename(columns={'高脂血症':'HLP', '腺苷脱氨酶':'ADA', 'DOSAGE':'Venlafaxine'}),max_display=5)
shap.summary_plot(shap_values, tran_x.columns, plot_type="bar",max_display=30)

#%%


#%%

#划分等级200，200-750，750

y_pred_cat = cat_predictions

test_x['谷浓度'] = test_y
test_x['预测值'] = y_pred_cat
a = test_x[test_x['谷浓度']<100 ]
#a = a[a['谷浓度']<150 ]
a['谷浓度预测准确率'] = (a['预测值'] - a['谷浓度']) / a['谷浓度']
a['谷浓度预测准确率']  = a['谷浓度预测准确率'].apply(lambda x:1 if abs(x)<=0.3 else 0)
print('0-150 ',round(a[a['谷浓度预测准确率']==1].shape[0] / a.shape[0],4), a.shape[0])


test_x['谷浓度'] = test_y
test_x['预测值'] = y_pred_cat
a = test_x[test_x['谷浓度']>=100 ]
a = a[a['谷浓度']<400 ]
a['谷浓度预测准确率'] = (a['预测值'] - a['谷浓度']) / a['谷浓度']
a['谷浓度预测准确率']  = a['谷浓度预测准确率'].apply(lambda x:1 if abs(x)<=0.3 else 0)
print('150-250 ',round(a[a['谷浓度预测准确率']==1].shape[0] / a.shape[0],4), a.shape[0]) 


test_x['谷浓度'] = test_y
test_x['预测值'] = y_pred_cat
a = test_x[test_x['谷浓度']>=400 ]
#a = a[a['谷浓度']<150 ]
a['谷浓度预测准确率'] = (a['预测值'] - a['谷浓度']) / a['谷浓度']
a['谷浓度预测准确率']  = a['谷浓度预测准确率'].apply(lambda x:1 if abs(x)<=0.3 else 0)
print('250+ ',round(a[a['谷浓度预测准确率']==1].shape[0] / a.shape[0],4), a.shape[0])






























