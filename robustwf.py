import re
import torch
import math
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
from os.path import exists
from sklearn.model_selection import train_test_split
from datamine_function import *
from causality_correlation import *
import json
from collections import defaultdict
from copy import deepcopy
from generate_chain import *
from save_and_read import *
from generate_multi_tab_function import *

from vit_pytorch import ViT
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer,MaxAbsScaler,RobustScaler
from sklearn import metrics
import time

# 序列丢失，重复，错序的相关函数
def duplicate_elements(lst, probability):
    duplicated_list = []
    for num in lst:
        duplicated_list.append(num)
        if random.random() < probability:
            duplicated_list.append(num)
    return duplicated_list

# 从测试集样本抽取样本对流量进行错序，重复，丢失等操作
def delete_random_data(lst, deletion_ratio):
    num_to_delete = int(len(lst) * deletion_ratio)
    indices = random.sample(range(len(lst)), num_to_delete)
    indices.sort(reverse=True) 
    
    for index in indices:
        del lst[index]
        
def shuffle_and_replace_subsequence(lst,probability):
    start = random.randint(0, len(lst)))  # 随机选择子序列的起始索引
    end = random.randint(start + 1, len(lst))  # 随机选择子序列的结束索引（不包括）
    # print(start,end)
    subsequence = lst[start:end]  # 提取子序列
    if(len(subsequence)<=probability*len(lst)*1.2 and len(subsequence)>=0.8*probability*len(lst)):
        random.shuffle(subsequence)  # 错序子序列
        lst[start:end] = subsequence  # 替换原始子序列
        return lst
    else:
        return shuffle_and_replace_subsequence(lst,probability) 


# 按照块进行丢失
def drop_data_randomly(input_list, drop_rate):
    # 计算需要丢失的数据点数量
    num_to_drop = int(len(input_list) * drop_rate)
    
    if num_to_drop == 0:
        print("丢失比例过小，不会丢失任何数据点。")
        return input_list

    # 随机选择一个起始点
    start_index = np.random.randint(0, int(len(input_list))*0.1)

    # 构建要丢失的数据点索引列表
    drop_indices = []
    for i in range(num_to_drop):
        drop_indices.append((start_index + i) % len(input_list))

    # 生成新的列表存放未丢失的数据
    output_list = [input_list[i] for i in range(len(input_list)) if i not in drop_indices]

    return output_list
  

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if len(union) == 0:
        return 0.0
    else:
        return len(intersection) / len(union)



# 读取数据 选取100个网站的数据作为训练集
my_app_list = np.array(pd.read_csv(r'./csy/code/Packet-print/packetprint_project-main/model/apps.txt',header=None)).flatten().tolist()
trace_path="./csy/code/web_fingerprinting/dataset/Monkey500/"
#在这里准备一些超参数
#选取的网站标签数目
max_sample_num=50
website_num=100
positive_training_percentage=1


# Seq-print的相关参数
#成链的最短长度 
chain_len=3

# 支持度和置信度相关
min_confidence=0.5
max_support_num_per_sample=20

# 向后关联的数目M 
assoc_num=4
# train_assoc_num=4
# test_assoc_num=5

# 因果关联阈值
mining_threshold=-0.5

# 是否需要开启模型的重新训练
# 0代表不管有无模型，均开启训练
# 1代表，若有模型则不训练
retrain_para=1
RF_retrain_para=1
TF_retrain_para=1

# txttovec的M和N的大小
max_N=200
max_M=7

# ？丢失，重复，错序比例
probability=0.1

# 重叠比例
overlapping=0.3

# 相关模型及存储路径
all_node_save_path="./all_node/"+"openworld_web="+str(website_num)+"thre="+str(mining_threshold)+"max_sup="+str(max_support_num_per_sample)+"N*M="+str(max_N)+"*"+str(max_M)+".txt"
all_template_save_path="./all_template/"+"openworld_web="+str(website_num)+"thre="+str(mining_threshold)+"max_sup="+str(max_support_num_per_sample)+"N*M="+str(max_N)+"*"+str(max_M)+".txt"
transformer_model_save_path="./model_save/"+"openworld_transformerweb="+str(website_num)+"thre="+str(mining_threshold)+"max_sup="+str(max_support_num_per_sample)+"N*M="+str(max_N)+"*"+str(max_M)+".pkl"
RF1_mode_save_path="./model_save/"+"openworld_RFweb="+str(website_num)+"thre="+str(mining_threshold)+"max_sup="+str(max_support_num_per_sample)+"N*M="+str(max_N)+"*"+str(max_M)+".pkl"

# transfomer VIT 模型训练的相关参数
lr = 0.001
num_epoch = 50
BATCH_SIZE= 50
TF_model=ViT(
    image_size=(max_N,max_M),
    patch_size=(1,max_M),
    num_classes=website_num,
    dim=256,
    depth=4,
    heads=4,
    mlp_dim=512,
    dropout=0.1,
    channels=1)

device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# 读取数据
positive_training_set=[]
for i in range(website_num):
    positive_file_index = list(range(max_sample_num))
    positive_file_index = positive_file_index[:int(max_sample_num * positive_training_percentage)]
    positive_file_list = []
    for sample_index in positive_file_index:
        file_name = trace_path + my_app_list[i] + '_' + str(sample_index) + '.txt'
        if exists(file_name) is False:
            continue
        data = np.array(pd.read_csv(file_name, header=None))
        sample =data[:, -2]
        tag1 = my_app_list[i]
        tag2 = i
        positive_training_set.append([sample,tag1,tag2])

data_set=np.array(positive_training_set)
data0= data_set[:,0]
label= data_set[:,2].astype(int)
print("data process end")


# 划分训练集和测试集
train_set,test_set,train_label,test_label = train_test_split(data0,label,test_size=0.2,random_state=10)
print("划分训练集和测试集")
print(train_set.shape)
print(label)



# 训练RF-labeling
if (exists(RF1_mode_save_path) and retrain_para):
    print("model exists")
    RF1=torch.load(RF1_mode_save_path)
else:
    print("no model")
    train_array=generate_distribution_3000_with_one_label(train_set,train_label)
    test_array=generate_distribution_3000_with_one_label(test_set,test_label)
    x_data_1=train_array[:,:-1]
    y_data_1=train_array[:,-1]
    RF1 = RandomForestClassifier(n_estimators=200)
    RF1.fit(x_data_1, y_data_1)
    torch.save(RF1,RF1_mode_save_path)


# Robust_WF 因果关联发现以及Transformer模型训练
print("all_node_path",all_node_save_path)
print("template_path",all_template_save_path)
print("TF model path",transformer_model_save_path)
if(exists(all_node_save_path) and exists(all_template_save_path) and exists(transformer_model_save_path) and retrain_para):
    print("all_node all_template ,tf model exists")
    all_node=read_all_node_txt(all_node_save_path)
    all_template=read_all_template_txt(all_template_save_path)
    TF_model=torch.load(transformer_model_save_path)

else:
    print("no model start train vit and get all_node all_template")
    all_node=[]
    all_template=[]
    trans_data=[]
    trans_label=[]
    for i in range(website_num):
        train_array,all_node1=generate_array_by_fliter_size_with_one_label(train_set,train_label,i,min_confidence,max_support_num_per_sample)
        all_node.append(all_node1)
        word2idx=generate_word2idx(all_node1)
        template,pc=generate_template_from_array_after_filter(train_array,word2idx,assoc_num,mining_threshold)
        all_template.append(template)
        all_event_chain=generate_event_chain_by_template_train(template,train_array,pc,word2idx,chain_len)
        for web_sample in range(len(all_event_chain)):
            vec_list=generate_vec_list_from_event_chain(all_event_chain[web_sample])
            vec_array=vec_list_to_vec_array(vec_list,max_N,max_M)
            trans_data.append(vec_array)
            trans_label.append(i)
    # print(trans_label)

    # 写入all_node和all_template
    f1 = open(all_node_save_path, 'w+', encoding='UTF-8', newline="")
    data_write_node_txt(f1, all_node)
    f1.close()

    f2 = open(all_template_save_path, 'w+', encoding='UTF-8', newline="")
    data_write_template_txt(f2, all_template)
    f2.close()

    trans_data=np.array(trans_data)
    trans_label=np.array(trans_label)
    trans_data=trans_data.reshape((len(trans_data),1,max_N,max_M))
    xtrain=torch.from_numpy(trans_data)
    ytrain=torch.from_numpy(trans_label)
    train_dataset = Data.TensorDataset(xtrain, ytrain)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)
    

    TF_model.to(device)
    loss_history = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(TF_model.parameters(), lr=lr, betas=(0.9, 0.99))
    # Train
    TF_model.train()
    for epoch in range(num_epoch):
        for step, (xtrain, ytrain) in enumerate(train_loader):
            xtrain = Variable(xtrain.float())
            ytrain = Variable(ytrain.long())
            xtrain = xtrain.to(device)
            ytrain = ytrain.to(device)
            # print(xtrain)
            output = TF_model(xtrain)
            loss = criterion (output, ytrain)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [%d/%d] Loss: %.4f'
            % (epoch + 1, num_epoch, loss.data))
        loss_history.append(loss.data)
        if epoch % 5 == 0 and epoch != 0:
            lr = lr * 0.8
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        print('learning rate:',lr)
    torch.save(TF_model,transformer_model_save_path)


# 生成多标签相应测试集
# print(set(test_label))
obverlapping=0.3
data_array=[]
# print("generate multi_data")
label_num=website_num
for i in range(1000):
    data_array.append(openworld_three_tab_random_label_data(test_set,test_label,overlapping))
    # random_number=random.randint(1,4)
    # print(random_number)
    # if(random_number==1):
    #     data_array.append(two_tab_random_label_data(test_set,test_label,overlapping))
    # elif(random_number==2):
    #     data_array.append(three_tab_random_label_data(test_set,test_label,overlapping))
    # elif(random_number==3):
    #     data_array.append(four_tab_random_label_data(test_set,test_label,overlapping)) 
    # elif(random_number==4):
    #     data_array.append(five_tab_random_label_data(test_set,test_label,overlapping))
    # print(data_array[i][-1])
# ytest代表标签1,ytest2代表标签2


# multi_xtest测试集数据包长序列
# new_multi_ylabel存储标签
multi_xtest=[]
multi_ytest1=[]
multi_ytest2=[]
multi_ytest3=[]
multi_ytest4=[]
multi_ytest5=[]
new_multi_ylabel=[]
ture_label=[]
for i in range(len(data_array)):
    list1=[]
    ture_label=[]
    # print(len(data_array[i]))
    for j in range(len(data_array[i])):
        if(j<(len(data_array[i])-5)):
            list1.append(data_array[i][j])
        elif(j==(len(data_array[i]))-5):
            label1=data_array[i][j]
            ture_label.append(label1)
            # print(label1)
        elif(j==(len(data_array[i]))-4):
            label2=data_array[i][j]
            ture_label.append(label2)
        elif(j==(len(data_array[i]))-3):
            label3=data_array[i][j]
            if(label3!=100):
                ture_label.append(label3)
        elif(j==(len(data_array[i]))-2):
            label4=data_array[i][j]
            if(label4!=100):
                ture_label.append(label4)
        elif(j==(len(data_array[i]))-1):
            label5=data_array[i][j]
            if(label5!=100):
                ture_label.append(label5)
    multi_xtest.append(list1)
    multi_ytest1.append(label1)
    multi_ytest2.append(label2)
    multi_ytest3.append(label3)
    multi_ytest4.append(label4)
    multi_ytest5.append(label5)
    new_multi_ylabel.append(ture_label)


# 丢失、错序、重复相关
# probability=0.15
# # 生成相应测试集,每次重新生成
# middle_data_array=[]
# print("generate multi_data")
# label_num=website_num
# # 丢失的对比实验
# new_test_set=[]
# for i in range(len(multi_xtest)):
#     middle_sample=multi_xtest[i]
#     middle_sample=drop_data_randomly(middle_sample, probability)
#     # middle_sample=duplicate_elements(middle_sample, probability)
#     # delete_random_data(middle_sample,probability)
#     # delete_random_data(lst, deletion_ratio)
#     # middle_sample = shuffle_and_replace_subsequence(middle_sample, probability)
#     # delete_random_data(middle_sample, probability)
#     new_test_set.append(middle_sample)
# # new_test_set=np.array(new_test_set)
# multi_xtest=new_test_set



# 随机森林标准化 输出可能的多标签
data_len_3=3006  
multi_test_array_1=np.zeros([len(multi_xtest),data_len_3])

# 多标签随机森林测试
for i in range(len(multi_xtest)):
    for j in range(len(multi_xtest[i])):
        if(multi_xtest[i][j]>=-1500 and multi_xtest[i][j]<=1500):
            multi_test_array_1[i][int(multi_xtest[i][j]+1500)]+=1
    if(i%1000==0):
        print(i)
    multi_test_array_1[i][3001]=multi_ytest1[i]
    multi_test_array_1[i][3002]=multi_ytest2[i]
    multi_test_array_1[i][3003]=multi_ytest3[i]
    multi_test_array_1[i][3004]=multi_ytest4[i]
    multi_test_array_1[i][3005]=multi_ytest5[i]
print(multi_test_array_1)

X_test=multi_test_array_1[:,:-5]
y_test1=multi_test_array_1[:,-5].astype(np.int64)
y_test2=multi_test_array_1[:,-4].astype(np.int64)
y_test3=multi_test_array_1[:,-3].astype(np.int64)
y_test4=multi_test_array_1[:,-2].astype(np.int64)
y_test5=multi_test_array_1[:,-1].astype(np.int64)

test1= X_test
for t in range(1):
    print("---------------Epoch"+str(t)+"----------------")
    print("ACC  TPR/RC  FPR   PR   F1")
    Estimators = RF1.estimators_
    predictions_all=np.array([tree.predict(test1) for tree in Estimators]).astype(np.int64)


# 统计决策树输出的各种标签的数量
# # 选取树最多的5个标签,置于pre_list中 这里我们不用像其他方法一样，在2-tab下依旧可以输出5label，因为我们在后续会对这里输出的标签进行进一步排除
predictions_TS=predictions_all.transpose()
pre_label_num=5
print(predictions_TS.shape)
acc=0
pre_list=[]
for i in range(len(predictions_TS)):
    list1=[]
    tree_count=[]
    count=0
    # print("---------")
    test_array=np.bincount(predictions_TS[i])
    # print(test_array)
    for j in range(pre_label_num):
        label_index=np.argsort(test_array)[-(j+1)]
        # print(label_index)
        if(test_array[label_index]>=1):
            list1.append(label_index)
            tree_count.append(test_array[label_index])
    pre_list.append(list1)


RobustWF_predicted=[]
RobustWF_predicted_2_tab=[]

# 相关测试
for i in range(len(ytest1)):
    if(i%10==0):
        print(i)
    correct=0
    label_predicted=[]
    label_2_tab_predicted=[]
    for j in range(len(pre_list[i])):
        test_all_node=all_node[pre_list[i][j]]
        test_all_template=all_template[pre_list[i][j]]
        test_filter_sample=generate_array_by_fliter_size_with_one_label_test(multi_xtest[i],test_all_node)
        # print(test_filter_sample)
        event_chain=generate_event_chain_test(test_filter_sample,test_all_node,test_all_template,assoc_num,mining_threshold,chain_len)
        # event_chain=generate_event_chain_test(test_filter_sample,test_all_node,test_all_template)
        test_vec_list=generate_vec_list_from_event_chain(event_chain)
        test_vec_array=vec_list_to_vec_array(test_vec_list,max_N,max_M)
        test_trans_data=test_vec_array.reshape((1,1,max_N,max_M))
        xtest=torch.from_numpy(test_trans_data)
        xtest = Variable(xtest.float())
        xtest=xtest.to(device)
        out = TF_model(xtest)
        _, predicted = torch.max(out.data, 1)
        # 检验输出的一致性
        if(int(predicted)==pre_list[i][j]):
            label_predicted.append(int(predicted))
        # print(111)
    # print(label_predicted)
    RobustWF_predicted.append(label_predicted)

# 输出jaccard score
jaccard_sim_num=0
for i in range(len(RobustWF_predicted)):
    jaccard_sim = pak_similarity(RobustWF_predicted[i], new_multi_ylabel[i])
    # print(RobustWF_predicted[i],new_multi_ylabel[i])
    jaccard_sim_num+=jaccard_sim
    # print(jaccard_sim)
print("RobustWF  pre_num=5, The mean jaccard_sim_num is######",jaccard_sim_num/len(RobustWF_predicted))







