import json
from collections import defaultdict
from copy import deepcopy
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.ensemble import RandomForestClassifier
from datamine_function import *
from causality_correlation import *
import re
import torch
import math
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset,DataLoader
import os
from os.path import exists

def generate_distribution_3000_with_one_label(dataset,label_list):
    data_len_1=3002  #1对应3001维的特征向量
    data_array=np.zeros([len(dataset),data_len_1])
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if(dataset[i][j]>=-1500 and dataset[i][j]<=1500):
                data_array[i][int(dataset[i][j]+1500)]+=1
        if(i/2000==0):
            print(i)
        data_array[i][3001]=label_list[i]
    return(data_array)

def get_data_from_label_and_index(dataset,label_list,label):
    new_data=[]
    new_list=[]
    for i in range(len(label_list)):
        if(label_list[i]==label):
            new_data.append(dataset[i])
            new_list.append(label_list[i])
    new_data=np.array(new_data)
    return new_data,new_list
# 因果关系挖掘
def generate_array_by_fliter_size_with_one_label(dataset,label_list,label,min_confidence,max_support_num_per_sample):
    middle_array,middle_label_list=get_data_from_label_and_index(dataset,label_list,label)
    root_node1,fre_node1,all_node1=get_root_node(middle_array,middle_label_list,label,min_confidence,max_support_num_per_sample)
    train_array=[]
    for row in range(len(middle_array)):
        new_array=[]
        for col in range(len(middle_array[row])):
            if(middle_array[row][col] in all_node1):
                new_array.append(middle_array[row][col])
        # print(len(middle_array))
        train_array.append(new_array)
    return train_array,all_node1

def generate_word2idx(all_node1):
    word_list=list(set(all_node1))
    word2idx={}
    for i,w in enumerate(word_list):
        word2idx[w]=i
    return word2idx

def generate_template_from_array_after_filter(train_array,word2idx,M,mining_threshold):
    N = len(word2idx)     # 告警事件类型的个数
    # M = 4       # 需要往后面关联的事件的个数,待调整 1-5
    # mining_threshold=-0.5
    pc=[]
    count = np.mat(np.zeros((N, N)))
    result = np.mat(np.zeros((N, N)))
    sum_column = np.mat(np.zeros((N, 1)))
    for row in range(len(train_array)):
        middle_pc=[]
        # print(test_array[i])
        for col in range(0, len(train_array[row])- M -1):
        # for i in range(0, 20):
            for k in range(0, M):       # range(0, 12)表示范围从0到11这12个数，不包括12
                temp = [[col,train_array[row][col]],[col+k,train_array[row][col+k]]]  # temp的形式[[time, ip, event_id, enent, type],[time, ip, event_id, enent, type]]
                # print(temp)
                # print(temp)
                x = word2idx[temp[0][1]]      # 事件id是从1开始编号的，而矩阵是从0开始编号的，所以要减1
                y = word2idx[temp[1][1]]
                if(x!=y):
                    count[x, y] = count[x, y]+1
                    middle_pc.append(temp)

        for col in range(len(train_array[row])- M -1, len(train_array[row]) - 1):
            for k in range(0, len(train_array[row]) - 1 - col):
                temp = [[col,train_array[row][col]],[col+k,train_array[row][col+k]]]  # temp的形式[[time, ip, event_id, type],[time, ip, event_id, type]]
                # print(temp[0][2], temp[1][2])
                x = word2idx[temp[0][1]]  # 事件id是从1开始编号的，而矩阵是从0开始编号的，所以要减1，x为因，y为果
                y = word2idx[temp[1][1]]
                if(x!=y):
                    count[x, y] = count[x, y] + 1
                    middle_pc.append(temp)   # pc里面的事件ID也是从1开始的
        pc.append(middle_pc)
            
    print("pc len is",len(pc))
    # print(count)
    num = 0
    template = []    # 因果关联模板对
    for row in range(0, N):
        for col in range(0, N):
            sum_column[row, 0] = sum_column[row, 0] + count[row, col]  # 计算每行元素之和
    for row in range(0, N):
        for col in range(0, N):
            result[row, col] = (count[row, col] - (sum_column[row, 0]-count[row, col]))/sum_column[row, 0]
            if result[row, col] >mining_threshold and row!=col :
            # if count[i,j]>0.5*len(train_array):
                template.append([row, col])  # 把频率较大的模板加进去,模板里面的事件是从1开始的
                num = num+1
    print("template num is",num)
    # print(template) 
    #结果需要展示吗     
    # result_f = result.reshape(N*N, order='F').tolist()[0]  # 把矩阵拉成列表，然后按列读取（order='F'）变换成一个列表（list）
    # # result_f
    # result_f.sort(reverse=True)  # 排序默认False表示升序，True表示降序
    return template,pc

def generate_event_chain_by_template_train(template,train_array,pc,word2idx,chain_len):
    all_event_chain=[]
    for website_sample in range(len(train_array)):
        # print(pc[i])
        hyper_alert = []
        for website_sample_seq in range(0,len(pc[website_sample])):
            x = pc[website_sample][website_sample_seq][0][1]
            x_index=pc[website_sample][website_sample_seq][0][0]
            y = pc[website_sample][website_sample_seq][1][1]
            y_index = pc[website_sample][website_sample_seq][1][0]
            temp = [word2idx[x], word2idx[y]]    # 事件x和y都是从1开始编号的
            if temp in template:
                hyper_alert.append([[x_index,x],[y_index,y]])
        # print("##############",hyper_alert)


        template_flag = [0] * len(template)
        # print(len(template_flag))
        template_chain = []
        event_tree=[]
        event_tree = defaultdict(list)
        event_node = []      # 用来表征所有非叶子节点的节点

        for i in range(len(hyper_alert)):
            # print(hyper_alert[i])
            # print(event_tree)
            event_tree[tuple(hyper_alert[i][0])].append(deepcopy(hyper_alert[i][1]))    # 把列表的链式关系变成字典的链式关系
            if hyper_alert[i][0] not in event_node:
                event_node.append(hyper_alert[i][0])     # 获取所有非叶子节点的节点
        
        # print("----------",event_node)

        index = [0] * len(event_node)  # 用来表征哪些点是根节点
        for key in event_tree:
            for j in range(len(event_tree[key])):  # 遍历每个key的value，也就是每个父节点的子节点
                if event_tree[key][j] in event_node:
                    index[event_node.index(event_tree[key][j])] = 1    # index的值为1表示对应的template_node不是根节点
        # print("len of index is ", len(index))
        # print(index)

        # print("++++++++++++",event_tree)

        # 遍历所有index为0的点，即根节点，生成相应的事件链
        event_chain = []
        graph_nodeSet = deepcopy(event_tree)
        for i in range(len(index)):
            if index[i] == 0:
                root = event_node[i]
                # print(i," ",root)
                graph_nodeSet = deepcopy(event_tree)
                event_chain = all_path(root, event_chain, event_tree, graph_nodeSet,chain_len)
        all_event_chain.append(event_chain)
        # print("event_chain is",event_chain)
    return all_event_chain

def data_write_txt(file, data):
    # print("data_write txt.")
    for row in data:
        # print(row)
        temp=''
        for item in row:
            # print(item)
            # temp.append(deepcopy(item[1]))
            s = str(int(item[1]))
            temp = temp + s + ' '
        temp = temp + '\n'
        # print(temp)
        file.write(temp)

def txttoarray(filepath,row_N,col_M):
    vec=np.zeros([row_N,col_M])
    try:
        test_list = np.array(pd.read_csv(filepath,header=None)).flatten().tolist()
    except:
        print("enter here?")
        return(vec)
    a=[]
    if(len(test_list)>=row_N):
        for i in range(row_N):
            list1=test_list[i].split(" ")
            del list1[len(list1)-1]
            if(len(list1)<=col_M):
                for j in range(len(list1)):
                    vec[i][j]=list1[j]
            else:
                for j in range(col_M):
                    vec[i][j]=list1[j]
    else:
        for i in range(len(test_list)):
            list1=test_list[i].split(" ")
            del list1[len(list1)-1]
            if(len(list1)<=col_M):
                for j in range(len(list1)):
                    vec[i][j]=list1[j]
            else:
                for j in range(col_M):
                    vec[i][j]=list1[j]        
    return(vec)

# 这里的event——chain，对应着一个label中的一个样本
# 生成N*M的特征向量
def generate_vec_list_from_event_chain(event_chain):
    vec_list=[]
    for i in range(len(event_chain)):
        middle_list=[]
        for j in range(len(event_chain[i])):
            middle_list.append(int(event_chain[i][j][1]))
        # print(middle_list)
        vec_list.append(middle_list)
    return vec_list
def vec_list_to_vec_array(vec_list,row_N,col_M):
    vec_array=np.zeros([row_N,col_M])
    if(len(vec_list)>=row_N):
        for i in range(row_N):
            if(len(vec_list[i])<=col_M):
                for j in range(len(vec_list[i])):
                    vec_array[i][j]=vec_list[i][j]
            else:
                for j in range(col_M):
                    vec_array[i][j]=vec_list[i][j]
    else:
        for i in range(len(vec_list)):
            if(len(vec_list[i])<=col_M):
                for j in range(len(vec_list[i])):
                    vec_array[i][j]=vec_list[i][j]
            else:
                for j in range(col_M):
                    vec_array[i][j]=vec_list[i][j]     
    return(vec_array)

# 测试集中随机森林+因果链的挖掘
def generate_distribution_3000_with_one_label_test(test_sample,test_sample_label):
    data_len_1=3002  #1对应3001维的特征向量
    data_array=np.zeros([1,data_len_1])
    for i in range(len(test_sample)):
        if(test_sample[i]>=-1500 and test_sample[i]<=1500):
            data_array[0][int(test_sample[i]+1500)]+=1
    data_array[0][3001]=test_sample_label
    return(data_array)

# 因果关系挖掘
def generate_array_by_fliter_size_with_one_label_test(test_sample,all_node):
    data_array=[]
    for i in range(len(test_sample)):
        if(test_sample[i] in all_node):
            data_array.append(test_sample[i])
    return data_array

# 因果链生成
def generate_event_chain_test(test_filter_sample,all_node,template,M,mining_threshold,chain_len):
    # print(template)
    # print(len(template))
    word2idx=generate_word2idx(all_node)
    N = len(word2idx)     # 告警事件类型的个数
    # M = 4       # 需要往后面关联的事件的个数,待调整 1-5
    # mining_threshold=-0.5
    pc=[]
    hyper_alert = []
    for i in range(0,len(test_filter_sample)-M-1):
        for j in range(0,M):
            temp = [[i,test_filter_sample[i]],[i+j,test_filter_sample[i+j]]]
            # print(temp[0][1],temp[1][1])
            # print([word2idx[temp[0][1]],word2idx[temp[1][1]]])
            if([word2idx[temp[0][1]],word2idx[temp[1][1]]] in template):
                hyper_alert.append(temp)
    for i in range(len(test_filter_sample)-M-1,len(test_filter_sample)-1):
        for j in range(0,len(test_filter_sample) - i -1):
            temp = [[i,test_filter_sample[i]],[i+j,test_filter_sample[i+j]]]
            if([word2idx[temp[0][1]],word2idx[temp[1][1]]] in template):
                hyper_alert.append(temp)
    # print(hyper_alert)
    template_flag = [0] * len(template)
    template_chain = []
    event_tree=[]
    event_tree = defaultdict(list)
    event_node = []      # 用来表征所有非叶子节点的节点

    for i in range(len(hyper_alert)):
        # print(hyper_alert[i])
        # print(event_tree)
        event_tree[tuple(hyper_alert[i][0])].append(deepcopy(hyper_alert[i][1]))    # 把列表的链式关系变成字典的链式关系
        if hyper_alert[i][0] not in event_node:
            event_node.append(hyper_alert[i][0])     # 获取所有非叶子节点的节点

    # print("----------",event_node)

    index = [0] * len(event_node)  # 用来表征哪些点是根节点
    for key in event_tree:
        for j in range(len(event_tree[key])):  # 遍历每个key的value，也就是每个父节点的子节点
            if event_tree[key][j] in event_node:
                index[event_node.index(event_tree[key][j])] = 1    # index的值为1表示对应的template_node不是根节点
    # print("len of index is ", len(index))
    # print(index)

    # print("++++++++++++",event_tree)

    # 遍历所有index为0的点，即根节点，生成相应的事件链
    event_chain = []
    graph_nodeSet = deepcopy(event_tree)
    for i in range(len(index)):
        if index[i] == 0:
            root = event_node[i]
            # print(i," ",root)
            graph_nodeSet = deepcopy(event_tree)
            event_chain = all_path(root, event_chain, event_tree, graph_nodeSet,chain_len)
    # print(event_chain)
    return(event_chain)
