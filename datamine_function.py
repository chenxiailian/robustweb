import socket
import struct
import os
import numpy as np
import pandas as pd
import sys
import csv
import json
from collections import defaultdict
from copy import deepcopy
from causality_correlation import *

def getborder(y_data,label):
    left=0
    right=len(y_data)-1
    res=len(y_data)
    while(left<=right):
        mid=left+((right-left)>>1)
        if(label<y_data[mid]):
            res=mid
            right=mid-1
        else:
            left=mid+1
    return res

def count_label_boundary(y_data,label):
    right=getborder(y_data,label)-1
    left=getborder(y_data,label-1)
    return(left,right)

def count_the_label_num(x_data,y_data,label):
    count_num=0
    left,right=count_label_boundary(y_data,label)
    count_num=right-left+1
    return(count_num)

# 挖掘频繁项，统计各个数字出现的次数
def get_frequencent_dict(x_data,y_data,label):
    frequence_dict={}
    left,right=count_label_boundary(y_data,label)
    for i in range(left,right+1):
        for j in range(len(x_data[i])):
            if x_data[i][j] not in frequence_dict.keys():
                frequence_dict[x_data[i][j]]=1
            else:
                frequence_dict[x_data[i][j]]+=1
    return(frequence_dict)

# 基于最大最小支持度获得一个节点列表
# 最大最小支持度该如何获得
# 0.1是最小支持度系数
# 20是访问网站主页时，流的总数
def get_fre_dict_after_frequece(x_data,y_data,label,max_support_num_per_sample):
    count_num=count_the_label_num(x_data,y_data,label)
    min_support=int(count_num*0.1)
    max_support=int(count_num*max_support_num_per_sample)
    frequence_dict=get_frequencent_dict(x_data,y_data,label)
    sort_list=sorted(frequence_dict.items(),reverse=True,key=lambda x:x[1])
    middle_dict={}
    for i in range(len(sort_list)):
        if(sort_list[i][1]>min_support and sort_list[i][1]<max_support):
            middle_dict[sort_list[i][0]]=sort_list[i][1]
    # print(middle_dict)
    return middle_dict

# 基于middle_list统计置信度相关的值
def stat_confidence_dict(x_data,y_data,label,min_confidence,max_support_num_per_sample):
    fre_dict=get_fre_dict_after_frequece(x_data,y_data,label,max_support_num_per_sample)
    fre_list=sorted(fre_dict.items(),reverse=True,key=lambda x:x[1])
    # print(fre_list)
    left,right=count_label_boundary(y_data,label)
    confidence_dict={}
    for i in range(len(fre_list)):
        confidence_dict[fre_list[i][0]]=0
    for i in range(len(fre_list)):
        for j in range(left,right+1):
            if fre_list[i][0] in x_data[j]:
                confidence_dict[fre_list[i][0]]+=1
    # print(confidence_dict)
    return(confidence_dict)

# 设置最小置信度，获得最终的序列数字列表
def get_confi_dict_after_confidence(x_data,y_data,label,min_confidence,max_support_num_per_sample):
    fre_dict=get_fre_dict_after_frequece(x_data,y_data,label,max_support_num_per_sample)
    # count_num=count_the_label_num(x_data,y_data,label)
    # min_confidence=0.5
    confidence_dict=stat_confidence_dict(x_data,y_data,label,min_confidence,max_support_num_per_sample)
    confidence_list=sorted(confidence_dict.items(),reverse=True,key=lambda x:x[1])
    middle_dict={}
    for i in range(len(confidence_list)):
        if(confidence_list[i][1]/count_the_label_num(x_data,y_data,label)>min_confidence):
            middle_dict[confidence_list[i][0]]=fre_dict[confidence_list[i][0]]
    return middle_dict

# 获取根节点列表和频繁节点列表
# confidence_dict存放的是选择出来的各个长度的出现次数
def get_root_node(x_data,y_data,label,min_confidence,max_support_num_per_sample):
    # print("enter here")
    confidence_dict=get_confi_dict_after_confidence(x_data,y_data,label,min_confidence,max_support_num_per_sample)
    sort_list=sorted(confidence_dict.items(),reverse=True,key=lambda x:x[1])
    # print(sort_list)
    root_node=[]
    fre_node=[]
    all_node=[]
    root_num=0
    for i in range(len(sort_list)):
        if(root_num<=10):
            if(sort_list[i][0]>200):
                root_node.append(sort_list[i][0])
                root_num+=1
            else:
                fre_node.append(sort_list[i][0])
        else:
            fre_node.append(sort_list[i][0])
        all_node.append(sort_list[i][0])
    # print(root_node)
    # print(fre_node)
    return(root_node,fre_node,all_node)

def get_all_node_list(x_data):
    frequence_dict={}
    for i in range(len(x_data)):
        for j in range(len(x_data[i])):
            if x_data[i][j] not in frequence_dict.keys():
                frequence_dict[x_data[i][j]]=1
            else:
                frequence_dict[x_data[i][j]]+=1
    sort_list=sorted(frequence_dict.items(),reverse=True,key=lambda x:x[1])
    middle_dict={}
    min_support=20
    max_support=10*len(x_data)
    for i in range(len(sort_list)):
        if(sort_list[i][1]>min_support and sort_list[i][1]<max_support):
            middle_dict[sort_list[i][0]]=sort_list[i][1]
    all_node=[]
    all_node.extend(middle_dict)
    return all_node
