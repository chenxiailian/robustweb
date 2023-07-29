import re
import torch
import math
import random
import os
import numpy as np
import pandas as pd

def data_write_node_txt(file, data):
    print("data_write txt.")
    for row in data:
        temp=''
        for item in row:
            # print(item)
            # temp.append(deepcopy(item[1]))
            s = str(item)
            temp = temp + s + ' '
        temp = temp[:-1]
        temp = temp +"\n"
        file.write(temp)
        

def data_write_template_txt(file, data):
    print("data_write txt.")
    for row in data:
        temp=''
        for item in row:
            # print(item)
            # temp.append(deepcopy(item[1]))
            s0 = str(item[0])
            s1 = str(item[1])
            temp = temp + s0 + ' '+s1+" "
        temp = temp[:-1]
        temp = temp +"\n"
        file.write(temp)
        
def read_all_node_txt(file_path):
    test_list = np.array(pd.read_csv(file_path,header=None)).flatten().tolist()
    all_node=[]
    for i in range(len(test_list)):
        list1=[float(j) for j in test_list[i].split(' ')]
        # print(list1)
        all_node.append(list1)
    return(all_node)

def read_all_template_txt(file_path):
    test_list = np.array(pd.read_csv(file_path,header=None)).flatten().tolist()
    print
    test_template_1=[]
    for i in range(len(test_list)):
        list1=[int(j) for j in test_list[i].split(' ')]
        new_list=[]
        for m in range(0,len(list1),2):
            middle_list=[list1[m],list1[m+1]]
            new_list.append(middle_list)
        test_template_1.append(new_list)
    return (test_template_1)    