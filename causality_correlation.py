import socket
import struct
import os
import numpy as np
import pandas as pd
import sys
import csv
from collections import defaultdict
from copy import deepcopy


# def data_write(file, data):
#     print("data_write.")
#     csv_write = csv.writer(file)
#     for row in data:
#         csv_write.writerow(row)


# def data_write_txt(file, data):
#     print("data_write txt.")
#     for row in data:
#         temp=''
#         for item in row:
#             # print(item)
#             # temp.append(deepcopy(item[1]))
#             s = str(item)
#             temp = temp + s + ' '
#         temp = temp + '\n'
#         file.write(temp)

# def data_write_txt2(file, data):
#     print("data_write txt2.")
#     for row in data:
#         temp=''
#         for item in row:
#             # print(item)
#             # temp.append(deepcopy(item[1]))
#             s = str(item[1])
#             temp = temp + s + ' '
#         temp = temp + '\n'
#         file.write(temp)

def dfs(root, chain, tree):
    nodeSet = []
    stack = []
    temp_chain = []
    nodeSet.append(root)
    stack.append(root)
    # temp_chain.append(root)
    while len(stack) > 0:
        cur = stack.pop()
        if tree[cur]:
            for next in tree[cur]:
                if next not in nodeSet:
                    stack.append(cur)
                    stack.append(next)
                    nodeSet.append(next)
                    break
        else:
            temp_chain = deepcopy(stack)
            temp_chain.append(cur)
            chain.append(deepcopy(temp_chain))
    return chain


# 在超告警对中生成事件链
def dfs_event(root, chain, tree):
    nodeSet = []
    stack = []
    temp_chain = []
    nodeSet.append(root)
    stack.append(root)
    # temp_chain.append(root)
    while len(stack) > 0:
        cur = stack.pop()
        if tree[tuple(cur)]:
            for next in tree[tuple(cur)]:
                if next not in nodeSet:
                    stack.append(cur)
                    stack.append(next)
                    nodeSet.append(next)
                    break
        else:
            temp_chain = deepcopy(stack)
            temp_chain.append(cur)
            chain.append(deepcopy(temp_chain))
    return chain
    
    
# tree_nodeSet是一个dict，key是每一个节点，value是key对应的下一个未访问的节点的集合    
def all_path(root, path, tree, tree_nodeSet,chain_len):
    nodeSet = []
    stack = []   # 存路径上的各个节点
    temp_chain = []
    # nodeSet.append(root)   # 已经访问过的节点
    stack.append(root)
    while len(stack) > 0:
        cur = stack.pop()
        for recover in tree[tuple(cur)]:    # 每次pop当前的尾节点，将该节点指向的所有节点的tree_nodeSet集合恢复出来
            tree_nodeSet[tuple(cur)] =tree[tuple(cur)]
        if tree[tuple(cur)]:
            for next in tree[tuple(cur)]:
                if next in tree_nodeSet[tuple(cur)]:
                    stack.append(cur)
                    stack.append(next)
                    tree_nodeSet[tuple(cur)].remove(next)
                    break
        else:
            temp_chain = deepcopy(stack)     # 在这里其实可以直接判断得到的链是否需要舍弃，减少内存的消耗和内存的拷贝
            temp_chain.append(cur)           # 1、链短；2、如果是做实验，可以将链中包含apt日志较少的链去除。
            if len(temp_chain)>=chain_len:
                path.append(deepcopy(temp_chain))
    return path