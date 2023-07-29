import re
import torch
import math
import numpy as np
import pandas as pd
import random
import os
from os.path import exists


#引入标签 
def generate_two_tab(list1,list2,label1,label2,overlap_ratio1):

    i=0
    j=0
    list4=[]
    count1=0
    count2=0
    new_list1_for=list1[0:int(len(list1)*(1-overlap_ratio1))].tolist()
    new_list1_back=list1[int(len(list1)*(1-overlap_ratio1)):].tolist()
    
    new_list2_for=list2[0:int(len(list2)*overlap_ratio1)].tolist()
    new_list2_back=list2[int(len(list2)*overlap_ratio1):].tolist()
   
    new_list4=np.zeros(len(new_list1_back)+len(new_list2_for))
    while(count1<(len(new_list1_back)) or count2<(len(new_list2_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count1<len(new_list1_back)):
                if(int(new_list1_back[count1])!=0):
                    new_list4[i]=new_list1_back[count1]
                    count1+=1
                    i+=1
                else:
                    count1+=1
        else:
            if(count2<len(new_list2_for)):
                if(int(new_list2_for[count2])!=0):
                    new_list4[i]=new_list2_for[count2]
                    count2+=1
                    i+=1
                else:
                    count2+=1
        
    new_list4=new_list4.tolist()
    # print(len(new_list1_for),len(new_list2_middle),len(new_list3_back),len(new_list4),len(new_list5))
    list4=new_list1_for+new_list4+new_list2_back
    list4.append(label1)
    list4.append(label2)
    list4.append(500)
    list4.append(500)
    list4.append(500)
    # list4.append(100)
    # list4.append(100)
    # list4.append(100)
    # print(len(list3))
    list4=np.array(list4)
    # print(list4)
    return list4

def generate_three_tab(list1,list2,list3,label1,label2,label3,overlap_ratio1,overlap_ratio2):

    i=0
    j=0
    list4=[]
    count1=0
    count2=0
    count3=0
    count4=0
    new_list1_for=list1[0:int(len(list1)*(1-overlap_ratio1))].tolist()
    new_list1_back=list1[int(len(list1)*(1-overlap_ratio1)):].tolist()
    
    new_list2_for=list2[0:int(len(list2)*overlap_ratio1)].tolist()
    new_list2_middle=list2[int(len(list2)*overlap_ratio1):int(len(list2)*(1-overlap_ratio2))].tolist()
    new_list2_back=list2[int(len(list2)*(1-overlap_ratio2)):].tolist()
    
    new_list3_for=list3[0:int(len(list3)*overlap_ratio2)].tolist()
    new_list3_back=list3[int(len(list3)*(overlap_ratio2)):].tolist() 
    
    new_list4=np.zeros(len(new_list1_back)+len(new_list2_for))
    while(count1<(len(new_list1_back)) or count2<(len(new_list2_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count1<len(new_list1_back)):
                if(int(new_list1_back[count1])!=0):
                    new_list4[i]=new_list1_back[count1]
                    count1+=1
                    i+=1
                else:
                    count1+=1
        else:
            if(count2<len(new_list2_for)):
                if(int(new_list2_for[count2])!=0):
                    new_list4[i]=new_list2_for[count2]
                    count2+=1
                    i+=1
                else:
                    count2+=1
                    
    new_list5=np.zeros(len(new_list2_back)+len(new_list3_for))
    while(count3<(len(new_list2_back)) or count4<(len(new_list3_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count3<len(new_list2_back)):
                if(int(new_list2_back[count3])!=0):
                    new_list5[j]=new_list2_back[count3]
                    count3+=1
                    j+=1
                else:
                    count3+=1
        else:
            if(count4<len(new_list3_for)):
                if(int(new_list3_for[count4])!=0):
                    new_list5[j]=new_list3_for[count4]
                    count4+=1
                    j+=1
                else:
                    count4+=1
        
    new_list4=new_list4.tolist()
    new_list5=new_list5.tolist()
    # print(len(new_list1_for),len(new_list2_middle),len(new_list3_back),len(new_list4),len(new_list5))
    list4=new_list1_for+new_list4+new_list2_middle+new_list5+new_list3_back
    list4.append(label1)
    list4.append(label2)
    list4.append(label3)
    # list4.append(100)
    # list4.append(100)
    list4.append(500)
    list4.append(500)
    # print(len(list3))
    list4=np.array(list4)
    # print(list4)
    return list4

def generate_four_tab(list1,list2,list3,list4,label1,label2,label3,label4,overlap_ratio1,overlap_ratio2,overlap_ratio3):

    i=0
    j=0
    k=0
    list5=[]
    count1=0
    count2=0
    count3=0
    count4=0
    count5=0
    count6=0
    new_list1_for=list1[0:int(len(list1)*(1-overlap_ratio1))].tolist()
    new_list1_back=list1[int(len(list1)*(1-overlap_ratio1)):].tolist()
    
    new_list2_for=list2[0:int(len(list2)*overlap_ratio1)].tolist()
    new_list2_middle=list2[int(len(list2)*overlap_ratio1):int(len(list2)*(1-overlap_ratio2))].tolist()
    new_list2_back=list2[int(len(list2)*(1-overlap_ratio2)):].tolist()
    
    new_list3_for=list3[0:int(len(list3)*overlap_ratio2)].tolist()
    new_list3_middle=list3[int(len(list3)*overlap_ratio2):int(len(list3)*(1-overlap_ratio3))].tolist()
    new_list3_back=list3[int(len(list3)*(1-overlap_ratio3)):].tolist() 
    
    new_list4_for=list4[0:int(len(list4)*overlap_ratio3)].tolist()
    new_list4_back=list4[int(len(list4)*overlap_ratio3):].tolist()
    
    new_list5=np.zeros(len(new_list1_back)+len(new_list2_for))
    while(count1<(len(new_list1_back)) or count2<(len(new_list2_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count1<len(new_list1_back)):
                if(int(new_list1_back[count1])!=0):
                    new_list5[i]=new_list1_back[count1]
                    count1+=1
                    i+=1
                else:
                    count1+=1
        else:
            if(count2<len(new_list2_for)):
                if(int(new_list2_for[count2])!=0):
                    new_list5[i]=new_list2_for[count2]
                    count2+=1
                    i+=1
                else:
                    count2+=1
                    
    new_list6=np.zeros(len(new_list2_back)+len(new_list3_for))
    while(count3<(len(new_list2_back)) or count4<(len(new_list3_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count3<len(new_list2_back)):
                if(int(new_list2_back[count3])!=0):
                    new_list6[j]=new_list2_back[count3]
                    count3+=1
                    j+=1
                else:
                    count3+=1
        else:
            if(count4<len(new_list3_for)):
                if(int(new_list3_for[count4])!=0):
                    new_list6[j]=new_list3_for[count4]
                    count4+=1
                    j+=1
                else:
                    count4+=1
                    
    new_list7=np.zeros(len(new_list3_back)+len(new_list4_for))
    while(count5<(len(new_list3_back)) or count6<(len(new_list4_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count5<len(new_list3_back)):
                if(int(new_list3_back[count5])!=0):
                    new_list7[k]=new_list3_back[count5]
                    count5+=1
                    k+=1
                else:
                    count5+=1
        else:
            if(count6<len(new_list4_for)):
                if(int(new_list4_for[count6])!=0):
                    new_list7[k]=new_list4_for[count6]
                    count6+=1
                    k+=1
                else:
                    count6+=1
        
    new_list5=new_list5.tolist()
    new_list6=new_list6.tolist()
    new_list7=new_list7.tolist()
    # print(len(new_list1_for),len(new_list2_middle),len(new_list3_back),len(new_list4),len(new_list5))
    list5=new_list1_for+new_list5+new_list2_middle+new_list6+new_list3_middle+new_list7+new_list4_back
    list5.append(label1)
    list5.append(label2)
    list5.append(label3)
    list5.append(label4)
    list5.append(500)
    # list5.append(100)
    # print(list5)
    # print(len(list3))
    list5=np.array(list5)
    return list5

# 三标签的暂时没有加入重叠度的概念

def generate_five_tab(list1,list2,list3,list4,list5,label1,label2,label3,label4,label5,overlap_ratio1,overlap_ratio2,overlap_ratio3,overlap_ratio4):

    i=0
    j=0
    k=0
    l=0
    list6=[]
    count1=0
    count2=0
    count3=0
    count4=0
    count5=0
    count6=0
    count7=0
    count8=0
    
    new_list1_for=list1[0:int(len(list1)*(1-overlap_ratio1))].tolist()
    new_list1_back=list1[int(len(list1)*(1-overlap_ratio1)):].tolist()
    
    new_list2_for=list2[0:int(len(list2)*overlap_ratio1)].tolist()
    new_list2_middle=list2[int(len(list2)*overlap_ratio1):int(len(list2)*(1-overlap_ratio2))].tolist()
    new_list2_back=list2[int(len(list2)*(1-overlap_ratio2)):].tolist()
    
    new_list3_for=list3[0:int(len(list3)*overlap_ratio2)].tolist()
    new_list3_middle=list3[int(len(list3)*overlap_ratio2):int(len(list3)*(1-overlap_ratio3))].tolist()
    new_list3_back=list3[int(len(list3)*(1-overlap_ratio3)):].tolist()
    
    new_list4_for=list4[0:int(len(list4)*overlap_ratio3)].tolist()
    new_list4_middle=list4[int(len(list4)*overlap_ratio3):int(len(list4)*(1-overlap_ratio3))].tolist()
    new_list4_back=list4[int(len(list4)*(1-overlap_ratio4)):].tolist() 
    
    new_list5_for=list5[0:int(len(list4)*overlap_ratio3)].tolist()
    new_list5_back=list5[int(len(list4)*overlap_ratio3):].tolist()
    
    new_list5=np.zeros(len(new_list1_back)+len(new_list2_for))
    while(count1<(len(new_list1_back)) or count2<(len(new_list2_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count1<len(new_list1_back)):
                if(int(new_list1_back[count1])!=0):
                    new_list5[i]=new_list1_back[count1]
                    count1+=1
                    i+=1
                else:
                    count1+=1
        else:
            if(count2<len(new_list2_for)):
                if(int(new_list2_for[count2])!=0):
                    new_list5[i]=new_list2_for[count2]
                    count2+=1
                    i+=1
                else:
                    count2+=1
                    
    new_list6=np.zeros(len(new_list2_back)+len(new_list3_for))
    while(count3<(len(new_list2_back)) or count4<(len(new_list3_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count3<len(new_list2_back)):
                if(int(new_list2_back[count3])!=0):
                    new_list6[j]=new_list2_back[count3]
                    count3+=1
                    j+=1
                else:
                    count3+=1
        else:
            if(count4<len(new_list3_for)):
                if(int(new_list3_for[count4])!=0):
                    new_list6[j]=new_list3_for[count4]
                    count4+=1
                    j+=1
                else:
                    count4+=1
                    
    new_list7=np.zeros(len(new_list3_back)+len(new_list4_for))
    while(count5<(len(new_list3_back)) or count6<(len(new_list4_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count5<len(new_list3_back)):
                if(int(new_list3_back[count5])!=0):
                    new_list7[k]=new_list3_back[count5]
                    count5+=1
                    k+=1
                else:
                    count5+=1
        else:
            if(count6<len(new_list4_for)):
                if(int(new_list4_for[count6])!=0):
                    new_list7[k]=new_list4_for[count6]
                    count6+=1
                    k+=1
                else:
                    count6+=1
                    
    new_list8=np.zeros(len(new_list4_back)+len(new_list5_for))
    while(count7<(len(new_list4_back)) or count8<(len(new_list5_for))):
        a=random.randint(1,2)
        if(a%2==0):
            if(count7<len(new_list4_back)):
                if(int(new_list4_back[count7])!=0):
                    new_list8[l]=new_list4_back[count7]
                    count7+=1
                    l+=1
                else:
                    count7+=1
        else:
            if(count8<len(new_list5_for)):
                if(int(new_list5_for[count8])!=0):
                    new_list8[l]=new_list5_for[count8]
                    count8+=1
                    l+=1
                else:
                    count8+=1
        
    new_list5=new_list5.tolist()
    new_list6=new_list6.tolist()
    new_list7=new_list7.tolist()
    new_list8=new_list8.tolist()
    # print(len(new_list1_for),len(new_list2_middle),len(new_list3_back),len(new_list4),len(new_list5))
    list6=new_list1_for+new_list5+new_list2_middle+new_list6+new_list3_middle+new_list7+new_list4_middle+new_list8+new_list5_back
    list6.append(label1)
    list6.append(label2)
    list6.append(label3)
    list6.append(label4)
    list6.append(label5)
    # print(len(list3))
    list6=np.array(list6)
    return list6

def two_tab_random_label_data(x_data,y_data,overlap_ratio1):
    label1, label2 = random.sample(set(y_data),2)
    # print(label1,label2)
    data1 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label1])
    data2 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label2])
    new_multi_label_data=generate_two_tab(data1,data2,label1,label2,overlap_ratio1)
    return new_multi_label_data

def three_tab_random_label_data(x_data,y_data,overlap_ratio1):
    label1, label2,label3 = random.sample(set(y_data),3)
    # print(label1,label2)
    data1 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label1])
    data2 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label2])
    data3 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label3])
    new_multi_label_data=generate_three_tab(data1,data2,data3,label1,label2,label3,overlap_ratio1,overlap_ratio1)
    return new_multi_label_data

def four_tab_random_label_data(x_data,y_data,overlap_ratio1):
    label1, label2,label3,label4 = random.sample(set(y_data),4)
    # print(label1,label2)
    data1 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label1])
    data2 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label2])
    data3 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label3])
    data4 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label4])
    new_multi_label_data=generate_four_tab(data1,data2,data3,data4,label1,label2,label3,label4,overlap_ratio1,overlap_ratio1,overlap_ratio1)
    return new_multi_label_data

def five_tab_random_label_data(x_data,y_data,overlap_ratio1):
    label1, label2,label3,label4,label5 = random.sample(set(y_data),5)
    # print(label1,label2)
    data1 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label1])
    data2 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label2])
    data3 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label3])
    data4 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label4])
    data5 = random.choice([x_data[i] for i, l in enumerate(y_data) if l == label5])
    new_multi_label_data=generate_five_tab(data1,data2,data3,data4,data5,label1,label2,label3,label4,label5,overlap_ratio1,overlap_ratio1,overlap_ratio1,overlap_ratio1)
    return new_multi_label_data
