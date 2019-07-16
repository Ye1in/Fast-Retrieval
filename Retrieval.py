import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import time
import shutil
from PIL import Image,ImageTk
import heapq
import re
import tkinter as tk
from  tkinter.filedialog import askopenfilename
from collections import Counter
import  math
from collections import defaultdict
import random

caffe_root = '/home/yl/yls_File/caffe-master/'
input_image_file = ''
test_dataset_label = caffe_root + 'data/cifar10-dataset/val.txt'
net_file=caffe_root + 'examples/cvprw15-cifar10/h=48/KevinNet_CIFAR10_48_deploy.prototxt'
caffe_model=caffe_root + 'examples/cvprw15-cifar10/h=48/KevinNet_CIFAR10_48.caffemodel'
vector_48_file = caffe_root + 'examples/cvprw15-cifar10/h=48/48-bits binary vector.txt'
vector_4096_file = caffe_root + 'examples/cvprw15-cifar10/h=48/4096-bits image vector.txt'
vector_48 = []
vector_4096 = []
name_list = []
type_list = []
vector_48_list = []
vector_4096_list = []

threshold_value = 8
pool = []
top = 20
Euclidean_distance_list = []
last_name = []
last_type = []
last_index = []
data = None
type = ''

exclusive_OR_list = []
exclusive_OR_statistics_list = []
top_pool_list = []
top_pool_name = []
top_pool_type = []

total_acc1 = 0
total_acc2 = 0
total_acc3 = 0
total_DCG1 = 0
total_DCG2 = 0
total_DCG3 = 0

total_top_acc = []

def matrix_to_binary(M,digits = 0):
    # var = []
    for i in range(len(M)):
        # for col in range(len(M[row])):
            M[i] = round(M[i],digits)
            # if(M[row][col] >= 0.5)
            #     var.insert(1)
            # else:
            #     var.insert(0)
    return M

def extra_binary_vector():
    net = caffe.Net(net_file,caffe_model,caffe.TEST)
    layer = 'fc8_kevin_encode' #提取48位二进制向量
    layer2 = 'fc7' #提取fc7层的特征
    if layer2 not in net.blobs:
        raise TypeError("Invalid layer name: " + layer2)
    if layer not in net.blobs:
        raise TypeError("Invalid layer name: " + layer)
    imagemean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1,3,227,227)
    # caffe.set_device(0)
    caffe.set_mode_cpu()
    img = caffe.io.load_image(input_image_file)
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    net.forward()
    feature_data = net.blobs[layer].data[0]
    feature_data = (feature_data - feature_data.min())/(feature_data.max() - feature_data.min())
    feature_data = matrix_to_binary(feature_data, 0)
    feature_data2 = net.blobs[layer2].data[0]
    feature_data2 = (feature_data2 - feature_data2.min())/(feature_data2.max() - feature_data2.min())
    output_file = caffe_root + 'examples/cvprw15-cifar10/image output' + '.txt'
    with open(output_file, 'w') as f:
        f.write(os.path.basename(input_image_file) + ':\n')
        np.savetxt(f, feature_data, fmt='%.0f', delimiter='\t', newline='')
        f.write('\n')
        np.savetxt(f, feature_data2, fmt='%.2f/', delimiter='\t', newline='')
    f.close()
    # vector_48.extend(feature_data.tolist())
    # vector_4096.extend(feature_data2.tolist())
    result_data = open(output_file,'r')
    result_data.seek(0)
    result_data.readline()
    result = result_data.readline()
    result = result[:-1]
    vector_48.clear()
    vector_48.append(result)
    result = result_data.readline()
    result = result[:-1]
    vector_4096_temp = result.split('/')
    vector_4096.clear()
    vector_4096.extend(vector_4096_temp)
    result_data.close()

def retrueval_from_txt():
    data_48 = open(vector_48_file,'r')
    data_4096 = open(vector_4096_file,'r') #默认为只读打开
    data_48.seek(0)
    data_4096.seek(0)
    txt = ' '
    vector_48_list.clear()
    name_list.clear()
    while txt:             #直到读取完文件
        txt = data_48.readline()  #读取一行文件,包括换行符
        txt = txt[:-2]     #去掉换行符与:
        name_list.append(txt)
        txt = data_48.readline()
        txt = txt[:-1]  # 去掉换行符
        vector_48_list.append(txt)
    data_48.close()
    txt = ' '
    vector_4096_list.clear()
    while txt:
        txt = data_4096.readline()
        txt = data_4096.readline()
        txt = txt[:-2]
        vector_4096_temp = txt.split('/')
        vector_4096_list.append(vector_4096_temp)
    name_list.pop(-1)
    vector_48_list.pop(-1)
    vector_4096_list.pop(-1)
    data_4096.close()

def xor_creat_pool():
    exclusive_OR_list.clear()
    exclusive_OR_statistics_list.clear()
    pool.clear()
    for i in range(len(vector_48_list)):
        temp = bin(int(vector_48_list[i],2) ^ int(vector_48[0],2))
        temp = temp[2:]
        exclusive_OR_list.append(temp.zfill(48))
        C = Counter(exclusive_OR_list[i])
        exclusive_OR_statistics_list.append(C['1'])
        if(exclusive_OR_statistics_list[i] <= threshold_value):
            pool.append(i)
    top_pool_set = set(heapq.nsmallest(top,exclusive_OR_statistics_list))
    top_pool_list.clear()
    for i in range(len(top_pool_set)):
        value = top_pool_set.pop()
        top_pool_list.extend([x for x,y in enumerate(exclusive_OR_statistics_list) if y == value])
    top_pool_name.clear()
    top_pool_name.extend(list(map(lambda x:name_list[x],top_pool_list)))
    test_dataset_label_data = open(test_dataset_label, 'r')
    test_dataset_label_data.seek(0)
    global data
    data = test_dataset_label_data.read()
    test_dataset_label_data.close()
    type_list.clear()
    for i in range(10000):
        reg = "/" + name_list[i] + "\s\d"
        reg_result = re.findall(reg, data)
        type_list.extend(reg_result[0][-1:])
    top_pool_type.clear()
    top_pool_type.extend(list(map(lambda x:type_list[x],top_pool_list)))
    # top_pool_type.extend([])
    # for n in range(len(top_pool_name)):
    #     reg = "/" + top_pool_name[n] + "[\S\s]{2}"
    #     lstr = re.findall(reg, data)
    #     str = lstr[0]
    #     pos = str.rfind(" ")
    #     str = str[pos + 1:]
    #     top_pool_type.extend(str)

def Euclidean_distance():
    Euclidean_distance_list.clear()
    for n in range(len(pool)):
        sum = 0
        index = pool[n]
        for length in range(len(vector_4096_list[0])):
            sum += abs(pow(float(vector_4096_list[index][length]),2) - pow(float(vector_4096[length]),2)) ** 0.5
        Euclidean_distance_list.append(sum)
    global last_index
    last_index.clear()
    last_index = list(map(Euclidean_distance_list.index, heapq.nsmallest(top, Euclidean_distance_list)))
    last_index = list(map(lambda x:pool[x], last_index))
    last_name.clear()
    last_name.extend(list(map(lambda x:name_list[x], last_index)))
    last_type.clear()
    last_type.extend(list(map(lambda x: type_list[x], last_index)))

    # test_dataset_label_data = open(test_dataset_label, 'r')
    # test_dataset_label_data.seek(0)
    # data = test_dataset_label_data.read()
    # global data
    # for n in range(len(last_name)):
    #     reg = "/" + last_name[n] + "[\S\s]{2}"
    #     lstr = re.findall(reg,data)
    #     str = lstr[0]
    #     pos = str.rfind(" ")
    #     str = str[pos+1:]
    #     last_type.extend(str)

def main_window():
    window = tk.Tk()
    window.title('以图搜图')
    window.geometry('%dx%d'%(window.winfo_screenwidth(),window.winfo_screenheight()))
    lable_frame = tk.Frame(window , bd = 2 )
    lable_frame.grid(row = 0 , column = 0 , stick = tk.W)
    E = tk.Entry(lable_frame , textvariable = input_image_file , width = 65 , bd = 3 , cursor = 'arrow')
    E.grid(row = 0 , column = 0 , stick = tk.E)
    # E.place(relx = 0.1, rely = 0.02 , relwidth = 0.5 , relheight = 0.05)
    L = tk.Label(lable_frame, text='类型', width=30, bd=3)
    B = tk.Button(lable_frame , text = '选择图片', width = 30 , bd = 3 , command = lambda:image_window(E,window,L))
    de = tk.Button(lable_frame , text = '类型分析', width = 20 , bd = 3 , command = lambda:detailed())
    de2 =  tk.Button(lable_frame , text = '分析', width = 20 , bd = 3 , command = lambda:analyzes())
    B.grid(row = 0 , column = 1)
    de.grid(row = 0 , column = 2)
    de2.grid(row = 0 , column = 3)
    L.grid(row = 0 , column = 4)
    # B.place(relx = 0.7, rely = 0.02 , relwidth = 0.2 , relheight = 0.05)
    window.mainloop()

def analyzes():
    analyze(10)
    analyze(20)
    analyze(30)
    analyze(40)
    analyze(50)
    analyze(60)
    analyze(70)
    analyze(80)
    analyze(90)
    analyze(100)


    poi = np.arange(10,101,10)
    fig = plt.figure(figsize=(6, 3))
    # plt.rcParams['font.sas-serig'] = ['SimHei']
    plt.xlabel('TopN')
    plt.ylabel('ACC')
    plt.title('ACC - TopN -Filter')
    plt.plot(poi, total_top_acc[0:len(total_top_acc)], c='red',label = 'ACC')
    plt.legend()
    plt.show()

def analyze(length = 20):
    d = defaultdict(list)
    for v, i in [(v, i) for i, v in enumerate(type_list)]:
        d[v].append(i)
    total_top_list = []
    for i in range(len(d)):
       randoms = set()
       while len(randoms) < length:
           randoms.add(random.randint(0,999))
       total_top_list.extend(list(map(lambda x: d[str(i)][x], randoms)))
    vector_48_list_now = list(map(lambda x:vector_48_list[x],total_top_list))
    vector_4096_list_now = list(map(lambda x:vector_4096_list[x],total_top_list))
    type_list_now = list(map(lambda x:type_list[x],total_top_list))
    total_xor_num_list = []
    for i in range(len(total_top_list)):
        xor_num_list = []
        for j in range(len(total_top_list)):
            temp = bin(int(vector_48_list[total_top_list[i]], 2) ^ int(vector_48_list_now[j], 2))
            temp = temp[2:]
            xor_num_list.append(Counter(temp)['1'])
        total_xor_num_list.append(xor_num_list)
    total_pool_list = []
    for n in range(len(total_xor_num_list)):
        pool_set = set(heapq.nsmallest(length, total_xor_num_list[n]))
        pool_list_now = []
        for i in range(len(pool_set)):
            value = pool_set.pop()
            pool_list_now.extend([x for x, y in enumerate(total_xor_num_list[n]) if y == value])
        total_pool_list.append(pool_list_now)

    tatal_Euclidean_list = []
    for i in range(len(total_pool_list)):
        Euclidean_list_now = []
        for j in range(length):
            sum = 0
            for m in range(len(vector_4096_list[0])):
                sum += abs(pow(float(vector_4096_list_now[total_pool_list[i][j]][m]), 2) - pow(float(vector_4096_list_now[i][m]), 2)) ** 0.5
            Euclidean_list_now.append(sum)
        tatal_Euclidean_list.append(Euclidean_list_now)

    total_type_list = []
    for i in range(len(total_pool_list)):
        last_type_list_now = list(map(lambda x: type_list_now[x], total_pool_list[i]))
        total_type_list.append(last_type_list_now)

    # Counter_list = []
    # for i in range(len(total_type_list)):
    #     Counter_list.append(Counter(total_type_list[i]))

    Confusion_matrix = []
    for i in range(int((len(total_top_list))/length)):
        # Confusion_matrix_part = []
        sum0 = 0
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        sum5 = 0
        sum6 = 0
        sum7 = 0
        sum8 = 0
        sum9 = 0
        for j in range(length*i,length*(i+1)):
            C = Counter(total_type_list[j][0:length])
            sum0 +=C['0']
            sum1 +=C['1']
            sum2 +=C['2']
            sum3 +=C['3']
            sum4 +=C['4']
            sum5 +=C['5']
            sum6 +=C['6']
            sum7 +=C['7']
            sum8 +=C['8']
            sum9 +=C['9']
        Confusion_matrix.append([sum0/(length**2), sum1/(length**2), sum2/(length**2), sum3/(length**2), sum4/(length**2),
                                 sum5/(length**2), sum6/(length**2), sum7/(length**2), sum8/(length**2), sum9/(length**2)])
    acc = (Confusion_matrix[0][0] + Confusion_matrix[1][1] + Confusion_matrix[2][2] + Confusion_matrix[3][3] + Confusion_matrix[4][4] + Confusion_matrix[5][5]+ Confusion_matrix[6][6] + Confusion_matrix[7][7] + Confusion_matrix[8][8] + Confusion_matrix[9][9])/10
    total_top_acc.append(acc)
    print('after top '+ str(length) + ' filter:\n')
    # print(Confusion_matrix)
    for i in range(10):
        for j in range(10):
            print('{0:.3f}'.format(Confusion_matrix[i][j]),end = ' ')
        print('\n')

        # Confusion_matrix_part.extend(C[str(0)])
            # Confusion_matrix_part.extend(C['1'])
            # Confusion_matrix_part.extend(C['2'])
            # Confusion_matrix_part.extend(C['3'])
            # Confusion_matrix_part.extend(C['4'])
            # Confusion_matrix_part.extend(C['5'])
            # Confusion_matrix_part.extend(C['6'])
            # Confusion_matrix_part.extend(C['7'])
            # Confusion_matrix_part.extend(C['8'])
            # Confusion_matrix_part.extend(C['9'])
            # for n in range(len(pool_list)):
            #     sum = 0
            #     index = pool_list[n]
            #     for length in range(len(vector_4096_list[0])):
            #         sum += abs(
            #             pow(float(vector_4096_list[index][length]), 2) - pow(float(vector_4096[length]), 2)) ** 0.5
            #     Euclidean_list.append(sum)
            # last_list = list(map(Euclidean_list.index, heapq.nsmallest(top, Euclidean_list)))
            # last_list = list(map(lambda x: pool_list[x], last_list))
            # temp_list2 = list(map(lambda x: type_list[x], last_list[0:top]))
            # C2 = Counter(temp_list2)
            # acc2 = C2[type] / len(temp_list2)
            # DCG2 = cal_DCG(temp_list2, type)
            # global total_acc2

def detailed():
    # detail_window = tk.Tk()
    # detail_window.title('类型分析')
    # detail_window.geometry('%dx%d'%(detail_window.winfo_screenwidth(),detail_window.winfo_screenheight()))
    d = defaultdict(list)
    for v, i in [(v, i) for i, v in enumerate(type_list)]:
        d[v].append(i)
    global type
    l = d[type]
    total_avg_acc1 = []
    total_avg_DCG1 = []
    total_avg_acc2 = []
    total_avg_DCG2 = []
    total_avg_acc3 = []
    total_avg_DCG3 = []
    for i in range(len(l)):
        detail_analyze(l[i],i + 1,total_avg_acc1,
                       total_avg_acc2,total_avg_DCG1,total_avg_DCG2,total_avg_acc3,total_avg_DCG3)

    poi = np.arange(0, len(l))
    fig = plt.figure(figsize=(6, 3))
    # plt.rcParams['font.sas-serig'] = ['SimHei']
    plt.xlabel('Image Number')
    plt.ylabel('Average ACC')
    plt.title('ACC')
    # x = np.arange(0,1000)
    # y1 = 2 * x + 1
    # y2 = x ** 2
    # l1, = plt.plot(x, y1, color = 'blue', linewidth=1.0, linestyle='-')
    # l2, = plt.plot(x, y2,  color='red', linewidth=1.0, linestyle='--')
    # plt.legend(handles=[l1, l2, ], labels=['Euclidean distance Filter', 'Hamming distance Filter'],loc = 'upper right')
    plt.plot(poi, total_avg_acc1[0:len(l)], c = 'red',label = 'Hamming distance Filter Only')
    plt.plot(poi, total_avg_acc2[0:len(l)], c ='blue',label = 'Euclidean distance Filter Only')
    plt.plot(poi, total_avg_acc3[0:len(l)], c ='green',label = 'Hamming distance and Euclidean distance Filter')
    plt.legend()
    plt.show()

    poi = np.arange(0, len(l))
    fig = plt.figure(figsize=(6, 3))
    # plt.rcParams['font.sas-serig'] = ['SimHei']
    plt.xlabel('Image Number')
    plt.ylabel('Average DCG')
    plt.title('DCG')
    # x = np.linspace(-1, 1, 50)
    # y1 = 2 * x + 1
    # y2 = x ** 2
    # l1, = plt.plot(x, y1, color = 'blue', linewidth=1.0, linestyle='-')
    # l2, = plt.plot(x, y2,  color='red', linewidth=1.0, linestyle='--')
    # plt.legend(handles=[l1, l2, ], labels=['Euclidean distance Filter', 'Hamming distance Filter'])
    plt.plot(poi, total_avg_DCG1[0:len(l)], c='red',label = 'Hamming distance Filter Only')
    plt.plot(poi, total_avg_DCG2[0:len(l)], c='blue',label = 'Euclidean distance Filter Only')
    plt.plot(poi, total_avg_DCG3[0:len(l)], c='green',label = 'Hamming distance and Euclidean distance Filter')
    plt.legend()
    plt.show()


def detail_analyze(index = 0,times = 0,total_avg_acc1 = [],
                   total_avg_acc2 = [],total_avg_DCG1 = [],total_avg_DCG2 = [],
                   total_avg_acc3 = [],total_avg_DCG3 = []):
    xor_list = []
    xor_list_count = []
    xor_pool = []
    pool_list = []
    Euclidean_list = []
    last_list = []
    final_Euclidean_list = []
    final_list = []
    for i in range(len(vector_48_list)):
        temp = bin(int(vector_48_list[i], 2) ^ int(vector_48_list[index], 2))
        temp = temp[2:]
        xor_list.append(temp.zfill(48))
        C = Counter(xor_list[i])
        xor_list_count.append(C['1'])
        if (xor_list_count[i] <= threshold_value):
            xor_pool.append(i)
    pool_set = set(heapq.nsmallest(top, xor_list_count))
    for i in range(len(pool_set)):
        value = pool_set.pop()
        pool_list.extend([x for x, y in enumerate(xor_list_count) if y == value])
    temp_list = list(map(lambda x: type_list[x], pool_list[0:top]))
    C1 = Counter(temp_list)
    acc1 = C1[type]/len(temp_list)
    DCG1 = cal_DCG(temp_list,type)
    global total_acc1
    global total_DCG1
    total_acc1 += acc1
    total_DCG1 += DCG1
    total_avg_acc1.append(total_acc1/times)
    total_avg_DCG1.append(total_DCG1/times)

    for n in range(len(pool_list)):
        sum = 0
        index = pool_list[n]
        for length in range(len(vector_4096_list[0])):
            sum += abs(pow(float(vector_4096_list[index][length]), 2) - pow(float(vector_4096[length]), 2)) ** 0.5
        Euclidean_list.append(sum)
    last_list = list(map(Euclidean_list.index, heapq.nsmallest(top, Euclidean_list)))
    last_list = list(map(lambda x: pool_list[x], last_list))
    temp_list2 = list(map(lambda x: type_list[x], last_list[0:top]))
    C2 = Counter(temp_list2)
    acc2 = C2[type]/len(temp_list2)
    DCG2 = cal_DCG(temp_list2,type)
    global total_acc2
    global total_DCG2
    total_acc2 += acc2
    total_DCG2 += DCG2
    total_avg_acc2.append(total_acc2/times)
    total_avg_DCG2.append(total_DCG2/times)

    for n in range(len(xor_pool)):
        sum = 0
        index = xor_pool[n]
        for length in range(len(vector_4096_list[0])):
            sum += abs(pow(float(vector_4096_list[index][length]), 2) - pow(float(vector_4096[length]), 2)) ** 0.5
        final_Euclidean_list.append(sum)
    final_list = list(map(final_Euclidean_list.index, heapq.nsmallest(top, final_Euclidean_list)))
    final_list = list(map(lambda x: xor_pool[x], final_list))
    temp_list3 = list(map(lambda x: type_list[x], final_list[0:top]))
    C3 = Counter(temp_list3)
    acc3 = C3[type]/len(temp_list3)
    DCG3 = cal_DCG(temp_list3,type)
    global total_acc3
    global total_DCG3
    total_acc3 += acc3
    total_DCG3 += DCG3
    total_avg_acc3.append(total_acc3/times)
    total_avg_DCG3.append(total_DCG3/times)

def image_window(E,window,L):
    global input_image_file
    input_image_file = askopenfilename(title = '选择文件' , filetypes=[('JPG', '*.jpg'), ('All Files', '*')] ,
                                       initialdir = '/home/yl/yls_File/caffe-master/data/cifar10-dataset/test/' )
    E.select_clear()
    E.insert(0,input_image_file)
    if(len(input_image_file) == 0):
        sys.exit(1)
    extra_binary_vector()
    retrueval_from_txt()
    xor_creat_pool()
    Euclidean_distance()
    # secondary = tk.Toplevel()
    # secondary.title('Top 20 Images')
    # secondary.geometry('1000x250')
    # start = input_image_file.rfind("/")
    # reg = "/" + input_image_file[start +1 :] + "[\S\s]{2}"
    # lstr = re.findall(reg, data)
    # type = ''
    # if (len(lstr) == 0):
    #     input_lable = tk.Tk()
    #     input_lable.title('图片类型')
    #     input_lable.geometry('200x100')
    #     E = tk.Entry(input_lable, textvariable=type, width=10, bd=3, cursor='arrow')
    #     E.grid(row = 0 , column = 0)
    #     B = tk.Button(input_lable, text='确认', width=10, bd=3, command=lambda:input_lable.destroy())
    #     B.grid(row = 0 , column = 1)
    #     # input_lable.mainloop()
    # else:
    #     type = lstr[0]
    #     pos = type.rfind(" ")
    #     type = type[pos + 1:]

    # C = Counter(last_type)
    # acc = C[type]/top
    # L['text'] = '类型 : ' + type  + ' '+ '准确率 : ' + str(acc)
    img_xor_list = []
    img_list = []
    button_list = []
    global type
    origin_image_frame = tk.Frame(window ,  width = 300,height = 300 , bd =2)
    origin_image_frame.grid(row = 1 , column = 0)
    oringin_lable =tk.Label(origin_image_frame , text = '输入图像:' , compound='center')
    oringin_lable.grid(row = 0 , column = 0)
    global img
    im = Image.open(input_image_file)
    imr = im.resize((100, 100), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(imr)
    oringin_image_lable = tk.Label(origin_image_frame , image=img, compound='center')
    oringin_image_lable.image = img
    oringin_image_lable.grid(row = 1 , column = 0)
    after_xor_filter = tk.Label(window, text='汉明距离筛选后:', compound='center')
    after_xor_filter.grid(row=2, column=0)
    after_xor_filter = tk.Label(window, text='DCG:' + ' 准确率:', compound='center')
    after_xor_filter.grid(row=3, column=0)
    image_pool_frame = tk.Frame(window , width = 300,height = 300 , bd =2)
    image_pool_frame.grid(row = 4 , column = 0)

    for n in range(top):
        # ft = tk.font.Font(family='Fixdsys', size=10, weight=tk.font.Font.BOLD)
        temp = last_index[n]
        name_lable = tk.Button(image_pool_frame, text = top_pool_name[n] + " 类型:" + top_pool_type[n] , bg = "pink", font = ("Arial", 12) ,
                               bd = 0 , command = lambda:retrieval_48_vector(temp))
        button_list.append(name_lable)
        name_lable.grid(row =  int(n/10)*2 ,column = n%10)
        # name_lable.place(relx = 0, rely = 0.1, relwidth = 0.65, relheight = 0.2)
        # type_lable = tk.LabelFrame(image_frame , text = last_type[n], bg = "pink" ,  font=("Arial", 12) , bd = 0)
        # type_lable.place(relx = 0.7, rely = 0.1, relwidth = 0.3, relheight = 0.2)
        im = Image.open(caffe_root + 'data/cifar10-dataset/test/' + top_pool_name[n])
        imr = im.resize((100, 100),Image.ANTIALIAS)
        img = ImageTk.PhotoImage(imr)
        # img.height()
        # img.width()
        img_xor_list.append(img)
        image_lable = tk.Label(image_pool_frame, image = img_xor_list[n], compound = 'center')
        image_lable.image = img_xor_list[n]
        image_lable.grid(row =  int(n/10)*2 + 1 ,column = n%10)
    after_Euclidean_filter = tk.Label(window, text='欧式距离筛选后:', compound='center')
    after_Euclidean_filter.grid(row=5, column=0)
    after_Euclidean_filter = tk.Label(window, text='DCG:' + ' 准确率:', compound='center')
    after_Euclidean_filter.grid(row=6, column=0)
    image_frame = tk.Frame(window , width = 300, height = 300, bd = 2 )
    image_frame.grid(row = 7 , column = 0)

    for n in range(len(last_name)):
        # ft = tk.font.Font(family='Fixdsys', size=10, weight=tk.font.Font.BOLD)
        temp = last_index[n]
        name_lable = tk.Button(image_frame, text = last_name[n] + " 类型:" + last_type[n] , bg = "pink", font = ("Arial", 12) ,
                               bd = 0 , command = lambda:retrieval_48_vector(temp))
        button_list.append(name_lable)
        name_lable.grid(row =  int(n/10)*2 ,column = n%10)
        # name_lable.place(relx = 0, rely = 0.1, relwidth = 0.65, relheight = 0.2)
        # type_lable = tk.LabelFrame(image_frame , text = last_type[n], bg = "pink" ,  font=("Arial", 12) , bd = 0)
        # type_lable.place(relx = 0.7, rely = 0.1, relwidth = 0.3, relheight = 0.2)
        im = Image.open(caffe_root + 'data/cifar10-dataset/test/' + last_name[n])
        imr = im.resize((100, 100),Image.ANTIALIAS)
        img = ImageTk.PhotoImage(imr)
        # img.height()
        # img.width()
        img_list.append(img)
        image_lable = tk.Label(image_frame, image = img_list[n], compound = 'center')
        image_lable.image = img_list[n]
        image_lable.grid(row =  int(n/10)*2 + 1 ,column = n%10)
        # image_lable.place(relx=0, rely=0.2, relwidth=1, relheight=1)
        # image_lable.place(relx = 0 , rely = 0.2, relwidth=1 , relheight = 0.8)
        # name_list[n]
        # print( str(n/10) + ","+str(n%10))
        # image_frame.pack()
    # secondary.mainloop()
    start = input_image_file.rfind("/")

    # reg = "/" + input_image_file[start + 1:] + "[\S\s]{2}"
    # reg_result = re.findall(reg, data)
    name = input_image_file[start +1:]
    if (name not in name_list):
        input_lable = tk.Tk()
        input_lable.title('图片类型')
        input_lable.geometry('185x35')
        E = tk.Entry(input_lable, textvariable=type, width=10, bd=3, cursor='arrow')
        E.grid(row=0, column=0)
        B = tk.Button(input_lable, text='确认', width=10, bd=3, command=lambda: after_click_commit(input_lable,E,after_xor_filter,after_Euclidean_filter,L))
        B.grid(row=0, column=1)
        # input_lable.mainloop()
    else:
        type = type_list[name_list.index(name)]
        # pos = type.rfind(" ")
        # type = type[pos + 1:]
        C1 = Counter(top_pool_type[0:top])
        C2 = Counter(last_type)
        acc1 = C1[type] / top
        acc2 = C2[type] / top
        L['text'] = '类型 : ' + type
        after_xor_filter['text'] = 'DCG: ' + str(cal_DCG(top_pool_type[0:20], type)) + ' 准确率: ' + str(acc1)
        after_Euclidean_filter['text'] = 'DCG: ' + str(cal_DCG(last_type, type)) + ' 准确率: ' + str(acc2)

def after_click_commit(input_lable,E,after_xor_filter,after_Euclidean_filter,L):
    global type
    type = E.get()
    C1 = Counter(top_pool_type[0:top])
    C2 = Counter(last_type)
    acc1 = C1[type] / top
    acc2 = C2[type] / top
    L['text'] = '类型 : ' + type
    after_xor_filter['text'] = 'DCG: ' + str(cal_DCG(top_pool_type[0:20],type)) + ' 准确率: ' + str(acc1)
    after_Euclidean_filter['text'] = 'DCG: ' + str(cal_DCG(last_type,type)) + ' 准确率: ' + str(acc2)
    input_lable.destroy()

def cal_DCG(list = [],type = 0):
    acc = 0.0
    for i in range(len(list)):
        if(list[i] == type):
            acc = acc + 1/math.log(i + 2,2)
        else:
            continue
    return acc

def retrieval_48_vector(index = 0):
    vector_48 = vector_48_list[index]
    vector_4096 = vector_4096_list[index]
    xor_creat_top_pool()
    show_related_images()

def xor_creat_top_pool():
    exclusive_OR_list.clear()
    exclusive_OR_statistics_list.clear()
    pool.clear()
    for i in range(len(vector_48_list)):
        temp = bin(int(vector_48_list[i],2) ^ int(vector_48[0],2))
        temp = temp[2:]
        exclusive_OR_list.append(temp.zfill(48))
        C = Counter(exclusive_OR_list[i])
        exclusive_OR_statistics_list.append(C['1'])
        # if(exclusive_OR_statistics_list[i] <= threshold_value):
        #     pool.append(i)
    global last_index
    last_index.clear()
    last_index = list(map(exclusive_OR_statistics_list.index, heapq.nsmallest(top, exclusive_OR_statistics_list)))
    last_name.clear()
    last_name.extend(list(map(lambda x:name_list[x], last_index)))

def show_related_images():
    secondary = tk.Toplevel()
    secondary.title('根据48位二进制向量检索相关到前20张图片')
    secondary.geometry('1000x250')
    img_list = []
    for n in range(len(last_name)):
        image_frame = tk.Frame(secondary ,  width = 300, height = 300, bd = 1 )
        image_frame.grid(row = int(n / 10) , column = n % 10)
        # ft = tk.font.Font(family='Fixdsys', size=10, weight=tk.font.Font.BOLD)
        name_lable = tk.LabelFrame(image_frame, text = last_name[n], bg = "pink", font = ("Arial", 12) , bd = 0)
        name_lable.place(relx = 0, rely = 0.1, relwidth = 0.65, relheight = 0.2)
        type_lable = tk.LabelFrame(image_frame , text = last_type[n], bg = "pink" ,  font=("Arial", 12) , bd = 0)
        type_lable.place(relx = 0.7, rely = 0.1, relwidth = 0.3, relheight = 0.2)
        im = Image.open(caffe_root + 'data/cifar10-dataset/test/' + last_name[n])
        img = ImageTk.PhotoImage(im)
        img_list.append(img)
        image_lable = tk.Label(image_frame, image = img_list[n], compound = 'center', command = lambda :retrieval_48_vector(last_index[n]))
        image_lable.image = img_list[n]
        image_lable.place(relx=0, rely=0.2, relwidth=1, relheight=1)
        # image_lable.place(relx = 0 , rely = 0.2, relwidth=1 , relheight = 0.8)
        # name_list[n]
        # print( str(n/10) + ","+str(n%10))
        # image_frame.pack()
    secondary.mainloop()

if __name__ == '__main__':
    main_window()
    # print(vector_48)
    # print(vector_4096)
    # print(name_list)
    # print(vector_48_list)
    # print(os.path.basename(input_image_file))
