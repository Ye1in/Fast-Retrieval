import sys
import numpy as np
import caffe
import os
import time
import shutil
from PIL import Image

caffe_root = '/home/yl/yls_File/caffe-master/'
image_dir = caffe_root + 'examples/cvprw15-cifar10/test/'
net_file=caffe_root +'examples/cvprw15-cifar10/h=48/KevinNet_CIFAR10_48_deploy.prototxt'
caffe_model=caffe_root + 'examples/cvprw15-cifar10/h=48/KevinNet_CIFAR10_48.caffemodel'


def get_image_files():
#获取path路径下所有图片的名字
    files = os.listdir(image_dir)
    s=[]
    for file in files:
        s.append(file)
    return s

def remove_bad_images():
#对于不能正常读取的图片，从文件夹下移除
    s = get_image_files()
    for image in s:
        try:
            image_file =  image_dir + '/' + image
            img = Image.open(image_file)
            img.verify()
            #img = caffe.io.load_image(image_file)
        except IOError:
            print(image_file)
            # shutil.move(image_file,'/home/fwei/fdata/errimg')

def matrix_to_binary(M,digits = 0):
    # var = []
    for i in range(len(M)):
        # for col in range(len(M[row])):
            M[i] = round(M[i],0)
            # if(M[row][col] >= 0.5)
            #     var.insert(1)
            # else:
            #     var.insert(0)
    return M

def extra_binary_vector():
    net = caffe.Net(net_file,caffe_model,caffe.TEST)
    layer = 'fc8_kevin' #提取48位二进制向量
    if layer not in net.blobs:
        raise TypeError("Invalid layer name: " + layer)
    imagemean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1,3,227,227)
    image_files = get_image_files()
    i = 0
    # caffe.set_device(0)
    caffe.set_mode_cpu()
    for input_image_file in image_files:
            img_file = image_dir + input_image_file
            print ('No '+str(i))
            print (img_file)
            print ('_______________________________')
            img = caffe.io.load_image(img_file)
            net.blobs['data'].data[...] = transformer.preprocess('data', img)
            net.forward()
            feature_data = net.blobs[layer].data[0]
            feature_data = (feature_data - feature_data.min())/(feature_data.max() - feature_data.min())
            # output_file = input_image_file+'.txt'
            # with open(output_file, 'w') as f:
            #     np.savetxt(f,feature_data,fmt='%.10f',delimiter='\n')
            # f.close()
            feature_data = matrix_to_binary(feature_data,0)
            print(feature_data)
            output_file = caffe_root + 'examples/cvprw15-cifar10/48-bits binary vector' + '.txt'
            with open(output_file,'a+') as f:
                f.write(input_image_file + ':\n')
                np.savetxt(f,feature_data,fmt='%.0f',delimiter = '\t',newline='')
                f.write('\n')
            i+=1
            f.close()
    print (i)

def extra_image_vector():
    net = caffe.Net(net_file,caffe_model,caffe.TEST)
    layer = 'fc7' #提取fc7层的特征
    if layer not in net.blobs:
        raise TypeError("Invalid layer name: " + layer)
    imagemean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1,3,227,227)
    image_files = get_image_files()
    i = 0
    # caffe.set_device(0)
    caffe.set_mode_cpu()
    for input_image_file in image_files:
            img_file = image_dir + input_image_file
            print ('No '+str(i))
            print (img_file)
            print ('_______________________________')
            img = caffe.io.load_image(img_file)
            net.blobs['data'].data[...] = transformer.preprocess('data', img)
            net.forward()
            feature_data = net.blobs[layer].data[0]
            feature_data = (feature_data - feature_data.min())/(feature_data.max() - feature_data.min())
            # output_file = input_image_file+'.txt'
            # with open(output_file, 'w') as f:
            #     np.savetxt(f,feature_data,fmt='%.10f',delimiter='\n')
            # f.close()
            # feature_data = matrix_to_binary(feature_data,0)
            print(feature_data)
            output_file = caffe_root + 'examples/cvprw15-cifar10/4096-bits image vector' + '.txt'
            with open(output_file,'a+') as f:
                f.write(input_image_file + ':\n')
                np.savetxt(f,feature_data,fmt='%.2f/',delimiter = '\t',newline='')
                f.write('\n')
            i+=1
            f.close()
    print (i)

if __name__ == '__main__':
    remove_bad_images()
    extra_binary_vector()
    extra_image_vector()