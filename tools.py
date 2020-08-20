import numpy as np
import consts

def OneHotEncoding(array):
    OneHotEncodedArray = []
    for img_label in array:
        target = [0] * consts.classes
        target[img_label] = 1
        OneHotEncodedArray.append(target)
    return np.array(OneHotEncodedArray)

def reshaped_for_input(array_of_imgs):
    shape = list(array_of_imgs.shape).copy()
    shape.append(consts.streams)
    return np.reshape(array_of_imgs,shape)

def FromOneHot(OneHotEncodedArray):
    normal_array = []
    for target in OneHotEncodedArray:
        normal_array.append(list(target).index(max(list(target))))
    return np.array(normal_array)

def get_mnist_prediction(prediction):
    return(list(prediction[0]).index(max(list(prediction[0]))))
