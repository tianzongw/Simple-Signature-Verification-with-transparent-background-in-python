
import numpy as np
import json
from matplotlib import (pyplot as plt, patches as patches)
from PIL import Image # pip install pillow
from skimage.filters import threshold_local # pip install scikit-image
from scipy.ndimage.measurements import center_of_mass
from sklearn import svm
import os
# from sklearn.model_selection import train_test_split, for cross-validation purposes
# from sklearn.model_selection import KFold, for k-fold cross-validation purposes



model = None
mapping = None

def rotate_bbox_area(img, deg):
  box = img.rotate(deg, expand=True).getbbox()
  return (box[3] - box[1]) * (box[2] - box[0])

def rotate_crop(img, deg, padding=0):
  img_rotate = img.rotate(deg, expand=True, resample=Image.BILINEAR)
  box = img_rotate.getbbox()
  if padding > 0:
      box = np.asarray(box) + [-padding, -padding, +padding, +padding]
  return img_rotate.crop(box)

tol_deg = 10
# smallest bounding box wihin -10~10 degrees rotation
def opt_rotate(img, padding=0):
  opt_deg = np.argmin(
      [rotate_bbox_area(img, i) for i in range(-tol_deg,tol_deg+1)]
      ) - tol_deg
  return rotate_crop(img, opt_deg, padding)

# downsampling
def img_reduce(img, side=28, mode=Image.ANTIALIAS):
  h = side + 1
  w = int(side * img.width / img.height) + 1
  img_reduced = img.copy()
  # the reduced image size is (w-1, h-1)
  img_reduced.thumbnail((w, h), mode)
  return img_reduced


# convert PIL.Image object to numpy.Array, for training
def img2arr(img):
  return np.asarray(img.getdata(), dtype=np.uint8).reshape(img.height, img.width, -1)


# process single signature with transparent background
def process_one(img):
  return img_reduce(opt_rotate(img, padding=1).convert('LA'))

# shuffle two numpy arraies in unison form, for ramdom spliting use
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

# Obtaining images and labels with image normalization and augmentation
# Crop the image first and create a dictionary to store the cropped image as value
# and the image path as key, for later use in test()

def readcsv(filename):
    global mapping
    data = np.genfromtxt(filename, delimiter = ",", dtype = "U75", skip_header = 2)

    # Obtaining images and labels
    img_path = data[:,3]
    label = data[:,4]

    # Ramdom spliting the data
    img_path, label = shuffle_in_unison(img_path, label)
    Xlist = []
    ylist = []
    X_list_path = []
    mapping = {}
    idx = 0
    cwd = os.getcwd()

    for item in img_path:

        #if using whole image
        #X_list.append(img2arr(Image.open(cwd + '\\' + item).convert('LA')).flatten)
        #if label[idx] == 'genuine':
        #    ylist.append(1)
        #else:
        #    ylist.append(0)
        # X_list_path.append(cwd + '\\' + item)


        #if using partial image, eg. the 28*28 sample
        #data normalization using the code provided
        img = Image.open(cwd + '\\' + item).convert('LA')
        img_reduced = process_one(img)
        img_arr = img2arr(img_reduced)[:,:,-1]
        center = np.round(center_of_mass(img_arr))
        h = img_arr.shape[0]
        #after trials, h//3 will obtain better squared shape signatures than h//2
        left = int(center[1]) - h//3
        mat = img_arr[:, left:left+h]
        #Filter our those images that are not 28*28, all image for training should have the same size
        if mat.shape[1] == 28:
            Xlist.append(mat.flatten())
            # print(cwd + '\\' + item)
            # print(idx)
            X_list_path.append(cwd + '\\' + item)
            if label[idx] == 'genuine':
                ylist.append(1)
            else:
                ylist.append(0)
        idx += 1
    X =  np.array(Xlist)
    y = np.array(ylist)
    X_path = np.array(X_list_path)

    # trying n-fold cross validation with 70% data used for training
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # trying K-fold cross-validation, in this example we set k = 4
    # kf = KFold(n_splits=4)
    # kf.get_n_splits(X)
    # for train_index, test_index in kf.split(X):
        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]

    # Use random shuffle with 300 sample for training for simplicity
    X_train = X[:300]
    X_test= X[300:]
    X_test_path = X_path[300:]
    y_train = y[:300]
    y_test = y[300:]
    # as test() requires image_path as input but the training and testing function
    # requires numpy array as input, a dictionary was created to map the image path
    # to the cropped image (28*28)

    mapping = dict(zip(X_test_path, X_test))
    return  X_train, X_test, y_train, y_test, X_test_path
    #print(mapping)

def train():
    global model
    X_train, X_test, y_train, y_test, X_test_path = readcsv('png.csv')
    # Training with SVM with different kernals and C, gamma
    model = svm.SVC(kernel = 'linear', C = 1, gamma = 1)
    #model = svm.SVC(kernel = 'linear', C = 0.8, gamma = 1)
    #model = svm.SVC(kernel = 'linear', C = 0.5, gamma = 1)
    #model = svm.SVC(kernel = 'linear', C = 1, gamma = 2)
    #model = svm.SVC(kernel = 'linear', C = 0.8, gamma = 2)
    #model = svm.SVC(kernel = 'linear', C = 0.5, gamma = 2)
    #model = svm.SVC(kernel = 'poly', gamma = 2)
    #model = svm.SVC(kernel = 'rbf', gamma = 2)
    model.fit(X_train, y_train)
    model.score(X_train, y_train)
    return model, X_test_path, y_test

def test(path_str):
    global model
    global mapping
    # Predicting the result
    return model.predict([mapping[path_str]])[0]

if __name__ == '__main__':

    predicted = []
    model, X_test_path, y_test = train()
    for item in X_test_path:
        path = item
        predicted.append(test(path))
    # Comparing the prediction with test labels, to see how accurate our model is
    num = 0
    for i in range(len(y_test)):
        if predicted[i] == y_test[i]:
            num += 1
    print("The prediction accuracy is " + str(format(num/len(y_test)*100,'.2f')) + '%')


# In[189]:
