from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from keras.utils import np_utils
import warnings
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import SGD,Adadelta
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from random import shuffle
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_trainSet():
    root="DATA"
    folders = os.listdir(root)
    nb = len(folders)
    x_data=[]
    y_label=[]
    for x in range(nb):
        label=np.zeros(nb)
        label[x]=1
        facesPath=os.listdir(root+"/"+folders[x])
        faces=[root+"/"+folders[x]+"/"+f for f in facesPath if f.endswith(".png")]
        for face in faces:
            img=cv2.imread(face)
            x_data.extend([img])
            y_label.extend([label])
    return (np.asarray(x_data), np.asarray(y_label), nb)

def mixData(xs,ys):
    xys=[]
    for i in range(0,len(xs)-1):
        xys.append((xs[i],ys[i]))
    shuffle(xys)
    x2,y2=[],[]
    for (x,y) in xys:
        x2.append(x)
        y2.append(y)
    return (np.asarray(x2), np.asarray(y2))



#*******************MY MODEL***********************************************
def my_model(include_top=True, nb_person=10):
    img_input = Input(shape=(100, 100, 3))
    #Block 1 avec 64 filtre
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1')(img_input)
    x=BatchNormalization()(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    x= Dropout(0.25)(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2')(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x= Dropout(0.25)(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2')(x)
    x=BatchNormalization()(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    x= Dropout(0.25)(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3')(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    x= Dropout(0.25)(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3')(x)
    x=BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    x= Dropout(0.25)(x)

    #Classifieur du modele
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x= Dropout(0.25)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dense(nb_person, activation='softmax', name='fc8')(x)
    model = Model(img_input, x)
    return model
#*******************END MY MODEL*****************************************************************


#*******************MAIN*************************************************************************
if __name__ == '__main__':
    from scipy import misc
    import copy
    (x_train, y_train, nb_class)=load_trainSet()
    (x_train, y_train)=mixData(x_train, y_train)
    model = my_model(nb_person=nb_class)
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    delta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=delta,  metrics=['accuracy'])
    print("Entrainement du modele")
    hist=model.fit(x_train, y_train, validation_split=0.33, batch_size=32, epochs=1000, verbose=1)
    
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('courbe de la perte DataSet avec augmentation / 1000 itération')
    plt.ylabel('Perte')
    plt.xlabel('itération')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


from google.colab import files
model.save_weights('my_model_weights_fam.h5')

uploaded = drive.CreateFile({'title': 'my_model_weights_fam.h5'})
uploaded.SetContentFile('model_weights_AUG_300.h5')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))