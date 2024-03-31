#Importing necessary libraries for reading images
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread,imshow
from glob import glob
from skimage.transform import resize
#Importing necessary libraries for using Transfer Learning algorithm
from keras.preprocessing import image as kimage
from keras.applications.mobilenet import preprocess_input
from keras.applications import MobileNet
from keras.applications.resnet50 import preprocess_input
from keras.applications import ResNet50
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.applications.vgg19 import preprocess_input
from keras.applications import VGG19

global y
global category
global x_trg
global x_test
global y_trg
global y_test

#Creating list of categories for representing dependent variables
y=[1,2,3,2,2,2,2,1,3,4,1,2,4,2,3,3,4,2,2,4,2,2,3,5,5,5,1,4,2,2,2,3,3,1,1,3,1,1,4,2,5,5,5,1,5,2,4,5,1,3,5,3,4,3,1,4,5,3,1,6,3,5,5,4,3,6,4,4,6,4,3,4,6,4,6,6,6,6,6,6,6]
print("Number of dependent variables:",len(y))

#Creating list of categories
category=["Badger","Antelope","Bear","Bee","Butterfly","Bat"]
x_trg,x_test,y_trg,y_test=0,0,0,0

#Importing necessary libraries
np.random.seed(1000)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def setdataset(matrix):
    global y
    global category
    global x_trg
    global x_test
    global y_trg
    global y_test

    #Creating training test datasets for dependent and independent variables
    x_trg,x_test,y_trg,y_test=train_test_split(matrix,y,random_state=0,test_size=0.2)
    print("Dimension of training, test dataset: ",x_trg.shape,x_test.shape)

def readimage(train,test):
    i=1
    plt.figure(1,figsize=(40,30))
    matrix=np.zeros([81,750000]) 

    #Reading and resizing the images
    for image in glob(train+'/*.jpg'):
        image_form=imread(image,as_gray=False)

        #Resizing all the images to same dimension
        img_resized=resize(image_form,(500,500))

        #Flattening the images to one dimension for applying algorithms
        item=img_resized.flatten()
        matrix[i-1,:]=item
        i=i+1

    #Displaying the dimension of the matrix and images
    print("The shape of the matrix is:",matrix.shape)
    plt.show()

    #Reading and displaying the test image for prediction
    image = imread(test,as_gray=False)
    img_resized=resize(image,(500,500))
    item=np.zeros([1,750000])
    item[0,:]=img_resized.flatten()

    setdataset(matrix)
    return item

def readimageTL(train):
    images_dict=dict()

    #Processing all images according to mobilenet algorithm
    for mix_image in glob(train+'/*.jpg'):

        #Generally accepted image size for trained models is 224 X 224 px
        image=kimage.load_img(mix_image,target_size=(224,224))

        #Converting image to array form
        image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))

        #Extracting image name from the path
        num=mix_image.split('\\')[-1].split('.')[0]

        #Mapping image and id
        images_dict[num]=image
    return images_dict
def mobnet(images_dict,test):
    #Creating the model
    mobile_net_model=MobileNet(include_top=False,weights='imagenet')

    #Creating the matrix and initializing matrix to 0
    images_matrix2=np.zeros([81,50176])
    for i, (num,image) in enumerate(images_dict.items()):

        #Flatten the matrix after using mobilenet model
        images_matrix2[i,:]=mobile_net_model.predict(image).ravel()

    image=kimage.load_img(test,target_size=(224,224))
    image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
    item=np.zeros([1,50176])
    item[0,:]=mobile_net_model.predict(image).ravel()
    setdataset(images_matrix2)
    return item

def resnet(images_dict,test):
    #Creating the model
    res_net_model=ResNet50(include_top=False,weights='imagenet')

    #Creating the matrix and initializing matrix to 0
    images_matrix=np.zeros([81,100352])
    for i, (num,image) in enumerate(images_dict.items()):

        #Flatten the matrix after using mobilenet model
        images_matrix[i,:]=res_net_model.predict(image).ravel()

    image=kimage.load_img(test,target_size=(224,224))
    image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
    item=np.zeros([1,100352])
    item[0,:]=res_net_model.predict(image).ravel()
    setdataset(images_matrix)
    return item

def vgg16(images_dict,test):
    #Creating the model
    vgg16_model=VGG16(include_top=False,weights='imagenet')

    #Creating the matrix and initializing matrix to 0
    images_matrix=np.zeros([81,25088])
    for i, (num,image) in enumerate(images_dict.items()):

        #Flatten the matrix after using mobilenet model
        images_matrix[i,:]=vgg16_model.predict(image).ravel()

    image=kimage.load_img(test,target_size=(224,224))
    image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
    item=np.zeros([1,25088])
    item[0,:]=vgg16_model.predict(image).ravel()
    setdataset(images_matrix)
    return item

def vgg19(images_dict,test):

    #Creating the model
    vgg19_model=VGG19(include_top=False,weights='imagenet')

    #Creating the matrix and initializing matrix to 0
    images_matrix=np.zeros([81,25088])
    for i, (num,image) in enumerate(images_dict.items()):

        #Flatten the matrix after using mobilenet model
        images_matrix[i,:]=vgg19_model.predict(image).ravel()

    image=kimage.load_img(test,target_size=(224,224))
    image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
    item=np.zeros([1,25088])
    item[0,:]=vgg19_model.predict(image).ravel()
    setdataset(images_matrix)
    return item


def naive(item):
    global y
    global category
    global x_trg
    global x_test
    global y_trg
    global y_test

    #Creating Naive-Bayes model
    from sklearn.naive_bayes import GaussianNB

    print("---------------------------Naive Bayes Model----------------------------")
    naive_model=GaussianNB()
    naive_model.fit(x_trg,y_trg)

    #Determining accuracy and creating confusion matrix of the model
    naive_pred=naive_model.predict(x_test)
    naive_acc_score=accuracy_score(y_test,naive_pred)
    naive_results=confusion_matrix(y_test,naive_pred)
    print("The accuracy of Naive bayes model is: %0.4f"%naive_acc_score)
    print("The confusion matrix is:\n",naive_results)

    #Determining accuracy of the test image
    naive_pred=naive_model.predict(item)
    naive_value=naive_pred[0]
    print("Category of test image is: ",category[naive_value-1])

def decision(item):
    global y
    global category
    global x_trg
    global x_test
    global y_trg
    global y_test

    #Creating Decision tree model
    from sklearn.tree import DecisionTreeClassifier
    print("---------------------------Decision Tree Model----------------------------")
    tree_model=DecisionTreeClassifier(random_state=0)
    tree_model.fit(x_trg,y_trg)

    #Determining accuracy and creating confusion matrix of the model
    tree_pred=tree_model.predict(x_test)
    tree_acc_score=accuracy_score(y_test,tree_pred)
    tree_results=confusion_matrix(y_test,tree_pred)
    print("The accuracy of decision tree model is: %0.4f"%tree_acc_score)
    print("The confusion matrix is:\n",tree_results)

    #Determining accuracy of the test image
    tree_pred=tree_model.predict(item)
    tree_value=tree_pred[0]
    print("Category of test image is: ",category[tree_value-1])

def forest(item):
    global y
    global category
    global x_trg
    global x_test
    global y_trg
    global y_test

    #Creating Random forest model
    from sklearn.ensemble import RandomForestClassifier
    print("---------------------------Random Forest Model----------------------------")
    forest_model=RandomForestClassifier(random_state=0)
    forest_model.fit(x_trg,y_trg)

    #Determining accuracy and creating confusion matrix of the model
    forest_pred=forest_model.predict(x_test)
    forest_acc_score=accuracy_score(y_test,forest_pred)
    forest_results=confusion_matrix(y_test,forest_pred)
    print("The accuracy of Random forest model is: %0.4f"%forest_acc_score)
    print("The confusion matrix is:\n",forest_results)

    #Determining accuracy of the test image
    forest_pred=forest_model.predict(item)
    forest_value=forest_pred[0]
    print("Category of test image is: ",category[forest_value-1])

def bag(item):
    global y
    global category
    global x_trg
    global x_test
    global y_trg
    global y_test

    #Creating Bagging model
    from sklearn.ensemble import BaggingClassifier
    print("---------------------------Bagging Model----------------------------")
    bag_model=BaggingClassifier(estimator=None,n_estimators=10,max_samples=1.0,max_features=1.0,bootstrap=True)
    bag_model.fit(x_trg,y_trg)

    #Determining accuracy and creating confusion matrix of the model
    bag_pred=bag_model.predict(x_test)
    bag_acc_score=accuracy_score(y_test,bag_pred)
    bag_results=confusion_matrix(y_test,bag_pred)
    print("The accuracy of Bagging model is: %0.4f"%bag_acc_score)
    print("The confusion matrix is:\n",bag_results)

    #Determining accuracy of the test image
    bag_pred=bag_model.predict(item)
    bag_value=bag_pred[0]
    print("Category of test image is: ",category[bag_value-1])

