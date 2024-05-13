#Importing necessary libraries for reading images
import numpy as np
import cv2
import os
from keras.preprocessing import image as kimage

np.random.seed(1000)
from sklearn.model_selection import train_test_split

class Classify:
    def __init__(self,category_file,image_dir,test):
        # Path to the text file containing class names
        self.category_file=category_file
        # Directory where images are located
        self.images_dir=image_dir
        # Directory where test image is located
        self.test=test
        
        # Read class names from the text file
        with open(self.category_file, 'r') as file:
            self.categories = [line.strip() for line in file]

        
        # Map class names to their respective image folders
        self.category_to_folder = {category: os.path.join(image_dir, category) for category in self.categories}

        self.labels=[]
        self.x_trg,self.x_test,self.y_trg,self.y_test=0,0,0,0
        self.name=""
    def setdataset(self,matrix):
        #Creating training test datasets for dependent and independent variables
        self.x_trg,self.x_test,self.y_trg,self.y_test=train_test_split(matrix,self.labels,random_state=0,test_size=0.2)
        print("Dimension of training, test dataset: ",self.x_trg.shape,self.x_test.shape)

    def readimage(self):

        images = []
        labels = []
        for category, folder_path in self.category_to_folder.items():
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                try:
                    img_form = cv2.imread(img_path)
                    img_resized = cv2.resize(img_form, (500, 500))
                    #img = img.astype(np.float32) / 255.0  # Normalize pixel values
                    img_resized=cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
                    images.append(img_resized.flatten())
                    labels.append(self.categories.index(category))
                except Exception as e:
                    print(f"Error loading image '{img_path}': {e}")
        self.labels=np.array(labels)

        item=np.zeros([1,750000])
        img= cv2.imread(self.test)
        img_resized= cv2.resize(img, (500, 500))
        item[0,:]=img_resized.flatten()

        self.setdataset(np.array(images))
        return item

    def mobnet(self):
        from keras.applications.mobilenet import preprocess_input
        from keras.applications import MobileNet
        images = []
        labels = []
        for category, folder_path in self.category_to_folder.items():
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = kimage.load_img(img_path, target_size=(224, 224))
                    img = preprocess_input(np.expand_dims(kimage.img_to_array(img),axis=0))
                    images.append(img)
                    labels.append(self.categories.index(category))
                except Exception as e:
                    print(f"Error loading image '{img_path}': {e}")
        self.labels=np.array(labels)
        images= np.array(images)
        # Load MobileNet model
        mobile_net_model = MobileNet(weights='imagenet', include_top=False)

        # Predict features for each preprocessed image
        features_list = []
        for preprocessed_image in images:
            # Predict features for the preprocessed image
            features = mobile_net_model.predict(preprocessed_image)
            features_list.append(features.ravel())

        image=kimage.load_img(self.test,target_size=(224,224))
        image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
        item=np.zeros([1,50176])
        item[0,:]=mobile_net_model.predict(image).ravel()
        self.setdataset(np.array(features_list))
        self.name=" with MobileNet"
        return item

    def resnet(self):
        from keras.applications.resnet50 import preprocess_input
        from keras.applications import ResNet50

        images = []
        labels = []
        for category, folder_path in self.category_to_folder.items():
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = kimage.load_img(img_path, target_size=(224, 224))
                    img = preprocess_input(np.expand_dims(kimage.img_to_array(img),axis=0))
                    images.append(img)
                    labels.append(self.categories.index(category))
                except Exception as e:
                    print(f"Error loading image '{img_path}': {e}")
        self.labels=np.array(labels)
        images= np.array(images)
        #Creating the model
        res_net_model=ResNet50(include_top=False,weights='imagenet')

        # Predict features for each preprocessed image
        features_list = []
        for preprocessed_image in images:
            # Predict features for the preprocessed image
            features =res_net_model.predict(preprocessed_image)
            features_list.append(features.ravel())
        
        image=kimage.load_img(self.test,target_size=(224,224))
        image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
        item=np.zeros([1,100352])
        item[0,:]=res_net_model.predict(image).ravel()
        self.setdataset(np.array(features_list))
        self.name=" with ResNet"
        return item

    def vgg16(self):
        from keras.applications.vgg16 import preprocess_input
        from keras.applications import VGG16

        images = []
        labels = []
        for category, folder_path in self.category_to_folder.items():
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = kimage.load_img(img_path, target_size=(224, 224))
                    img = preprocess_input(np.expand_dims(kimage.img_to_array(img),axis=0))
                    images.append(img)
                    labels.append(self.categories.index(category))
                except Exception as e:
                    print(f"Error loading image '{img_path}': {e}")
        self.labels=np.array(labels)
        images= np.array(images)
        #Creating the model
        vgg16_model=VGG16(include_top=False,weights='imagenet')

        # Predict features for each preprocessed image
        features_list = []
        for preprocessed_image in images:
            # Predict features for the preprocessed image
            features = vgg16_model.predict(preprocessed_image)
            features_list.append(features.ravel())
            image=kimage.load_img(self.test,target_size=(224,224))

        image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
        item=np.zeros([1,25088])
        item[0,:]=vgg16_model.predict(image).ravel()
        self.setdataset(np.array(features_list))
        self.name=" with VGG16"
        return item

    def vgg19(self):
        from keras.applications.vgg19 import preprocess_input
        from keras.applications import VGG19

        images = []
        labels = []
        for category, folder_path in self.category_to_folder.items():
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = kimage.load_img(img_path, target_size=(224, 224))
                    img = preprocess_input(np.expand_dims(kimage.img_to_array(img),axis=0))
                    images.append(img)
                    labels.append(self.categories.index(category))
                except Exception as e:
                    print(f"Error loading image '{img_path}': {e}")
        self.labels=np.array(labels)
        images= np.array(images)
        #Creating the model
        vgg19_model=VGG19(include_top=False,weights='imagenet')

        # Predict features for each preprocessed image
        features_list = []
        for preprocessed_image in images:
            # Predict features for the preprocessed image
            features = vgg19_model.predict(preprocessed_image)
            features_list.append(features.ravel())
        
        image=kimage.load_img(self.test,target_size=(224,224))
        image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
        item=np.zeros([1,25088])
        item[0,:]=vgg19_model.predict(image).ravel()
        self.setdataset(np.array(features_list))
        self.name=" with VGG19"
        return item


    def naive(self,item):
        title="Naive Bayes Model"+self.name
        #Creating Naive-Bayes model
        from sklearn.naive_bayes import GaussianNB

        print("---------------------------Naive Bayes Model----------------------------")
        naive_model=GaussianNB()
        naive_model.fit(self.x_trg,self.y_trg)

        #Determining accuracy and creating confusion matrix of the model
        naive_pred=naive_model.predict(self.x_test)
        y_probas = naive_model.predict_proba(self.x_test)

        #Determining accuracy of the test image
        naive_pred_item=naive_model.predict(item)
        naive_value=naive_pred_item[0]
        print(naive_pred_item,self.categories)
        return naive_model,naive_pred,y_probas,title,self.categories[naive_value],self.x_trg,self.y_trg,self.y_test

    def decision(self,item):

        title="Decision Tree"+self.name
        #Creating Decision tree model
        from sklearn.tree import DecisionTreeClassifier
        print("---------------------------Decision Tree Model----------------------------")
        tree_model=DecisionTreeClassifier(random_state=0)
        tree_model.fit(self.x_trg,self.y_trg)

        #Determining accuracy and creating confusion matrix of the model
        tree_pred=tree_model.predict(self.x_test)
        y_probas = tree_model.predict_proba(self.x_test)

        #Determining accuracy of the test image
        tree_pred_item=tree_model.predict(item)
        tree_value=tree_pred_item[0]
        return tree_model,tree_pred,y_probas,title,self.categories[tree_value],self.x_trg,self.y_trg,self.y_test

    def forest(self,item):
        title="Random Forest"+self.name
        #Creating Random forest model
        from sklearn.ensemble import RandomForestClassifier
        print("---------------------------Random Forest Model----------------------------")
        forest_model=RandomForestClassifier(random_state=0)
        forest_model.fit(self.x_trg,self.y_trg)

        #Determining accuracy and creating confusion matrix of the model
        forest_pred=forest_model.predict(self.x_test)
        y_probas = forest_model.predict_proba(self.x_test)

        #Determining accuracy of the test image
        forest_pred_item=forest_model.predict(item)
        forest_value=forest_pred_item[0]
        return forest_model,forest_pred,y_probas,title,self.categories[forest_value],self.x_trg,self.y_trg,self.y_test

    def bag(self,item):
        title="Bagging Model"+self.name
        #Creating Bagging model
        from sklearn.ensemble import BaggingClassifier
        print("---------------------------Bagging Model----------------------------")
        bag_model=BaggingClassifier(estimator=None,n_estimators=10,max_samples=1.0,max_features=1.0,bootstrap=True)
        bag_model.fit(self.x_trg,self.y_trg)

        #Determining accuracy and creating confusion matrix of the model
        bag_pred=bag_model.predict(self.x_test)
        y_probas = bag_model.predict_proba(self.x_test)

        #Determining accuracy of the test image
        bag_pred_item=bag_model.predict(item)
        bag_value=bag_pred_item[0]

        return bag_model,bag_pred,y_probas,title,self.categories[bag_value],self.x_trg,self.y_trg,self.y_test
