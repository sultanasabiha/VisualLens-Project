#Importing necessary libraries
from keras.preprocessing import image as kimage
import numpy as np
#Importing necessary libraries for image similarity
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,manhattan_distances
#Creating a list of all the images
import pandas as pd
import cv2
import os
class Similar():
    def __init__(self,image_dir,test):
        # Directory where images are located
        self.image_dir=image_dir
        # Directory where test image is located
        self.test=test
        self.name=""
        self.images_dict=dict()
        self.pos=0
    def loadimages(self):
        #Load all images
        for filename in os.listdir(self.image_dir):
            img_path = self.image_dir+'/' +filename
            #Generally accepted image size for trained models is 224 X 224 px
            image=kimage.load_img(img_path,target_size=(224,224))
            #Converting image to array form
            image=kimage.img_to_array(image)
            #Map image and id
            self.images_dict[img_path]=image

    def getMatrix(self):
        self.loadimages()
        print(self.test)
        image_matrix1=np.zeros(([len(self.images_dict),150528]))
        for i, (num,image) in enumerate (self.images_dict.items()):
            if num == self.test:
                self.pos=i
            #Flatten the matrix
            image_matrix1[i,:]=image.flatten()
            i=i+1
        return image_matrix1
    
    def cosine(self,image_matrix):
        title="Cosine Similarity"+self.name

        #Cosine similarity for image similarity
        print("--------------using cosine similarity----------------")
        cos=cosine_similarity(image_matrix)

        #Converting to a dataframe
        cos_df=pd.DataFrame(cos)

        #Determining information of the seventh product 
        product_info=cos_df.iloc[self.pos].values

        #Displaying the index of 4 similar images
        sim_img_index=np.argsort(-product_info)[1:5]
        print(sim_img_index)

        return sim_img_index, title
        
    def euclidean(self,image_matrix):
        title="Euclidean Similarity"+self.name

        #Euclidean Distance for image similarity
        print("--------------using euclidean similarity----------------")
        eu=euclidean_distances(image_matrix)

        #Converting to a dataframe
        eu_df=pd.DataFrame(eu)


        #Determining information of the seventh product 
        product_info=eu_df.iloc[self.pos].values

        #Displaying the index of 4 similar images
        sim_img_index=np.argsort(product_info)[1:5]
        print(sim_img_index)

        return sim_img_index,title

    def getImages(self,img_index):
        #Displaying images
        images_list=[]
        for i, (path,image) in enumerate(self.images_dict.items()):
            print(i,img_index)
            if i in list(img_index):
                images_list.append(cv2.cvtColor(cv2.resize(cv2.imread(path),(224,224)),cv2.COLOR_BGR2RGB))
        return images_list


    def manhattan(self,image_matrix):
        title="Manhattan Similarity"+self.name

        #Manhattan Distance for image similarity
        print("--------------using manhattan similarity----------------")
        man=manhattan_distances(image_matrix)

        #Converting to a dataframe
        man_df=pd.DataFrame(man)

        #Determining information of the seventh product 
        product_info=man_df.iloc[self.pos].values

        #Displaying the index of 4 similar images
        sim_img_index=np.argsort(product_info)[1:5]
        print(sim_img_index)
        return sim_img_index,title

    def mobnet(self):
        self.name=" with MobileNet"
        
        #Using MobileNet algorithm
        from keras.applications import MobileNet
        from keras.applications.mobilenet import preprocess_input
        images_dict=dict()

        #Process all images according to MobileNet Algorithm
        for filename in os.listdir(self.image_dir):
            mix_image = self.image_dir+'/' +filename
            image=kimage.load_img(mix_image,target_size=(224,224))
            image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
            num=mix_image.split('\\')[-1].split('.')[0]
            images_dict[num]=image

        #Creatimg mobilenet model
        mobile_net_model=MobileNet(include_top=False,weights='imagenet')

        #Initialize matirx to 0. Dimension of matrix is dependent on model used
        image_matrix2=np.zeros([len(images_dict),50176])
        for i, (num,image) in enumerate(images_dict.items()):
            if num == self.test:
                self.pos=i
            image_matrix2[i,:]=mobile_net_model.predict(image).ravel()
        return image_matrix2
    def resnet(self):
        self.name=" with ResNet50"

        #Using ResNet50 algorithm
        from keras.applications import ResNet50
        from keras.applications.resnet50 import preprocess_input
        images_dict=dict()

        #Process all images according to MobileNet Algorithm
        for filename in os.listdir(self.image_dir):
            mix_image = self.image_dir+'/' +filename
            image=kimage.load_img(mix_image,target_size=(224,224))
            image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
            num=mix_image.split('\\')[-1].split('.')[0]
            images_dict[num]=image

        #Creatimg mobilenet model
        res_net_model=ResNet50(include_top=False,weights='imagenet')

        #Initialize matirx to 0. Dimension of matrix is dependent on model used
        image_matrix3=np.zeros([len(images_dict),100352])
        for i, (num,image) in enumerate(images_dict.items()):
            if num == self.test:
                self.pos=i
            image_matrix3[i,:]=res_net_model.predict(image).ravel()
        return image_matrix3

    def vgg16(self):
        self.name=" with VGG16"

        #Using VGG16 algorithm
        from keras.applications import VGG16
        from keras.applications.vgg16 import preprocess_input
        images_dict=dict()

        #Process all images according to MobileNet Algorithm
        for filename in os.listdir(self.image_dir):
            mix_image = self.image_dir+'/' +filename
            image=kimage.load_img(mix_image,target_size=(224,224))
            image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
            num=mix_image.split('\\')[-1].split('.')[0]
            images_dict[num]=image

        #Creatimg mobilenet model
        vgg16_net_model=VGG16(include_top=False,weights='imagenet')

        #Initialize matirx to 0. Dimension of matrix is dependent on model used
        image_matrix4=np.zeros([len(images_dict),25088])
        for i, (num,image) in enumerate(images_dict.items()):
            if num == self.test:
                self.pos=i
            image_matrix4[i,:]=vgg16_net_model.predict(image).ravel()
        return image_matrix4

    def vgg19(self):
        self.name=" with VGG19"

        #Using VGG19 algorithm
        from keras.applications import VGG19
        from keras.applications.vgg19 import preprocess_input
        images_dict=dict()

        #Process all images according to MobileNet Algorithm
        for filename in os.listdir(self.image_dir):
            mix_image = self.image_dir+'/' +filename
            image=kimage.load_img(mix_image,target_size=(224,224))
            image=preprocess_input(np.expand_dims(kimage.img_to_array(image),axis=0))
            num=mix_image.split('\\')[-1].split('.')[0]
            images_dict[num]=image

        #Creatimg mobilenet model
        vgg19_net_model=VGG19(include_top=False,weights='imagenet')

        #Initialize matirx to 0. Dimension of matrix is dependent on model used
        image_matrix5=np.zeros([len(images_dict),25088])
        for i, (num,image) in enumerate(images_dict.items()):
            if num == self.test:
                self.pos=i
            image_matrix5[i,:]=vgg19_net_model.predict(image).ravel()
        return image_matrix5
