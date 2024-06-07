from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering
import numpy
import pandas
import os
from keras.preprocessing import image as kimage


class Cluster:
    def __init__(self,image_dir,n_clusters):

        # Directory where images are located
        self.images_dir=image_dir
        # No. of clusters present
        self.n_clusters=n_clusters
                
        self.images_dict=dict()
        self.name=""
    def loadimages(self):
    
        #Load all images
        for filename in os.listdir(self.images_dir):
            img_path = self.images_dir+'/'+filename
            #Generally accepted image size for trained models is 224 X 224 px
            image=kimage.load_img(img_path,target_size=(224,224))
            #Converting image to array form
            image=kimage.img_to_array(image)
            #Map image and path
            self.images_dict[img_path]=image
    
    def getMatrix(self):
        self.loadimages()
        image_matrix=numpy.zeros(([len(self.images_dict),150528]))
        for i, (path,image) in enumerate (self.images_dict.items()):
            #Flatten the matrix
            image_matrix[i,:]=image.flatten()
            i=i+1
        return image_matrix
    def kmeans(self,image_matrix):
        title="K-Means"+self.name
        kmeans=KMeans(n_clusters=self.n_clusters,random_state=10)
        y_means=kmeans.fit_predict(image_matrix)
        #print("The cluster of images:",y_means)

        paths=list(self.images_dict.keys())
        data={'Path':paths,'Predicted':y_means}
        kmeansdf=pandas.DataFrame(data,columns=['Path','Predicted'])
        #Storing in a dataframe so the images can be accessed according to their predicted cluster 
        #print("Details of clusters are:\n",kmeansdf["Predicted"].value_counts())

        return kmeansdf,title,y_means
    def agglo(self,image_matrix):
        title="Agglomerative Hierarchical Clustering"+self.name
        agglo=AgglomerativeClustering(n_clusters=self.n_clusters)
        y_agglo=agglo.fit_predict(image_matrix)
        #print("The cluster of images:",y_agglo)

        paths=list(self.images_dict.keys())
        data={'Path':paths,'Predicted':y_agglo}
        agglodf=pandas.DataFrame(data,columns=['Path','Predicted'])
        #Storing in a dataframe so the images can be accessed according to their predicted cluster 
        #print("Details of clusters are:\n",agglodf["Predicted"].value_counts())

        return agglodf,title,y_agglo
        
    def spec(self,image_matrix):
        title="Spectral Clustering"+self.name
        spec=SpectralClustering(n_clusters=self.n_clusters)
        y_spec=spec.fit_predict(image_matrix)
        #print("The cluster of images:",y_spec)

        paths=list(self.images_dict.keys())
        data={'Path':paths,'Predicted':y_spec}
        agglodf=pandas.DataFrame(data,columns=['Path','Predicted'])
        #Storing in a dataframe so the images can be accessed according to their predicted cluster 
        #print("Details of clusters are:\n",agglodf["Predicted"].value_counts())

        return agglodf,title,y_spec

    def mobnet(self):
        self.name=" with Mobilenet"
        from keras.applications.mobilenet import preprocess_input
        from keras.preprocessing import image as kimage
        from keras.applications import MobileNet

        mobile_net_model=MobileNet(include_top=False,weights='imagenet')
        mobilenet_matrix=numpy.zeros([len(self.images_dict),50176])
        i=0
        #Process all images according to MobileNet Algorithm
        for filename in os.listdir(self.images_dir):
            mix_image = self.images_dir+'/' +filename
            image=kimage.load_img(mix_image,target_size=(224,224))
            image=preprocess_input(numpy.expand_dims(kimage.img_to_array(image),axis=0))
            mobilenet_matrix[i,:]=mobile_net_model.predict(image).ravel()
            i+=1
        return mobilenet_matrix

    def resnet(self):
        self.name=" with ResNet50"

        from keras.applications.resnet50 import preprocess_input
        from keras.applications import ResNet50

        resnet_model=ResNet50(include_top=False,weights='imagenet')
        resnet_matrix=numpy.zeros([len(self.images_dict),100352])
        i=0
        #Process all images according to MobileNet Algorithm
        for filename in os.listdir(self.images_dir):
            mix_image = self.images_dir +'/'+filename
            image=kimage.load_img(mix_image,target_size=(224,224))
            image=preprocess_input(numpy.expand_dims(kimage.img_to_array(image),axis=0))
            resnet_matrix[i,:]=resnet_model.predict(image).ravel()
            i+=1
        return resnet_matrix
    
    def vgg16(self):
        self.name=" with VGG16"

        from keras.applications.vgg16 import preprocess_input
        from keras.applications import VGG16

        vgg16_model=VGG16(include_top=False,weights='imagenet')
        vgg16_matrix=numpy.zeros([len(self.images_dict),25088])
        i=0
        #Process all images according to MobileNet Algorithm
        for filename in os.listdir(self.images_dir):
            mix_image = self.images_dir+'/' +filename
            image=kimage.load_img(mix_image,target_size=(224,224))
            image=preprocess_input(numpy.expand_dims(kimage.img_to_array(image),axis=0))
            vgg16_matrix[i,:]=vgg16_model.predict(image).ravel()
            i+=1
        return vgg16_matrix
    
    def vgg19(self):
        self.name=" with VGG19"

        from keras.applications.vgg19 import preprocess_input
        from keras.applications import VGG19

        vgg19_model=VGG19(include_top=False,weights='imagenet')
        vgg19_matrix=numpy.zeros([len(self.images_dict),25088])
        i=0
        #Process all images according to MobileNet Algorithm
        for filename in os.listdir(self.images_dir):
            mix_image = self.images_dir+'/' +filename
            image=kimage.load_img(mix_image,target_size=(224,224))
            image=preprocess_input(numpy.expand_dims(kimage.img_to_array(image),axis=0))
            vgg19_matrix[i,:]=vgg19_model.predict(image).ravel()
            i+=1
        return vgg19_matrix
    
   
