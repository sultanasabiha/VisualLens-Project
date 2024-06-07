import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
matplotlib.use("TKAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from scikitplot.metrics import plot_confusion_matrix, plot_roc,plot_precision_recall,plot_silhouette,plot_calibration_curve,plot_cumulative_gain
from scikitplot.estimators import plot_learning_curve
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,multilabel_confusion_matrix,davies_bouldin_score,calinski_harabasz_score
import numpy as np
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from skimage.metrics import structural_similarity as ssim

import cv2
from collections import defaultdict
from sklearn.metrics import pairwise_distances
import sys,os

class VisualizeClass:
    def __init__(self,type):
        self.top=tk.Toplevel()
        iconPath = self.resource_path('visual_lens.ico')
        self.top.iconbitmap(iconPath)
        self.top.iconify()
        self.notebook=ttk.Notebook(self.top)
        self.top.geometry('672x672')
        self.notebook.pack(expand=True,fill='both')
        self.type=type
        self.names=[]
        self.accs=[]
        self.f1s=[]
        self.precisions=[]
        self.recalls=[]
        self.probas=[]
        self.y_test=0
    def resource_path(delf,relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    def plot_results(self,clf,pred,probas,name,category,x_trg,y_trg,y_test):
        tab=TabClass(self.notebook,self.type)
        tab.plot(clf,pred,probas,name,category,x_trg,y_trg,y_test)
        self.notebook.add(tab,text=name)
        self.accs.append(accuracy_score(y_test,pred))
        self.names.append(name)
        self.f1s.append(f1_score(y_test,pred,average='weighted'))
        self.precisions.append(precision_score(y_test,pred,average='weighted'))
        self.recalls.append(recall_score(y_test,pred,average='weighted'))
        self.probas.append(probas)
        self.y_test=y_test
    def plot_comparison(self):
        tab=TabClass(self.notebook,self.type)
        tab.pieplot(self.accs,self.names,self.f1s,self.precisions,self.recalls,self.probas,self.y_test)
        self.notebook.add(tab,text="Comparison")
        self.top.deiconify()

class TabClass(tk.Frame):
    def __init__(self, parent,type):
        super().__init__(parent)
        self.pack(expand=True,fill=tk.BOTH)
        self.outercanvas=tk.Canvas(self)
        self.outercanvas.pack(expand=True,fill=tk.BOTH)
        self.innercanvas=tk.Canvas(self.outercanvas)
        self.outercanvas.create_window(5,5,anchor=tk.NW,window=self.innercanvas)
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.outercanvas.yview)
        self.scrollbar.place(relx=1,rely=0,relheight=1,anchor='ne')
        self.outercanvas.config(yscrollcommand=self.scrollbar.set)
        self.outercanvas.bind("<Configure>",self.on_configure)
        self.type=type

    def on_configure(self,event):
        self.outercanvas.configure(scrollregion=self.outercanvas.bbox("all"))  

    def setCanvas(self):
        canvas = FigureCanvasTkAgg(Figure(figsize=(6, 5)), master=self.innercanvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH,padx=10,pady=10,ipadx=10,ipady=10)
        canvas.draw()
        toolbar=NavigationToolbar2Tk(canvas,self.innercanvas,pack_toolbar=False)
        toolbar.pack(anchor='w',fill='x')
        return canvas
    
    def plot(self,clf,pred,probas,name,category,x_trg,y_trg,y_test):

        tk.Label(self,text="Category of test image:"+category).pack(side=tk.TOP, fill=tk.BOTH,padx=10,pady=10,ipadx=10,ipady=10)

        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        plot_confusion_matrix(y_test, pred,normalize=True,ax=ax1)
        ax1.set_title('Confusion Matrix for '+name)

        if self.type==2:
            cm = multilabel_confusion_matrix(y_test, pred)
            count=len(cm)
            canvas=self.setCanvas()
            rows=len(cm)//2 if (len(cm)%2==0) else (len(cm)//2)+1
            # Plot confusion matrix
            axes = canvas.figure.subplots(nrows=rows, ncols=2)
            c=0
            for i in range(rows):
                for j in range(2):
                    ax = axes[i,j]
                    if count>0:
                        ax.matshow(cm[c], cmap=plt.cm.Blues, alpha=0.7)
                        for x in range(cm[c].shape[1]):
                            for y in range(cm[c].shape[0]):
                                ax.text(x, y, f"{cm[c][y, x]}", va='center', ha='center')
                        ax.set_xlabel('Predicted label')
                        ax.set_ylabel('True label')
                        ax.set_title('Class '+str(c))
                        ax.set_xticks([0, 1])
                        ax.set_yticks([0, 1])
                        ax.xaxis.set_ticks_position('bottom')
                        c+=1
                        count-=1
                    else:
                        ax.set_yticks([])
                        ax.set_xticks([])
                        ax.set_axis_off()
            canvas.figure.suptitle("Confusion Matrix for each Class")

            canvas=self.setCanvas()
            ax=canvas.figure.add_subplot(111)
            for i in range(len(clf.classes_)):
                fraction_of_positives, mean_predicted_value = calibration_curve(y_test == clf.classes_[i], probas[:, i], n_bins=4)
                ax.plot(mean_predicted_value, fraction_of_positives, marker='o',label='Class {}'.format(clf.classes_[i]))

            ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title('Calibration Curves for the classes')
            ax.legend()

        canvas=self.setCanvas()
        ax2=canvas.figure.add_subplot(111)
        plot_roc(y_test, probas,ax=ax2)
        ax2.set_title('ROC Curve for '+name)

        canvas=self.setCanvas()
        ax3=canvas.figure.add_subplot(111)
        plot_learning_curve(clf,x_trg, y_trg,ax=ax3)
        ax3.set_title('Learning curve for '+name)

        canvas=self.setCanvas()
        ax4=canvas.figure.add_subplot(111)
        plot_precision_recall(y_test,probas,ax=ax4)
        ax4.set_title('Precision Recall curve for'+name)
        
        if self.type==1:
            canvas=self.setCanvas()
            ax1=canvas.figure.add_subplot(111)      
            plot_cumulative_gain(y_test,probas,ax=ax1)

    def pieplot(self,accs,names,f1s,precisions,recalls,probas,y_test):
        count=len(accs)
     
        explode=[0]*count
        maxval=max(accs)
        for i in range(count):
            if accs[i]==maxval:
                explode[i]= 0.1 
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        ax1.pie(accs,explode=explode,autopct=lambda pct: f"{pct:.3f}%",shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9})
        ax1.axis('equal')
        ax1.set_title("Accuracy of the Algorithms")
        ax1.legend(title='Models',loc='lower right',labels=names,bbox_to_anchor=(0.5,0.5))


        explode=[0]*count
        maxval=max(f1s)
        for i in range(count):
            if f1s[i]==maxval:
                explode[i]= 0.1 
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        ax1.pie(f1s,explode=explode,autopct=lambda pct: f"{pct:.3f}%",shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9})
        ax1.axis('equal')
        ax1.set_title("F1 score of the Algorithms")
        ax1.legend(title='Models',loc='lower right',labels=names,bbox_to_anchor=(0.5,0.5))

        explode=[0]*count
        maxval=max(precisions)
        for i in range(count):
            if precisions[i]==maxval:
                explode[i]= 0.1
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        ax1.pie(precisions,explode=explode,autopct=lambda pct: f"{pct:.3f}%",shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9})
        ax1.axis('equal')
        ax1.set_title("Precision of the Algorithms")
        ax1.legend(title='Models',loc='lower right',labels=names,bbox_to_anchor=(0.5,0.5))

        explode=[0]*count
        maxval=max(recalls)
        for i in range(count):
            if recalls[i]==maxval:
                explode[i]= 0.1
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        ax1.pie(recalls,explode=explode,autopct=lambda pct: f"{pct:.3f}%",shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9})
        ax1.axis('equal')
        ax1.set_title("Recall of the Algorithms")
        ax1.legend(title='Models',loc='lower right',labels=names,bbox_to_anchor=(0.5,0.5))

        if self.type==1:
            canvas=self.setCanvas()
            ax1=canvas.figure.add_subplot(111)      
            plot_calibration_curve(y_test,probas,names,ax=ax1)

class VisualizeSim:
    def __init__(self,test):
        self.top=tk.Toplevel()
        iconPath = self.resource_path('visual_lens.ico')
        self.top.iconbitmap(iconPath)
        self.top.iconify()
        self.notebook=ttk.Notebook(self.top)
        self.top.geometry('672x672')
        self.notebook.pack(expand=True,fill='both')
        self.org=cv2.cvtColor(cv2.resize(cv2.imread(test),(224,224)),cv2.COLOR_BGR2RGB)
 
        self.ssim_images=defaultdict(list)
        self.orb_images=defaultdict(list)
        self.corr_images=defaultdict(list)
        self.bhatta_images=defaultdict(list)
        self.intersect_images=defaultdict(list)

    def resource_path(delf,relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    def plot_results(self,images_list,name):
        tab=TabSim(self.notebook)
        tab.plot(images_list,self.org)
        self.notebook.add(tab,text=name)

        for i in range(len(images_list)):
            ssim,num_matches,corr,bhatta,intersect=self.calculate_metrics(self.org,images_list[i])

            self.ssim_images[name].append(ssim)
            self.orb_images[name].append(num_matches)
            self.corr_images[name].append(corr)
            self.bhatta_images[name].append(bhatta)
            self.intersect_images[name].append(intersect)

    def calculate_metrics(self,img1, img2):

        win_size = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
        if win_size % 2 == 0:
            win_size -= 1
        (score, diff) = ssim(img1, img2, full=True,win_size=win_size,channel_axis=2)
        num_matches=self.calculate_orb(img1,img2)
        corr,bhatta,intersect=self.histogram_comparison(img1,img2)


        return score,num_matches,corr,bhatta,intersect
    def calculate_orb(self,image1, image2):
        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find key points and descriptors
        kp1, des1 = orb.detectAndCompute(image1, None)
        kp2, des2 = orb.detectAndCompute(image2, None)

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        return len(matches)

    def histogram_comparison(self,image1, image2):

        # Convert images to HSV color space
        hsvA = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        hsvB = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

        # Compute the histogram for each image
        histA = cv2.calcHist([hsvA], [0, 1], None, [50, 60], [0, 180, 0, 256])
        histB = cv2.calcHist([hsvB], [0, 1], None, [50, 60], [0, 180, 0, 256])

        # Normalize the histograms
        cv2.normalize(histA, histA, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(histB, histB, 0, 1, cv2.NORM_MINMAX)

        # Compare histograms using the correlation method
        correlation = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
        bhatta=cv2.compareHist(histA,histB,cv2.HISTCMP_BHATTACHARYYA)
        intersect=cv2.compareHist(histA,histB,cv2.HISTCMP_INTERSECT)
        return correlation,bhatta,intersect

    def plot_comparison(self):
        tab=TabSim(self.notebook)
        tab.barplot(self.ssim_images,self.orb_images,self.corr_images,self.bhatta_images,self.intersect_images)
        self.notebook.add(tab,text="Comparison")
        self.top.deiconify()

class TabSim(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(expand=True,fill=tk.BOTH)
        self.outercanvas=tk.Canvas(self)
        self.outercanvas.pack(expand=True,fill=tk.BOTH)
        self.innercanvas=tk.Canvas(self.outercanvas)
        self.outercanvas.create_window(5,5,anchor=tk.NW,window=self.innercanvas)
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.outercanvas.yview)
        self.scrollbar.place(relx=1,rely=0,relheight=1,anchor='ne')
        self.outercanvas.config(yscrollcommand=self.scrollbar.set)
        self.outercanvas.bind("<Configure>",self.on_configure)

    def on_configure(self,event):
        self.outercanvas.configure(scrollregion=self.outercanvas.bbox("all"))  

    def setCanvas(self):
        canvas = FigureCanvasTkAgg(Figure(figsize=(6, 5)), master=self.innercanvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH,padx=10,pady=10,ipadx=10,ipady=10)
        canvas.draw()
        toolbar=NavigationToolbar2Tk(canvas,self.innercanvas,pack_toolbar=False)
        toolbar.pack(anchor='w',fill='x')
        return canvas
    
    def plot(self,images,org):
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        ax1.imshow(org)
        ax1.set_title('Original Image')
        ax1.set_yticks([])
        ax1.set_xticks([])
                
        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                ax.imshow(images[c])
                ax.set_title('Image '+str(c))
                c+=1
                ax.set_yticks([])
                ax.set_xticks([])
        canvas.figure.suptitle("Predicted Similar Images")

    def barplot(self,ssim_images,orb_images,corr_images,bhatta_images,intersect_images):
        
        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        canvas.figure.subplots_adjust(hspace=0.3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(ssim_images)))  # Generating a range of colors

        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                img=[]
                for k in ssim_images.keys():
                    img.append(ssim_images[k][c])
                ax.bar(ssim_images.keys(),img,label=ssim_images.keys(),color=colors)
                highlight_index=img.index(max(img))
                # Get the position of the highlighted rectangle
                highlighted_rectangle = ax.patches[highlight_index]
                x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
                y = highlighted_rectangle.get_height()

                # Add a marker on top of the highlighted rectangle
                ax.scatter(x, y, color='black', marker='o', zorder=3)

                ax.set_xticks([])
                ax.set_title('Image '+str(c+1))
                c+=1
        axes[1,1].legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        canvas.figure.suptitle("Structural Similarity Index (SSIM)")

        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        canvas.figure.subplots_adjust(hspace=0.3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(orb_images)))  # Generating a range of colors

        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                img=[]
                for k in orb_images.keys():
                    img.append(orb_images[k][c])
                ax.bar(orb_images.keys(),img,label=orb_images.keys(),color=colors)
                highlight_index=img.index(max(img))
                # Get the position of the highlighted rectangle
                highlighted_rectangle = ax.patches[highlight_index]
                x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
                y = highlighted_rectangle.get_height()

                # Add a marker on top of the highlighted rectangle
                ax.scatter(x, y, color='black', marker='o', zorder=3)

                ax.set_xticks([])
                ax.set_title('Image '+str(c+1))
                c+=1
        plt.gca().spines['bottom'].set_color('black')
        axes[1,1].legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        canvas.figure.suptitle("Feature Based Similarity")

        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        canvas.figure.subplots_adjust(hspace=0.3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(corr_images)))  # Generating a range of colors

        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                img=[]
                for k in corr_images.keys():
                    img.append(corr_images[k][c])
                ax.bar(corr_images.keys(),img,label=corr_images.keys(),color=colors)
                highlight_index=img.index(max(img))
                # Get the position of the highlighted rectangle
                highlighted_rectangle = ax.patches[highlight_index]
                x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
                y = highlighted_rectangle.get_height()

                # Add a marker on top of the highlighted rectangle
                ax.scatter(x, y, color='black', marker='o', zorder=3)

                ax.set_xticks([])
                ax.set_title('Image '+str(c+1))
                c+=1
        axes[1,1].legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        canvas.figure.suptitle("Histogram-Coorelation Comparison")
        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        canvas.figure.subplots_adjust(hspace=0.3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(bhatta_images)))  # Generating a range of colors

        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                img=[]
                for k in bhatta_images.keys():
                    img.append(bhatta_images[k][c])
                ax.bar(bhatta_images.keys(),img,label=bhatta_images.keys(),color=colors)
                highlight_index=img.index(min(img))
                # Get the position of the highlighted rectangle
                highlighted_rectangle = ax.patches[highlight_index]
                x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
                y = highlighted_rectangle.get_height()

                # Add a marker on top of the highlighted rectangle
                ax.scatter(x, y, color='black', marker='o', zorder=3)

                ax.set_xticks([])
                ax.set_title('Image '+str(c+1))
                c+=1
        axes[1,1].legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        canvas.figure.suptitle("Histogram Based Comparison on Bhattacharya Distance")

        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        canvas.figure.subplots_adjust(hspace=0.3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(intersect_images)))  # Generating a range of colors

        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                img=[]
                for k in intersect_images.keys():
                    img.append(intersect_images[k][c])
                ax.bar(intersect_images.keys(),img,label=intersect_images.keys(),color=colors)
                highlight_index=img.index(max(img))
                # Get the position of the highlighted rectangle
                highlighted_rectangle = ax.patches[highlight_index]
                x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
                y = highlighted_rectangle.get_height()

                # Add a marker on top of the highlighted rectangle
                ax.scatter(x, y, color='black', marker='o', zorder=3)

                ax.set_xticks([])
                ax.set_title('Image '+str(c+1))
                c+=1
        axes[1,1].legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        canvas.figure.suptitle("Histogram-Intersection Comparison")

class VisualizeClus:
    def __init__(self,n_clusters):
        self.top=tk.Toplevel()
        iconPath = self.resource_path('visual_lens.ico')
        self.top.iconbitmap(iconPath)
        self.top.iconify()
        self.notebook=ttk.Notebook(self.top)
        self.top.geometry('672x672')
        self.notebook.pack(expand=True,fill='both')
        self.n_clusters=n_clusters
        self.names=[]
        self.db_index=[]
        self.ch_index=[]
        self.coh_index=[]
        self.sep_index=[]
        self.dunn_index=[]
    def resource_path(delf,relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def plot_results(self,images_df,name,mat,y_labels):
        self.tab=TabClus(self.notebook)
        self.tab.plot(images_df,self.n_clusters,mat,y_labels)
        self.notebook.add(self.tab,text=name)
        self.calculate_metrics(mat,y_labels)
        self.names.append(name)      

    def calculate_metrics(self,mat,labels):
        db = davies_bouldin_score(mat, labels)
        ch= calinski_harabasz_score(mat, labels)
        coh = self.cohesion(mat,labels)
        sep = self.separation(mat,labels)
        dunn = self.dunn(mat,labels)

        self.db_index.append(db)
        self.ch_index.append(ch)
        self.coh_index.append(coh)
        self.sep_index.append(sep)
        self.dunn_index.append(dunn)

    
    def cohesion(self,X, labels):
        total_cohesion = 0
        for cluster_label in np.unique(labels):
            cluster_points = X[labels == cluster_label]
            centroid = np.mean(cluster_points, axis=0)
            total_cohesion += np.sum(np.linalg.norm(cluster_points - centroid, axis=1))
        return total_cohesion / len(X)
    def separation(self,X, labels):
        centroids = []
        for label in np.unique(labels):
            cluster_points = X[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        
        separation_sum = 0
        total_combinations = len(centroids) * (len(centroids) - 1) / 2
        
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                separation_sum += np.linalg.norm(centroids[i] - centroids[j])
        
        return separation_sum / total_combinations

    def dunn(self,X, labels):
        intra_cluster_distances = []
        for label in np.unique(labels):
            cluster_points = X[labels == label]
            intra_cluster_distances.append(np.max(pairwise_distances(cluster_points)))
        
        max_intra_cluster_distance = np.max(intra_cluster_distances)
        
        inter_cluster_distances = []
        for i in range(len(np.unique(labels))):
            for j in range(i + 1, len(np.unique(labels))):
                centroid_i = np.mean(X[labels == i], axis=0)
                centroid_j = np.mean(X[labels == j], axis=0)
                inter_cluster_distances.append(np.linalg.norm(centroid_i - centroid_j))
        
        min_inter_cluster_distance = np.min(inter_cluster_distances)
        
        return min_inter_cluster_distance / max_intra_cluster_distance

    def plot_dendo(self,mat):
        self.tab.dendo(mat)

    def plot_comparison(self):
        tab=TabClus(self.notebook)
        tab.barplot(self.db_index,self.ch_index,self.coh_index,self.sep_index,self.dunn_index,self.names)
        self.notebook.add(tab,text="Comparison")
        self.top.deiconify()

class TabClus(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(expand=True,fill=tk.BOTH)
        self.outercanvas=tk.Canvas(self)
        self.outercanvas.pack(expand=True,fill=tk.BOTH)
        self.innercanvas=tk.Canvas(self.outercanvas)
        self.outercanvas.create_window(5,5,anchor=tk.NW,window=self.innercanvas)
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.outercanvas.yview)
        self.scrollbar.place(relx=1,rely=0,relheight=1,anchor='ne')
        self.outercanvas.config(yscrollcommand=self.scrollbar.set)
        self.outercanvas.bind("<Configure>",self.on_configure)
    def on_configure(self,event):
        self.outercanvas.configure(scrollregion=self.outercanvas.bbox("all"))  

    def setCanvas(self):
        canvas = FigureCanvasTkAgg(Figure(figsize=(6, 5)), master=self.innercanvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH,padx=10,pady=10,ipadx=10,ipady=10)
        canvas.draw()
        toolbar=NavigationToolbar2Tk(canvas,self.innercanvas,pack_toolbar=False)
        toolbar.pack(anchor='w',fill='x')
        return canvas
    def dendo(self,mat):
        from scipy.cluster.hierarchy import ward,dendrogram
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        result=ward(mat)
        dendrogram(result,ax=ax1)
        ax1.set_xlabel("Observations")
        ax1.set_ylabel("Clusters")

    def plot(self,images_df,n_clusters,mat,y_labels):
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        plot_silhouette(mat,y_labels,ax=ax1)

        #Displaying the images belonging to the clusters
        for i in range(n_clusters):
            cluster=images_df[(images_df['Predicted']==i)]
            images_cluster=[]

            for path in cluster['Path']:
                images_cluster.append(cv2.cvtColor(cv2.resize(cv2.imread(path),(224,224)),cv2.COLOR_BGR2RGB))
            canvas = FigureCanvasTkAgg(Figure(figsize=(6, 5)), master=self.innercanvas)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH,padx=10,pady=10,ipadx=10,ipady=10)
            canvas.draw()
            count=len(images_cluster)
            rows=count//4 if (count%4==0) else (count//4)+1
            axes = canvas.figure.subplots(nrows=rows, ncols=4)
            c=0
            if rows==1:
                for k in range(4):
                    ax=axes[k]
                    if count>0:
                        ax.imshow(images_cluster[c])
                        c+=1
                        ax.set_yticks([])
                        ax.set_xticks([])
                        count-=1
                    else:
                        ax.set_yticks([])
                        ax.set_xticks([])
                        ax.set_axis_off()
            else:
                for j in range(rows):
                    for k in range(4):
                        ax = axes[j,k]
                        if count>0:
                            ax.imshow(images_cluster[c])
                            c+=1
                            ax.set_yticks([])
                            ax.set_xticks([])
                            count-=1
                        else:
                            ax.set_yticks([])
                            ax.set_xticks([])
                            ax.set_axis_off()
            canvas.figure.suptitle("Cluster "+str(i))

    def barplot(self,db_index,ch_index,coh_index,sep_index,dunn_index,names):
        
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        colors = plt.cm.viridis(np.linspace(0, 1, len(db_index)))  # Generating a range of colors

        ax1.bar(names,db_index,label=names,color=colors)
        ax1.set_xticks([])
        highlight_index=db_index.index(min(db_index))
        # Get the position of the highlighted rectangle
        highlighted_rectangle = ax1.patches[highlight_index]
        x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
        y = highlighted_rectangle.get_height()

        # Add a marker on top of the highlighted rectangle
        ax1.scatter(x, y, color='black', marker='o', zorder=3)


        ax1.legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        ax1.set_title("Davies–Bouldin Index")
        
        
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        colors = plt.cm.viridis(np.linspace(0, 1, len(ch_index)))  # Generating a range of colors
        ax1.set_xticks([])

        ax1.bar(names,ch_index,label=names,color=colors)
        highlight_index=ch_index.index(max(ch_index))
        # Get the position of the highlighted rectangle
        highlighted_rectangle = ax1.patches[highlight_index]
        x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
        y = highlighted_rectangle.get_height()

        # Add a marker on top of the highlighted rectangle
        ax1.scatter(x, y, color='black', marker='o', zorder=3)

        ax1.legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        ax1.set_title("Calinski-Harabasz Index")
        
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        colors = plt.cm.viridis(np.linspace(0, 1, len(ch_index)))  # Generating a range of colors
        ax1.set_xticks([])

        ax1.bar(names,coh_index,label=names,color=colors)
        highlight_index=coh_index.index(min(coh_index))
        # Get the position of the highlighted rectangle
        highlighted_rectangle = ax1.patches[highlight_index]
        x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
        y = highlighted_rectangle.get_height()

        # Add a marker on top of the highlighted rectangle
        ax1.scatter(x, y, color='black', marker='o', zorder=3)

        ax1.legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        ax1.set_title("Cohesion")

        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        colors = plt.cm.viridis(np.linspace(0, 1, len(ch_index)))  # Generating a range of colors
        ax1.set_xticks([])

        ax1.bar(names,sep_index,label=names,color=colors)
        highlight_index=sep_index.index(max(sep_index))
        # Get the position of the highlighted rectangle
        highlighted_rectangle = ax1.patches[highlight_index]
        x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
        y = highlighted_rectangle.get_height()

        # Add a marker on top of the highlighted rectangle
        ax1.scatter(x, y, color='black', marker='o', zorder=3)

        ax1.legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        ax1.set_title("Separation")

        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        colors = plt.cm.viridis(np.linspace(0, 1, len(ch_index)))  # Generating a range of colors
        ax1.set_xticks([])

        ax1.bar(names,dunn_index,label=names,color=colors)
        highlight_index=dunn_index.index(max(dunn_index))
        # Get the position of the highlighted rectangle
        highlighted_rectangle = ax1.patches[highlight_index]
        x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
        y = highlighted_rectangle.get_height()

        # Add a marker on top of the highlighted rectangle
        ax1.scatter(x, y, color='black', marker='o', zorder=3)

        ax1.legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        ax1.set_title("Dunn Index")
  
