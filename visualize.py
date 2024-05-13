import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
matplotlib.use("TKAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from scikitplot.metrics import plot_confusion_matrix, plot_roc,plot_precision_recall
from scikitplot.estimators import plot_learning_curve
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,multilabel_confusion_matrix
import numpy as np
from matplotlib.figure import Figure
from sewar import rmse,uqi,sam,vifp
import cv2
from collections import defaultdict

class VisualizeClass:
    def __init__(self):
        self.top=tk.Toplevel()
        self.notebook=ttk.Notebook(self.top)
        self.top.geometry('672x576')
        self.notebook.pack(expand=True,fill='both')
        self.names=[]
        self.accs=[]
        self.f1s=[]
        self.precisions=[]
        self.recalls=[]

    def plot_results(self,clf,pred,probas,name,category,x_trg,y_trg,y_test):
        tab=TabClass(self.notebook)
        tab.plot(clf,pred,probas,name,category,x_trg,y_trg,y_test)
        self.notebook.add(tab,text=name)
        self.x_trg=x_trg
        self.y_trg=y_trg
        self.accs.append(accuracy_score(y_test,pred))
        self.names.append(name)
        self.f1s.append(f1_score(y_test,pred,average='weighted'))
        self.precisions.append(precision_score(y_test,pred,average='weighted'))
        self.recalls.append(recall_score(y_test,pred,average='weighted'))

    def plot_comparison(self):
        tab=TabClass(self.notebook)
        tab.pieplot(self.x_trg,self.y_trg,self.accs,self.names,self.f1s,self.precisions,self.recalls)
        self.notebook.add(tab,text="Comparison")

class TabClass(tk.Frame):
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
        return canvas
    
    def plot(self,clf,pred,probas,name,category,x_trg,y_trg,y_test):

        tk.Label(self,text="Category of test image:"+category).pack(side=tk.TOP, fill=tk.BOTH,padx=10,pady=10,ipadx=10,ipady=10)

        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        plot_confusion_matrix(y_test, pred,normalize=True,ax=ax1)
        ax1.set_title('Confusion Matrix for '+name)

        cm = multilabel_confusion_matrix(y_test, pred)
        print(cm)
        print(len(cm))
        canvas=self.setCanvas()
        rows=len(cm)//2 if (len(cm)%2==0) else (len(cm)//2)+1
        # Plot confusion matrix
        axes = canvas.figure.subplots(nrows=rows, ncols=2)
        c=0
        for i in range(rows):
            for j in range(2):
                ax = axes[i,j]
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
        
        canvas=self.setCanvas()
        ax2=canvas.figure.add_subplot(111)
        plot_roc(y_test, probas,ax=ax2)
        ax2.set_title('ROC Curve for '+name)

        canvas=self.setCanvas()
        ax3=canvas.figure.add_subplot(111)
        plot_learning_curve(clf, x_trg, y_trg,ax=ax3)
        ax3.set_title('Learning curve for '+name)

        canvas=self.setCanvas()
        ax4=canvas.figure.add_subplot(111)
        plot_precision_recall(y_test,probas,ax=ax4)
        ax4.set_title('Precision Recall curve for'+name)
        
    def pieplot(self,x_trg,y_trg,accs,names,f1s,precisions,recalls):
        count=len(accs)
        x_trg=np.array(x_trg)
        y_trg=np.array(y_trg)
        
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


class VisualizeSim:
    def __init__(self,test):
        self.top=tk.Toplevel()
        self.notebook=ttk.Notebook(self.top)
        self.top.geometry('672x576')
        self.notebook.pack(expand=True,fill='both')
        self.org=cv2.cvtColor(cv2.resize(cv2.imread(test),(224,224)),cv2.COLOR_BGR2RGB)
        self.rmse_images=defaultdict(list)
        self.uqi_images=defaultdict(list)
        self.sam_images=defaultdict(list)
        self.vifp_images=defaultdict(list)

    def plot_results(self,images_list,name):
        tab=TabSim(self.notebook)
        tab.plot(images_list,self.org)
        self.notebook.add(tab,text=name)

        for i in range(len(images_list)):
            rmse,uqi,sam,vifp=self.calculate_metrics(self.org,images_list[i])
            self.rmse_images[name].append(rmse)
            self.uqi_images[name].append(uqi)
            self.sam_images[name].append(sam)
            self.vifp_images[name].append(vifp)
            
    def calculate_metrics(self,img1, img2):
        rmse_scores=rmse(img1, img2)
        uiq_scores=uqi(img1, img2)
        sam_scores=sam(img1, img2)
        vifp_scores=vifp(img1, img2)

        return rmse_scores,uiq_scores,sam_scores,vifp_scores


    def plot_comparison(self):
        tab=TabSim(self.notebook)
        tab.barplot(self.rmse_images,self.uqi_images,self.sam_images,self.vifp_images)
        self.notebook.add(tab,text="Comparison")

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

    def barplot(self,rmse_images,uqi_images,sam_images,vifp_images):
        
        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        canvas.figure.subplots_adjust(hspace=0.3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(rmse_images)))  # Generating a range of colors

        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                img=[]
                for k in rmse_images.keys():
                    img.append(rmse_images[k][c])
                ax.bar(rmse_images.keys(),img,label=rmse_images.keys(),color=colors)
                ax.set_xticks([])
                ax.set_title('Image '+str(c+1))
                highlight_index=img.index(min(img))
                # Get the position of the highlighted rectangle
                highlighted_rectangle = ax.patches[highlight_index]
                x = highlighted_rectangle.get_x() + highlighted_rectangle.get_width() / 2
                y = highlighted_rectangle.get_height()

                # Add a marker on top of the highlighted rectangle
                ax.scatter(x, y, color='black', marker='o', zorder=3)

                c+=1
        axes[1,1].legend(title='Models',loc='lower center',bbox_to_anchor=(0.5,0.5))
        canvas.figure.suptitle("Root Mean Square Error (RMSE) Values")
        
        
        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        canvas.figure.subplots_adjust(hspace=0.3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(rmse_images)))  # Generating a range of colors

        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                img=[]
                for k in uqi_images.keys():
                    img.append(uqi_images[k][c])
                ax.bar(uqi_images.keys(),img,label=uqi_images.keys(),color=colors)
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
        canvas.figure.suptitle("Universal Quality Image Index (UQI) Values")
        
        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        canvas.figure.subplots_adjust(hspace=0.3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(rmse_images)))  # Generating a range of colors

        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                img=[]
                for k in sam_images.keys():
                    img.append(uqi_images[k][c])
                ax.bar(sam_images.keys(),img,label=sam_images.keys(),color=colors)
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
        canvas.figure.suptitle("Spectral Angle Mapper (SAM) Values")

        canvas=self.setCanvas()
        axes = canvas.figure.subplots(nrows=2, ncols=2)
        canvas.figure.subplots_adjust(hspace=0.3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(rmse_images)))  # Generating a range of colors

        c=0
        for i in range(2):
            for j in range(2):
                ax = axes[i,j]
                img=[]
                for k in uqi_images.keys():
                    img.append(vifp_images[k][c])
                ax.bar(vifp_images.keys(),img,label=vifp_images.keys(),color=colors)
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
        canvas.figure.suptitle("Visual Information Fidelity (VIF) Values")
        
        '''
        x = np.arange(len(imgs))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        canvas=self.setCanvas()
        ax = canvas.figure.subplots()

        for model, measurement in ssim_images.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=model)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('SSIM Score')
        ax.set_title('Structured Similarity Index Values')
        ax.set_xticks(x + width, imgs)
        ax.legend(loc='lower right')
        '''
        
