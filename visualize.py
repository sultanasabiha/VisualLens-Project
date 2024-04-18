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
class Visualize:
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

    def plotIndi(self,clf,pred,probas,name,category,x_trg,y_trg,y_test):
        tab=Tab(self.notebook)
        tab.plot(clf,pred,probas,name,category,x_trg,y_trg,y_test)
        self.notebook.add(tab,text=name)
        self.x_trg=x_trg
        self.y_trg=y_trg
        self.accs.append(accuracy_score(y_test,pred))
        self.names.append(name)
        self.f1s.append(f1_score(y_test,pred,average='weighted'))
        self.precisions.append(precision_score(y_test,pred,average='weighted'))
        self.recalls.append(recall_score(y_test,pred,average='weighted'))

    def plotComp(self):
        tab=Tab(self.notebook)
        tab.pieplot(self.x_trg,self.y_trg,self.accs,self.names,self.f1s,self.precisions,self.recalls)
        self.notebook.add(tab,text="Comparison")

class Tab(tk.Frame):
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
        ax1.pie(accs,explode=explode,labels=names,autopct=lambda pct: f"{pct:.3f}%",shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9})
        ax1.axis('equal')
        ax1.set_title("Accuracy of the Algorithms")

        explode=[0]*count
        maxval=max(f1s)
        for i in range(count):
            if f1s[i]==maxval:
                explode[i]= 0.1 
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        ax1.pie(f1s,explode=explode,labels=names,autopct=lambda pct: f"{pct:.3f}%",shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9})
        ax1.axis('equal')
        ax1.set_title("F1 score of the Algorithms")

        explode=[0]*count
        maxval=max(precisions)
        for i in range(count):
            if precisions[i]==maxval:
                explode[i]= 0.1
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        ax1.pie(precisions,explode=explode,labels=names,autopct=lambda pct: f"{pct:.3f}%",shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9})
        ax1.axis('equal')
        ax1.set_title("Precision of the Algorithms")

        explode=[0]*count
        maxval=max(recalls)
        for i in range(count):
            if recalls[i]==maxval:
                explode[i]= 0.1
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        ax1.pie(recalls,explode=explode,labels=names,autopct=lambda pct: f"{pct:.3f}%",shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9})
        ax1.axis('equal')
        ax1.set_title("Recall of the Algorithms")
