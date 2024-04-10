import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import matplotlib.gridspec as gridspec
matplotlib.use("TKAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from scikitplot.metrics import plot_confusion_matrix, plot_roc,plot_precision_recall
from scikitplot.estimators import plot_learning_curve

from matplotlib.figure import Figure

class Visualize:
    def __init__(self):
        self.top=tk.Toplevel()
        self.notebook=ttk.Notebook(self.top)
        self.notebook.pack(expand=True,fill='both')

    def plotIndi(self,clf,pred,probas,name,category,x_trg,y_trg,y_test):
        tab=Tab(self.notebook)
        tab.plot(clf,pred,probas,name,category,x_trg,y_trg,y_test)
        self.notebook.add(tab,text=name)

    def plotComp(self,names,accs):
        tab=Tab(self.notebook)
        tab.pieplot(names,accs)
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
        plot_confusion_matrix(y_test, pred,ax=ax1)
        ax1.set_title('Confusion Matrix for '+name)
        
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
    def pieplot(self,names,accs):
        explode=[0]*len(accs)
        explode[accs.index(max(accs))]=0.1
        canvas=self.setCanvas()
        ax1=canvas.figure.add_subplot(111)
        ax1.pie(accs,explode=explode,labels=names,autopct=lambda pct: f"{pct:.3f}%",shadow={'ox': -0.04, 'edgecolor': 'none', 'shade': 0.9})
        ax1.axis('equal')
        ax1.set_title("Accuracies of the Algorithms")

