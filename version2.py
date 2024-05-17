import tkinter as tk
from classification import *
from tkinter import filedialog
from visualize import *
from tkinter import messagebox
from similarity import *
from clustering import *

class checkOptions(tk.Frame):
    def __init__(self, parent=None, picks=[], anchor=tk.W):
        tk.Frame.__init__(self, parent)
        self.vars = []
        for pick in picks:
            var = tk.IntVar()
            chk = tk.Checkbutton(self, text=pick, variable=var)
            chk.pack(side="top",expand=True,anchor="w")
            self.vars.append(var)
    def state(self):
        return map((lambda var: var.get()), self.vars)

class Classification(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
       
        f=tk.LabelFrame(self,text="Select classification type --",pady=10,padx=10)
        f.pack(side="top",fill="both",expand=True,anchor="w",padx=10,pady=10)
        types=[
            ("Binary Classification",1),
            ("Multiclass Classification",2)
        ]

        self.choice=tk.IntVar()
        self.choice.set(0)
        for type,value in types:
            tk.Radiobutton(f,text=type,variable=self.choice,value=value).pack(side="top",anchor="w")
        f=tk.Frame(self)
        f.pack(side="top",expand=True,fill='x',pady=10)
        tk.Label(f,text="Upload an image dataset :").pack(side="left",anchor="w",fill='x')
        tk.Button(f,text="Browse ",command=self.openfile).pack(side="right",anchor="e",fill='x',expand=True)

        tk.Label(self,text="Machine Learning Algorithms :").pack(side="top",anchor="w")
        self.mloptions = checkOptions(self,["Naive Bayes Model","Decision Tree Model","Random Forest Model","Bagging Model"])
        self.mloptions.pack(side=tk.TOP,  fill=tk.X,anchor="w")
        self.mloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Label(self,text="Transfer Learning Algorithms :").pack(side="top",anchor="w",pady=(10,0))
        self.tloptions = checkOptions(self,["MobileNet Model","ResNet50","VGG16","VGG19"])
        self.tloptions.pack(side="top",fill="x",anchor="w")
        self.tloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Button(self,text='Quit',command=self.quit).pack(side="bottom",fill='x',pady=5) 
        tk.Button(self, text='Back',command=lambda: controller.show_frame(Start)).pack(side="bottom",fill='x',pady=5)
        self.exe=tk.Button(self, text='Execute',command=self.execute,state="disabled")
        self.exe.pack(side="bottom",fill='x',pady=5)
        self.count=0
        self.cobj=None
        self.vis=None
        self.t=tk.Label(self)
        self.t.pack(side="top",fill="x",expand=True)

        self.category_file,self.image_dir,self.test="","",""

    def allstates(self): 
        ml=list(self.mloptions.state())
        tl=list(self.tloptions.state())
        res=[ml,tl]
        return res
    
    def openfile(self):
        if self.choice.get()==0:
            messagebox.showinfo("Warning","Please select the type of classification")
        else:
            self.image_dir=filedialog.askdirectory(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select folder containing images: ")
            self.category_file=filedialog.askopenfilename(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select file containing labels", filetypes=(("All Files", "*.*"),("Text Files",".txt"),("Word Files",".doc")))
            self.test=filedialog.askopenfilename(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select a file for testing", filetypes=(("All Files", "*.*"),("JPG",".jpg"),("PNG",".png")))
    
            if self.category_file =="" or self.image_dir =="" or self.test =="":
                messagebox.showinfo("Warning","Dataset not uploaded")
            else:
                self.t.config(text="File is successfully loaded")
                self.exe.config(state='normal')

    def executeML(self,ml,item):
        for i in range(len(ml)):
            if ml[i]==1:
                if i==0:
                    nbc,ny_pred,ny_probas,nname,ncategory,x,y,y_test=self.cobj.naive(item)
                    self.vis.plot_results(nbc,ny_pred,ny_probas,nname,ncategory,x,y,y_test)
                    self.count+=1
                    print(nbc)
                elif i==1:
                    dtc,dy_pred,dy_probas,dname,dcategory,x,y,y_test=self.cobj.decision(item)
                    self.vis.plot_results(dtc,dy_pred,dy_probas,dname,dcategory,x,y,y_test)
                    self.count+=1
                elif i==2:
                    rfc,fy_pred,fy_probas,fname,fcategory,x,y,y_test=self.cobj.forest(item)
                    self.vis.plot_results(rfc,fy_pred,fy_probas,fname,fcategory,x,y,y_test)
                    self.count+=1
                else:
                    bc,by_pred,by_probas,bname,bcategory,x,y,y_test=self.cobj.bag(item)
                    self.vis.plot_results(bc,by_pred,by_probas,bname,bcategory,x,y,y_test)
                    self.count+=1
    def execute(self):
        self.cobj=Classify(self.category_file,self.image_dir,self.test)
        chosen=self.allstates()
        ml=chosen[0]
        tl=chosen[1]

        if 1 not in ml and 1 not in tl:
            messagebox.showinfo("Warning","Please select an algorithm")
        elif 1 not in ml and 1 in tl:
            messagebox.showinfo("Warning","Please select an ML algorithm")
        else:
            self.vis=VisualizeClass(self.choice.get())
            if 1 in ml:
                item=self.cobj.readimage()        
                self.executeML(ml,item)
            if 1 in tl:    
                for i in range(len(tl)):
                    if tl[i]==1:
                        if i==0:
                            item=self.cobj.mobnet()
                            self.executeML(chosen[0],item)
                            self.count+=1
                        elif i==1:
                            item=self.cobj.resnet()
                            self.executeML(chosen[0],item)
                            self.count+=1
                        elif i==2:
                            item=self.cobj.vgg16()
                            self.executeML(chosen[0],item)
                            self.count+=1
                        else:
                            item=self.cobj.vgg19()
                            self.executeML(chosen[0],item)
                            self.count+=1
            if self.count>1:
                self.vis.plot_comparison()

class Clustering(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        f=tk.Frame(self)
        f.pack(side="top",fill='both',pady=5)
        tk.Label(f,text="Enter the number of clusters : ").pack(side="left",anchor="w",fill='x')
        self.text=tk.Entry(f,width=10)
        self.text.pack(side="right",anchor="e",fill='x',expand=True)
        
        f=tk.Frame(self)
        f.pack(side="top",fill='x',pady=10)
        tk.Label(f,text="Upload an image dataset :").pack(side="left",anchor="w",fill='x')
        tk.Button(f,text="Browse ",command=self.openfile).pack(side="right",anchor="e",fill='x',expand=True)
       
        tk.Label(self,text="Machine Learning Algorithms :").pack(side="top",anchor="w")
        self.mloptions = checkOptions(self, ["K Means Algorithm","Agglomerative Hierarchical Algorithm","Spectral Clustering"])
        self.mloptions.pack(side=tk.TOP,anchor="w",expand=True,fill='both')
        self.mloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Label(self,text="Transfer Learning Algorithms :").pack(side="top",anchor="w",pady=(10,0))
        self.tloptions = checkOptions(self, ["MobileNet Model","ResNet50","VGG16","VGG19"])
        self.tloptions.pack(side="top",anchor="w",expand=True,fill='both')
        self.tloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Button(self,text='Quit',command=self.quit).pack(side="bottom",fill='x',pady=5) 
        tk.Button(self, text='Back',command=lambda: controller.show_frame(Start)).pack(side="bottom",pady=5,fill='x')
        self.exe=tk.Button(self, text='Execute',command=self.execute,state="disabled")
        self.exe.pack(side="bottom",fill='x',pady=5)
        self.count=0
        self.cobj=None
        self.vis=None
        self.t=tk.Label(self)
        self.t.pack(side="top",fill="x",expand=True)

        self.image_dir=""

    def allstates(self): 
        ml=list(self.mloptions.state())
        tl=list(self.tloptions.state())
        res=[ml,tl]
        return res
    
    def openfile(self):
        if self.text.get()=='':
            messagebox.showinfo("Warning","Please enter no. of clusters")
        else:
            self.n_clusters=int(self.text.get())  
            self.image_dir=filedialog.askdirectory(initialdir="c:/Users/Sabiha/Desktop/Project",title="Select folder containing images: ")
            if self.image_dir =="":
                messagebox.showinfo("Warning","Dataset not uploaded")
            else:
                self.t.config(text="File is successfully loaded")
                self.exe.config(state='normal')

    def executeML(self,ml,mat):
        for i in range(len(ml)):
            if ml[i]==1:
                if i==0:
                    results_df,name,y_means=self.cobj.kmeans(mat)
                    self.vis.plot_results(results_df,name,mat,y_means)
                    self.count+=1
                elif i==1:
                    results_df,name,y_agglo=self.cobj.agglo(mat)
                    self.vis.plot_results(results_df,name,mat,y_agglo)
                    self.vis.plot_dendo(mat)
                    self.count+=1
                else:
                    results_df,name,y_spec=self.cobj.spec(mat)
                    self.vis.plot_results(results_df,name,mat,y_spec)
                    self.count+=1

    def execute(self):
        self.cobj=Cluster(self.image_dir,self.n_clusters)
        chosen=self.allstates()
        ml=chosen[0]
        tl=chosen[1]

        if 1 not in ml and 1 not in tl:
            messagebox.showinfo("Warning","Please select an algorithm")
        elif 1 not in ml and 1 in tl:
            messagebox.showinfo("Warning","Please select an ML algorithm")
        else:
            self.vis=VisualizeClus(self.n_clusters)
            if 1 in ml:
                matrix=self.cobj.getMatrix()        
                self.executeML(ml,matrix)
            if 1 in tl:    
                for i in range(len(tl)):
                    if tl[i]==1:
                        if i==0:
                            mat=self.cobj.mobnet()
                            self.executeML(chosen[0],mat)
                            self.count+=1
                        elif i==1:
                            mat=self.cobj.resnet()
                            self.executeML(chosen[0],mat)
                            self.count+=1
                        elif i==2:
                            mat=self.cobj.vgg16()
                            self.executeML(chosen[0],mat)
                            self.count+=1
                        else:
                            mat=self.cobj.vgg19()
                            self.executeML(chosen[0],mat)
                            self.count+=1
            if self.count>1:
                self.vis.plot_comparison()
  
class Container(tk.Tk):
    def __init__(self,*args,**kwargs):
        tk.Tk.__init__(self,*args,**kwargs)
        root=tk.Frame(self)
        root.pack(side="top", fill="both", expand = True,padx=10,pady=10,ipady=5,ipadx=5)
        root.grid_columnconfigure(0, weight=4)
        root.grid_columnconfigure(1, weight=1)

        self.frames={}
        for F in (Start,Similarity,Classification,Clustering):
            frame=F(root,self)
            self.frames[F]=frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(Start)
    def show_frame(self,con):
        frame=self.frames[con]
        frame.tkraise()

class Start(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        f=tk.LabelFrame(self,text=" Select an application ",pady=10,padx=10)
        f.pack(side="top",fill="both",expand=True,anchor="w",padx=10,pady=10)

        Apps=[
            ("Image Similarity",1),
            ("Image Classification",2),
            ("Image Clustering",3)
        ]
        self.choice=tk.IntVar()
        self.choice.set(0)
        for app,value in Apps:
            tk.Radiobutton(f,text=app,variable=self.choice,value=value).pack(side="top",anchor="w",ipady=10)
        self.btn=tk.Button(self,text="Proceed",command=lambda:self.getchoice(controller))
        self.btn.pack(side="top",fill="x",pady=5)
        tk.Button(self,text='Quit',command=self.quit).pack(side="bottom",fill='x',pady=5)

    def getchoice(self,controller):
        if self.choice.get()==0:
            messagebox.showinfo("Warning","Please select a task")
        elif(self.choice.get()==1):
            go=Similarity
        elif(self.choice.get()==2):
            go=Classification
        elif(self.choice.get()==3):
            go=Clustering
        controller.show_frame(go)
  

class Similarity(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        f=tk.Frame(self)
        f.pack(side="top",fill='x',pady=10)
        tk.Label(f,text="Upload an image dataset :").pack(side="left",anchor="w",fill='x')
        tk.Button(f,text="Browse ",command=self.openfile).pack(side="right",anchor="e",fill='x',expand=True)
        tk.Label(self,text="Machine Learning Algorithms :").pack(side="top",anchor="w")
        self.mloptions = checkOptions(self, ["Cosine Similarity","Euclidean Distance","Manhattan Distance"])
        self.mloptions.pack(side=tk.TOP,  fill='both',anchor="w",expand=True)
        self.mloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Label(self,text="Transfer Learning Algorithms :").pack(side="top",anchor="w",pady=(10,0))
        self.tloptions = checkOptions(self, ["MobileNet Model","ResNet50","VGG16","VGG19"])
        self.tloptions.pack(side="top",fill="both",expand=True,anchor="w")
        self.tloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Button(self,text='Quit',command=self.quit).pack(side="bottom",fill='x',pady=5)
        tk.Button(self, text='Back',command=lambda: controller.show_frame(Start)).pack(side="bottom",fill='x',pady=5)
        self.exe=tk.Button(self, text='Execute',command=self.execute,state="disabled")
        self.exe.pack(side="bottom",fill='x',pady=5)

        self.count=0
        self.sobj=None
        self.vis=None
        self.t=tk.Label(self)
        self.t.pack(side="top",fill="x",expand=True)

        self.image_dir,self.test="",""

    def allstates(self): 
        ml=list(self.mloptions.state())
        tl=list(self.tloptions.state())
        res=[ml,tl]
        return res
    
    def openfile(self):

        self.image_dir=filedialog.askdirectory(initialdir="c:/Users/Sabiha/Desktop/Project",title="Select folder containing images: ")
        self.test=filedialog.askopenfilename(initialdir="c:/Users/Sabiha/Desktop/Project",title="Select a file for testing", filetypes=(("All Files", "*.*"),("JPG",".jpg"),("PNG",".png")))
  
        if self.image_dir =="" or self.test =="":
            messagebox.showinfo("Warning","Dataset not uploaded")
        else:
            self.t.config(text="File is successfully loaded")
            self.exe.config(state='normal')

    def executeML(self,ml,mat):
        for i in range(len(ml)):
            if ml[i]==1:
                if i==0:
                    indexes,name=self.sobj.cosine(mat)
                    images=self.sobj.getImages(indexes)
                    self.vis.plot_results(images,name)
                    self.count+=1
                elif i==1:
                    indexes,name=self.sobj.euclidean(mat)
                    images=self.sobj.getImages(indexes)
                    self.vis.plot_results(images,name)
                    self.count+=1
                else:
                    indexes,name=self.sobj.manhattan(mat)
                    images=self.sobj.getImages(indexes)
                    self.vis.plot_results(images,name)
                    self.count+=1

    def execute(self):
        self.sobj=Similar(self.image_dir,self.test)
        chosen=self.allstates()
        ml=chosen[0]
        tl=chosen[1]

        if 1 not in ml and 1 not in tl:
            messagebox.showinfo("Warning","Please select an algorithm")
        elif 1 not in ml and 1 in tl:
            messagebox.showinfo("Warning","Please select an ML algorithm")
        else:
            self.vis=VisualizeSim(self.test)
            if 1 in ml:
                matrix=self.sobj.getMatrix()        
                self.executeML(ml,matrix)
            if 1 in tl:    
                for i in range(len(tl)):
                    if tl[i]==1:
                        if i==0:
                            mat=self.sobj.mobnet()
                            self.executeML(chosen[0],mat)
                            self.count+=1
                        elif i==1:
                            mat=self.sobj.resnet()
                            self.executeML(chosen[0],mat)
                            self.count+=1
                        elif i==2:
                            mat=self.sobj.vgg16()
                            self.executeML(chosen[0],mat)
                            self.count+=1
                        else:
                            mat=self.sobj.vgg19()
                            self.executeML(chosen[0],mat)
                            self.count+=1
            if self.count>1:
                self.vis.plot_comparison()


app=Container()
app.mainloop()



   
