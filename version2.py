import tkinter as tk
from classification import *
from tkinter import filedialog
from visualize import *
global train
global test

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
        tk.Label(self,text="Machine Learning Algorithms :").pack(side="top",anchor="w")
        self.mloptions = checkOptions(self,["Naive Bayes Model","Decision Tree Model","Random Forest Model","Bagging Model"])
        self.mloptions.pack(side=tk.TOP,  fill=tk.X,anchor="w")
        self.mloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Label(self,text="Transfer Learning Algorithms :").pack(side="top",anchor="w",pady=(10,0))
        self.tloptions = checkOptions(self,["MobileNet Model","ResNet50","VGG16","VGG19"])
        self.tloptions.pack(side="top",fill="x",anchor="w")
        self.tloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Button(self, text='Execute',command=self.execute).pack(side="bottom")
        tk.Button(self, text='Back',command=lambda: controller.show_frame(Start)).pack(side="bottom")
        self.flag=False
        self.cobj=Classify()
        self.vis=None
        self.names=[]
        self.probas_list=[]
        self.accs=[]
    def allstates(self): 
        ml=list(self.mloptions.state())
        tl=list(self.tloptions.state())
        res=[ml,tl]
        return res

    def executeML(self,ml,item):
        global train
        global test

        for i in range(len(ml)):
            if ml[i]==1:
                if i==0:
                    nbc,ny_pred,ny_probas,nacc,nname,ncategory,x,y,y_test=self.cobj.naive(item)
                    self.vis.plotIndi(nbc,ny_pred,ny_probas,nname,ncategory,x,y,y_test)
                    self.names.append(nname)
                    self.probas_list.append(ny_probas)
                    self.accs.append(nacc)
                elif i==1:
                    dtc,dy_pred,dy_probas,dacc,dname,dcategory,x,y,y_test=self.cobj.decision(item)
                    self.vis.plotIndi(dtc,dy_pred,dy_probas,dname,dcategory,x,y,y_test)
                    self.names.append(dname)
                    self.probas_list.append(dy_probas)
                    self.accs.append(dacc)
                elif i==2:
                    rfc,fy_pred,fy_probas,facc,fname,fcategory,x,y,y_test=self.cobj.forest(item)
                    self.vis.plotIndi(rfc,fy_pred,fy_probas,fname,fcategory,x,y,y_test)
                    self.names.append(fname)
                    self.probas_list.append(fy_probas)
                    self.accs.append(facc)

                else:
                    bc,by_pred,by_probas,bacc,bname,bcategory=self.cobj.bag(item)
                    self.vis.plotIndi(bc,by_pred,by_probas,bname,bcategory,x,y,y_test)
                    self.names.append(bname)
                    self.probas_list.append(by_probas)
                    self.accs.append(bacc)
        return y_test
    def execute(self):
        self.vis=Visualize()
        chosen=self.allstates()
        ml=chosen[0]
        tl=chosen[1]
        global train
        global test

        if 1 in ml:
            item=self.cobj.readimage(train,test)        
            self.executeML(ml,item)
        if 1 in tl:    
            images_dict=self.cobj.readimageTL(train)
            for i in range(len(tl)):
                if tl[i]==1:
                    if i==0:
                        item=self.cobj.mobnet(images_dict,test)
                        y=self.executeML(chosen[0],item)
                    elif i==1:
                        item=self.cobj.resnet(images_dict,test)
                        y=self.executeML(chosen[0],item)
                    elif i==2:
                        item=self.cobj.vgg16(images_dict,test)
                        y=self.executeML(chosen[0],item)
                    else:
                        item=self.cobj.vgg19(images_dict,test)
                        y=self.executeML(chosen[0],item)
        if len(self.accs)>1:
            self.vis.plotComp(self.names,self.accs)

class Clustering(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        tk.Label(self,text="Machine Learning Algorithms :").pack(side="top",anchor="w")
        self.mloptions = checkOptions(self, ["K Means Algorithm"])
        self.mloptions.pack(side=tk.TOP,  fill=tk.X,anchor="w")
        self.mloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Label(self,text="Transfer Learning Algorithms :").pack(side="top",anchor="w",pady=(10,0))
        self.tloptions = checkOptions(self, ["MobileNet Model","ResNet50","VGG16","VGG19"])
        self.tloptions.pack(side="top",fill="x",anchor="w")
        self.tloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Button(self, text='Back',command=lambda: controller.show_frame(Start)).pack(side="bottom")
        tk.Button(self, text='Peek', command=self.allstates).pack(side="bottom")

    def allstates(self): 
        return list(self.mloptions.state(), self.tloptions.state())

    '''def proceed(self,file):
        chosen=self.allstates()
        for i in chosen:
            for j in i:
                if i==
                if j[0]==1:
                    kmeans()
                
        pass'''
  
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
        f=tk.Frame(self)
        f.pack(side="top",expand=True)
        self.flag=False
        tk.Label(f,text="Upload an image dataset :").pack(side="left",anchor="w")
        tk.Button(f,text="Browse ",command=self.openfile).pack(side="right",anchor="e")

        f=tk.LabelFrame(self,text="Select an application --",pady=10,padx=10)
        f.pack(side="top",fill="both",expand=True,anchor="w",padx=10,pady=10)

        Apps=[
            ("Image Similarity",1),
            ("Image Classification",2),
            ("Image Clustering",3)
        ]
        self.choice=tk.IntVar()
        self.choice.set(0)
        for app,value in Apps:
            tk.Radiobutton(f,text=app,variable=self.choice,value=value).pack(side="top",anchor="w")
        self.btn=tk.Button(self,text="Proceed",command=lambda:self.getchoice(controller))
        self.btn.pack(side="top",fill="x",expand=True)
        self.t=tk.Label(self)
        self.t.pack(side="top",fill="x",expand=True)

    def getchoice(self,controller):
        if self.flag==False:
            self.t.config(text="Dataset not uploaded")
            self.flag=False
        else:
            if self.choice.get()==0:
                self.t.config(text="Please select a task")
            elif(self.choice.get()==1):
                self.t.config(text="")
                go=Similarity
            elif(self.choice.get()==2):
                self.t.config(text="")
                go=Classification
            elif(self.choice.get()==3):
                self.t.config(text="")
                go=Clustering
            controller.show_frame(go)
  
    def openfile(self):
        global train
        global test
        train=filedialog.askdirectory(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select a file for training")
        test=filedialog.askopenfilename(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select a file for testing", filetypes=(("All Files", "*.*"),("CSV files",".csv")))
        
        self.t.config(text="File is successfully loaded")
        self.flag=True


class Similarity(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        tk.Label(self,text="Machine Learning Algorithms :").pack(side="top",anchor="w")
        self.mloptions = checkOptions(self, ["Cosine Similarity","Euclidean Distance","Manhattan Distance"])
        self.mloptions.pack(side=tk.TOP,  fill=tk.X,anchor="w")
        self.mloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Label(self,text="Transfer Learning Algorithms :").pack(side="top",anchor="w",pady=(10,0))
        self.tloptions = checkOptions(self, ["MobileNet Model","ResNet50","VGG16","VGG19"])
        self.tloptions.pack(side="top",fill="x",anchor="w")
        self.tloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Button(self, text='Back',command=lambda: controller.show_frame(Start)).pack(side="bottom")
        tk.Button(self, text='Peek', command=self.allstates).pack(side="bottom")


    def allstates(self): 
        print(list(self.mloptions.state()), list(self.tloptions.state()))

    

app=Container()
app.mainloop()



   
