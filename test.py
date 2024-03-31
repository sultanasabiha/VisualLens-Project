import tkinter as tk
from classification import *
from tkinter import filedialog

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
        self.tloptions = checkOptions(self, ["MobileNet Model","ResNet50","VGG16","VGG19"])
        self.tloptions.pack(side="top",fill="x",anchor="w")
        self.tloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Button(self, text='Back',command=lambda: controller.show_frame(Start)).pack(side="bottom")
        tk.Button(self, text='Peek', command=self.allstates).pack(side="bottom")
        self.exe=tk.Button(self, text='Execute',command=self.execute).pack(side="bottom")


    def allstates(self): 
        ml=list(self.mloptions.state())
        tl=list(self.tloptions.state())
        res=[ml,tl]
        return res

    def executeML(self,chosen,item):
        for i in range(len(chosen)):
            if chosen[i]==1:
                if i==0:
                    naive(item)
                elif i==1:
                    decision(item)
                elif i==2:
                    forest(item)
                else:
                    bag(item)
        
    def execute(self):
        global train
        global test
        chosen=self.allstates()
        item=readimage(train,test)
        self.executeML(chosen[0],item)

        tl=chosen[1]
        if 1 in tl:
            images_dict=readimageTL(train)
            for i in range(len(tl)):
                if tl[i]==1:
                    
                    if i==0:
                        item=mobnet(images_dict,test)
                        self.executeML(chosen[0],item)
                    elif i==1:
                        item=resnet(images_dict,test)
                        self.executeML(chosen[0],item)
                    elif i==2:
                        item=vgg16(images_dict,test)
                        self.executeML(chosen[0],item)
                    else:
                        item=vgg19(images_dict,test)
                        self.executeML(chosen[0],item)

        

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



   
