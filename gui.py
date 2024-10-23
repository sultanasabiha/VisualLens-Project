import tkinter as tk
from classification import *
from tkinter import filedialog
from visualize import *
from tkinter import messagebox
from similarity import *
from clustering import *
import pickle
import sys

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


        f=tk.Frame(self)
        f.pack(side="top",expand=True,fill='x',pady=10)
        tk.Button(f,text="Upload Personalized Algorithm",command=self.load_pickle).pack(side="left",anchor="w",fill='x',expand=True)
        tk.Button(f,text=" ? ",command=self.show_info).pack(side="right",anchor="e",expand=True,fill='x')
        self.p_frame=tk.Frame(self)
        self.p_frame.pack(side='top',fill="x",expand=True)

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
        self.flag=False
        self.Name,self.Images_list=[],[]
        self.options=[]
    def show_info(self):
        items = ['Name of the algorithm as Name','List containing 4 most similar images as Images_list']
        formatted_list = '\n\n'.join(f'• {item}' for item in items)
        messagebox.showinfo("Uploading Personalized Algorithms","Please upload the algorithm by pickling the following data with the same name and order as mentioned into a tuple:\n\n"+formatted_list)
    def load_pickle(self):
        self.flag=True
        pickle_file=filedialog.askopenfilename(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select the file to be deserialized --", filetypes=(("All Files", "*.*"),("Python Files",".py"),("Pickle Files",".pkl")))
        if pickle_file =="":
            messagebox.showwarning("Warning","Pickle file not uploaded")
        else:
            # Deserialize the tuple from the file
            with open(pickle_file, 'rb') as f:
                Name,Images_list= pickle.load(f)
            self.Name.append(Name)
            self.Images_list.append(Images_list)
        
            self.add_option(Name,len(self.Name))
    def add_option(self,option_text,index):
        # Create a frame to hold the option and its button
        option_frame = tk.Frame(self.p_frame)
        option_frame.pack(fill='x', pady=5)
        
        # Create the label for the option
        checked_state = tk.BooleanVar()
        checked_state.set(True)
        c=tk.Checkbutton(option_frame, text=option_text, variable=checked_state,command=lambda:checked_state.set(True))
        c.pack(side='left',anchor='w',pady=5)
        
        # Create the button to remove the option
        remove_button = tk.Button(option_frame, text="Remove", command=lambda: self.remove_option(option_frame,index))
        remove_button.pack(side='right', padx=5)

        self.options.append((option_frame, index))

    def remove_option(self,option_frame,index):
        option_frame.destroy()
        self.options = [opt for opt in self.options if opt[1] != index]
        del self.Name[index-1]
        del self.Images_list[index-1]

    def allstates(self): 
        ml=list(self.mloptions.state())
        tl=list(self.tloptions.state())
        res=[ml,tl]
        return res
    
    def openfile(self):

        self.image_dir=filedialog.askdirectory(initialdir="c:/Users/Sabiha/Desktop/Project",title="Select folder containing images: ")
        self.test=filedialog.askopenfilename(initialdir="c:/Users/Sabiha/Desktop/Project",title="Select a file for testing", filetypes=(("All Files", "*.*"),("JPG",".jpg"),("PNG",".png")))
  
        if self.image_dir =="" or self.test =="":
            messagebox.showwarning("Warning","Dataset not uploaded")
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

        if 1 not in ml and 1 not in tl and self.flag==False:
            messagebox.showwarning("Warning","Please select an algorithm")
        elif 1 not in ml and 1 in tl:
            messagebox.showwarning("Warning","Please select an ML algorithm")
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
            if self.flag==True:
                for i in range(len(self.Name)):
                    self.vis.plot_results(self.Images_list[i],self.Name[i])
                    self.count+=1
            if self.count>1:
                self.vis.plot_comparison()

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
        self.mloptions = checkOptions(self,["Naive Bayes Model","Decision Tree Model","Random Forest Model"])
        self.mloptions.pack(side=tk.TOP,  fill=tk.X,anchor="w")
        self.mloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        tk.Label(self,text="Transfer Learning Algorithms :").pack(side="top",anchor="w",pady=(10,0))
        self.tloptions = checkOptions(self,["MobileNet Model","ResNet50","VGG16","VGG19"])
        self.tloptions.pack(side="top",fill="x",anchor="w")
        self.tloptions.config(relief=tk.GROOVE, bd=2,padx=10,pady=10)

        f=tk.Frame(self)
        f.pack(side="top",expand=True,fill='x',pady=10)
        tk.Button(f,text="Upload Personalized Algorithm",command=self.load_pickle).pack(side="left",anchor="w",fill='x',expand=True)
        tk.Button(f,text=" ? ",command=self.show_info).pack(side="right",anchor="e",expand=True,fill='x')
        self.p_frame=tk.Frame(self)
        self.p_frame.pack(side='top',fill="x",expand=True)

        tk.Button(self,text='Quit',command=self.quit).pack(side="bottom",fill='x',pady=5) 
        tk.Button(self, text='Back',command=lambda: controller.show_frame(Start)).pack(side="bottom",fill='x',pady=5)
        self.exe=tk.Button(self, text='Execute',command=self.execute,state="disabled")
        self.exe.pack(side="bottom",fill='x',pady=5)
        self.count=0
        self.cobj=None
        self.vis=None
        self.t=tk.Label(self)
        self.t.pack(side="top",fill="x",expand=True)
        self.flag=False
        self.category_file,self.image_dir,self.test="","",""
        self.Name,self.Classifier, self.x_train,self.y_train,self.y_test,self.y_pred,self.y_probas,self.category_timg=[],[],[],[],[],[],[],[]
        self.options=[]
    def show_info(self):
        items = ['Name of the algorithm as Name','Classifier object as Classifier','Predicted y-values as y_pred', 'Probabality of y-values as y_probas','Training set of x-values as x_train', 'Training set of y-values as y_train', 'Testing set of y-values as y_test','Category of test image as category_timg']
        formatted_list = '\n\n'.join(f'• {item}' for item in items)
        messagebox.showinfo("Uploading Personalized Algorithms","Please upload the algorithm by pickling the following data with the same name and order as mentioned into a tuple:\n\n"+formatted_list)
    def load_pickle(self):
        self.flag=True
        pickle_file=filedialog.askopenfilename(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select the file to be deserialized --", filetypes=(("All Files", "*.*"),("Python Files",".py"),("Pickle Files",".pkl")))
        if pickle_file =="":
            messagebox.showwarning("Warning","Pickle file not uploaded")
        else:
        # Deserialize the tuple from the file
            with open(pickle_file, 'rb') as f:
                Name,Classifier,y_pred,y_probas,x_train,y_train,y_test,category_timg= pickle.load(f)
            self.Name.append(Name)
            self.Classifier.append(Classifier)
            self.x_train.append(x_train)
            self.y_train.append(y_train)
            self.y_test.append(y_test)
            self.y_pred.append(y_pred)
            self.y_probas.append(y_probas)
            self.category_timg.append(category_timg)

            self.add_option(Name,len(self.Name))
    def add_option(self,option_text,index):
        # Create a frame to hold the option and its button
        option_frame = tk.Frame(self.p_frame)
        option_frame.pack(fill='x', pady=5)
        
        # Create the label for the option
        checked_state = tk.BooleanVar()
        checked_state.set(True)
        c=tk.Checkbutton(option_frame, text=option_text, variable=checked_state,command=lambda:checked_state.set(True))
        c.pack(side='left',anchor='w',pady=5)
        
        # Create the button to remove the option
        remove_button = tk.Button(option_frame, text="Remove", command=lambda: self.remove_option(option_frame,index))
        remove_button.pack(side='right', padx=5)

        self.options.append((option_frame, index))

    def remove_option(self,option_frame,index):
        option_frame.destroy()
        self.options = [opt for opt in self.options if opt[1] != index]
        del self.Name[index-1]
        del self.Classifier[index-1]
        del self.x_train[index-1]
        del self.y_pred[index-1]
        del self.y_train[index-1]
        del self.y_probas[index-1]
        del self.y_test[index-1]
        del self.category_timg[index-1]


    def allstates(self): 
        ml=list(self.mloptions.state())
        tl=list(self.tloptions.state())
        res=[ml,tl]
        return res
    
    def openfile(self):
        if self.choice.get()==0:
            messagebox.showwarning("Warning","Please select the type of classification")
        else:
            self.image_dir=filedialog.askdirectory(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select folder containing images: ")
            self.category_file=filedialog.askopenfilename(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select file containing labels", filetypes=(("All Files", "*.*"),("Text Files",".txt"),("Word Files",".doc")))
            self.test=filedialog.askopenfilename(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select a file for testing", filetypes=(("All Files", "*.*"),("JPG",".jpg"),("PNG",".png")))
    
            if self.category_file =="" or self.image_dir =="" or self.test =="":
                messagebox.showwarning("Warning","Dataset not uploaded")
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

    def execute(self):
        self.cobj=Classify(self.category_file,self.image_dir,self.test)
        chosen=self.allstates()
        ml=chosen[0]
        tl=chosen[1]

        if 1 not in ml and 1 not in tl and self.flag==False:
            messagebox.showwarning("Warning","Please select an algorithm")
        elif 1 not in ml and 1 in tl:
            messagebox.showwarning("Warning","Please select an ML algorithm")
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
            if self.flag==True:
                for i in range(len(self.Name)):
                    self.vis.plot_results(self.Classifier[i],self.y_pred[i],self.y_probas[i],self.Name[i],self.category_timg[i],self.x_train[i],self.y_train[i],self.y_test[i])
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

        f=tk.Frame(self)
        f.pack(side="top",expand=True,fill='x',pady=10)
        tk.Button(f,text="Upload Personalized Algorithm",command=self.load_pickle).pack(side="left",anchor="w",fill='x',expand=True)
        tk.Button(f,text=" ? ",command=self.show_info).pack(side="right",anchor="e",expand=True,fill='x')
        self.p_frame=tk.Frame(self)
        self.p_frame.pack(side='top',fill="x",expand=True)

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
        self.flag=False
        self.Name,self.results_df,self.Images_matrix,self.y_result=[],[],[],[]
        self.options=[]

    def show_info(self):
        items = ['Name of the algorithm as Name','Pandas dataframe of clustering result containing a Path column(path to the image) and a Predicted column(predicted class of that image) as results_df','Matrix containing all the loaded images as Image_matrix','Resultant matrix as y_result']
        formatted_list = '\n\n'.join(f'• {item}' for item in items)
        messagebox.showinfo("Uploading Personalized Algorithms","Please upload the algorithm by pickling the following data with the same name and order as mentioned into a tuple:\n\n"+formatted_list)
    def load_pickle(self):
        self.flag=True
        pickle_file=filedialog.askopenfilename(initialdir="c:\\Users\\Sabiha\\Desktop\\Project",title="Select the file to be deserialized --", filetypes=(("All Files", "*.*"),("Python Files",".py"),("Pickle Files",".pkl")))
        if pickle_file =="":
            messagebox.showwarning("Warning","Pickle file not uploaded")
        else:
            # Deserialize the tuple from the file
            with open(pickle_file, 'rb') as f:
                Name,results_df,Images_matrix,y_result= pickle.load(f)
            self.Name.append(Name)
            self.results_df.append(results_df)
            self.Images_matrix.append(Images_matrix)
            self.y_result.append(y_result)

            self.add_option(Name,len(self.Name))
    def add_option(self,option_text,index):
        # Create a frame to hold the option and its button
        option_frame = tk.Frame(self.p_frame)
        option_frame.pack(fill='x', pady=5)
        
        # Create the label for the option
        checked_state = tk.BooleanVar()
        checked_state.set(True)
        c=tk.Checkbutton(option_frame, text=option_text, variable=checked_state,command=lambda:checked_state.set(True))
        c.pack(side='left',anchor='w',pady=5)
        
        # Create the button to remove the option
        remove_button = tk.Button(option_frame, text="Remove", command=lambda: self.remove_option(option_frame,index))
        remove_button.pack(side='right', padx=5)

        self.options.append((option_frame, index))

    def remove_option(self,option_frame,index):
        option_frame.destroy()
        self.options = [opt for opt in self.options if opt[1] != index]
        del self.Name[index-1]
        del self.results_df[index-1]
        del self.y_result[index-1]

    def allstates(self): 
        ml=list(self.mloptions.state())
        tl=list(self.tloptions.state())
        res=[ml,tl]
        return res
    
    def openfile(self):
        if self.text.get()=='':
            messagebox.showwarning("Warning","Please enter no. of clusters")
        else:
            self.n_clusters=int(self.text.get())  
            self.image_dir=filedialog.askdirectory(initialdir="c:/Users/Sabiha/Desktop/Project",title="Select folder containing images: ")
            if self.image_dir =="":
                messagebox.showwarning("Warning","Dataset not uploaded")
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

        if 1 not in ml and 1 not in tl and self.flag==False:
            messagebox.showwarning("Warning","Please select an algorithm")
        elif 1 not in ml and 1 in tl:
            messagebox.showwarning("Warning","Please select an ML algorithm")
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
            if self.flag==True:
                for i in range(len(self.Name)):
                    self.vis.plot_results(self.results_df[i],self.Name[i],self.Images_matrix[i],self.y_result[i])
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
            messagebox.showwarning("Warning","Please select a task")
        elif(self.choice.get()==1):
            go=Similarity
        elif(self.choice.get()==2):
            go=Classification
        elif(self.choice.get()==3):
            go=Clustering
        controller.show_frame(go)
  
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

app=Container()
app.title("Visual-Lens")
iconPath = resource_path('visual_lens.ico')
app.iconbitmap(iconPath)

app.mainloop()



   
