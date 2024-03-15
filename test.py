from tkinter import *
from tkinter import filedialog
root=Tk()
root.title("Test Version")
root.geometry("400x500")

    
def proceed(val):

    if(val==1):
        frame1=LabelFrame(root,padx=10,pady=10,height=250,width=250)
        frame1.grid(row=8,column=0,sticky=W+E,padx=10,pady=10)
        frame1.grid_propagate(0)
        Label(frame1,text="Machine Learning Algorithms :").grid(row=9,column=0)
        algos_trad=[
            ("Cosine Similarity",1),
            ("Euclidean Distance",2),
            ("Manhattan Distance",3)
        ]
        algos=IntVar()
        algos.set(1)
        row=10
        for algo,value in algos_trad:
            Radiobutton(frame1,text=algo,variable=algos,value=value).grid(row=row,column=0,sticky=W)
            row+=1
        trad=algos.get()
        
        Label(frame1,text="Transfer Learning Algorithms :").grid(row=13,column=0)
        algos_trans=[
            ("MobileNet Model",4),
            ("ResNet50",5),
            ("VGG16",6),
            ("VGG19",7)
        ]
        algost=IntVar()
        algost.set(1)
        row=14
        for algo,value in algos_trans:
            Radiobutton(frame1,text=algo,variable=algost,value=value).grid(row=row,column=0,sticky=W)
            row+=1
        trans=algost.get()
        

    if(val==2):

        frame2=LabelFrame(root,padx=10,pady=10,height=250,width=250)
        frame2.grid(row=8,column=0,sticky=W+E,padx=10,pady=10)
        frame2.grid_propagate(0)
        Label(frame2,text="Machine Learning Algorithms :").grid(row=9,column=0)
        algos_trad=[
            ("Naive Bayes Model",1),
            ("Decision Tree Model",2),
            ("Random Forest Model",3),
            ("Bagging Model",4)
        ]
        algos=IntVar()
        algos.set(1)
        row=10
        for algo,value in algos_trad:
            Radiobutton(frame2,text=algo,variable=algos,value=value).grid(row=row,column=0,sticky=W)
            row+=1
        trad=algos.get()
        
        Label(frame2,text="Transfer Learning Algorithms :").grid(row=14,column=0)
        algos_trans=[
        ("MobileNet Model",4),
        ("ResNet50",5),
        ("VGG16",6),
        ("VGG19",7)
        ]
        algost=IntVar()
        algost.set(1)
        row=15
        for algo,value in algos_trans:
            Radiobutton(frame2,text=algo,variable=algost,value=value).grid(row=row,column=0,sticky=W)
            row+=1
        trans=algost.get()
    if(val==3):

        frame3=LabelFrame(root,padx=10,pady=10,height=250,width=250)
        frame3.grid(row=8,column=0,sticky=W+E,padx=10,pady=10)
        frame3.grid_propagate(0)
        Label(frame3,text="Machine Learning Algorithms :").grid(row=9,column=0,sticky=W)
        algos_trad=[
            ("K Means Algorithm",1)
        ]
        algos=IntVar()
        algos.set(1)
        row=10
        for algo,value in algos_trad:
            Radiobutton(frame3,text=algo,variable=algos,value=value).grid(row=row,column=0,sticky=W)
            row+=1
        trad=algos.get()
        
        Label(frame3,text="Transfer Learning Algorithms :").grid(row=11,column=0)
        algos_trans=[
        ("MobileNet Model",4),
        ("ResNet50",5),
        ("VGG16",6),
        ("VGG19",7)
        ]
        algost=IntVar()
        algost.set(1)
        row=12
        for algo,value in algos_trans:
            Radiobutton(frame3,text=algo,variable=algost,value=value).grid(row=row,column=0,sticky=W)
            row+=1
        trans=algost.get()
def open():
    root.filename=filedialog.askopenfilename(initialdir="c:\\Users\\Sabiha\\Downloads",title="Select a file", filetypes=(("All Files", "*.*"),("CSV Files", "*.csv")))
    Label(root,text="File is successfully loaded").grid(row=1,column=0,sticky=W)
    
    Label(root,text="Select an application --",pady=10).grid(row=2,column=0,sticky=W)
    Apps=[
    ("Image Similarity",1),
    ("Supervised Image Classification",2),
    ("Unsupervised Image Clustering",3)
    ]
    choice=IntVar()
    choice.set(0)
    row=3
    for app,value in Apps:
        Radiobutton(root,text=app,variable=choice,value=value).grid(row=row,column=0,sticky=W)
        row+=1
    btn=Button(root,text="Proceed",command=lambda: proceed(choice.get())).grid(row=7,column=0)
 
open_label=Label(root,text="Upload an image dataset :").grid(row=0,column=0,sticky=W)
open_button=Button(root,text="Browse ",command=open).grid(row=0,column=1,sticky=W)


root.mainloop()