import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

import numpy as np
from keras.models import load_model
model = load_model('traffic_classifier_sign')

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }


  
result=''
root = tk.Tk()
root.geometry('800x600')
root.title('Traffic Sign Classifier')
root.configure(background='cornsilk2')
my_label=Label(root,background='cornsilk2')
label=Label(root,background='cornsilk2')


def classify(filename):
    image = Image.open(filename)
    image = image.resize((30,30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    ans = model.predict_classes([image])[0]
    result = classes[ans+1]
    print(result)
    label.configure(fg='dark olive green',text=result,font=('arial',20,'bold','italic'))
    
    
def classify_button(filename):
    classify_b=Button(root,text='Classify image',command=lambda:classify(filename),padx=10,pady=5)
    classify_b.configure(background='brown4', foreground='dark olive green',font=('arial',10,'bold','italic'))
    classify_b.place(relx=0.79,rely=0.46)
    
def open():
    filename=filedialog.askopenfilename(title='Select image')
    my_img=Image.open(filename)
    my_img = my_img.resize((250, 250), Image.ANTIALIAS) 
    
    my_img=ImageTk.PhotoImage(my_img)
    my_label.configure(image=my_img)
    my_label.image=my_img
    classify_button(filename)
    
        

my_btn=Button(root,text="Upload image",command=open,padx=10,pady=5)
my_btn.configure(background='brown4', foreground='dark olive green',font=('arial',10,'bold','italic'))
my_btn.pack(side=BOTTOM,pady=50)
my_label.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(root, text="Know Your Traffic Sign",pady=20, font=('arial',30,'bold'))
heading.configure(bg='cornsilk2',fg='dark olive green')
heading.pack()
root.mainloop()