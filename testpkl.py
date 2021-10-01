import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
new_model = load_model(r"E:\Machine_learning\OD exe\new data\new_model.h5")

new_model.summary()


dir_path=r'E:\Machine_learning\OD exe\testing'

for i in os.listdir(dir_path):
    img=image.load_img(dir_path+'//'+i,target_size=(200,200))
    plt.imshow(img)
    plt.show()
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    images=np.vstack([X])
    val=new_model.predict(images)
    if val==0:
        print("ng")
    else:
        print("ok")

