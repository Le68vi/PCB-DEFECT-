from tensorflow.keras.applications.resnet50 import ResNet,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

#1..Load the pre-trained model 
model=Resnet50(weights='imagenet')

#2..Load and preprocess an image
img_path='your_name.jpg'
img=image.load_img(img_path,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

#3..Make predictions
preds=model.predict(x)
print('predicted:',decode_predictions(preds,top=3)[0])