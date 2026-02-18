
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator




#import keras
#Robust Data Augmentation Pipeline
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixels
    rotation_range=15,           # Slight PCB rotation
    width_shift_range=0.1,       # Horizontal shift
    height_shift_range=0.1,      # Vertical shift
    zoom_range=0.15,             # Camera zoom variation
    shear_range=0.1,             # Perspective distortion
    horizontal_flip=True,        # If PCB orientation allows
    vertical_flip=False,         # Usually false for PCBs
    brightness_range=[0.7,1.3],  # Lighting variation
    fill_mode='nearest'
)
#For validation data (no augmentation, only normalization):
val_datagen = ImageDataGenerator(rescale=1./255)

#Load Images Using Flow From Directory
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


#Visualize Augmented Samples
import matplotlib.pyplot as plt

x_batch, y_batch = next(train_generator)

plt.figure(figsize=(10,6))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_batch[i])
    plt.axis('off')
plt.show()

