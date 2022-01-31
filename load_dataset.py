import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from tensorflow.keras import optimizers



#The dimension of our image will be 300 by 300 pixel
IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)


train_files = glob.glob('training_data/*')
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('/')[-1].split('.')[0].strip() for fn in train_files]


validation_files = glob.glob('validation_data/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('/')[-1].split('.')[0].strip() for fn in validation_files]
#Each image is now of size 300 x 300 and has three channels for Red, Green, and Blue (RGB)
print('Train dataset shape:', train_imgs.shape,'\tValidation dataset shape:', validation_imgs.shape)


#Pixel values for images are between 0 and 255. Deep Neural networks work well with smaller input values. Scaling each image with values between 0 and 1.
train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

# visualize a sample image
print(train_imgs[1].shape)
img_1 = array_to_img(train_imgs[1])
img_1.show()


# Encoding text category labels of Cats and Dogs
le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)
print(train_labels[1495:1505], train_labels_enc[1495:1505])


# ImageDataGenerator generates batches of tensor image data with real-time data augmentation.
#For our training and validation datasets, we will zoom the image randomly by a factor of 0.3 using the zoom_range parameter. We rotate the image randomly by 50 degrees using the rotation_range parameter. Translating the image randomly horizontally or vertically by a 0.2 factor of the image’s width or height using the width_shift_range and the height_shift_range parameters. Applying shear-based transformations randomly using the shear_range parameter. Randomly flipping half of the images horizontally using the horizontal_flip parameter. Leveraging the fill_mode parameter to fill in new pixels for images after we apply any of the preceding operations (especially rotation or translation). In this case, we just fill in the new pixels with their nearest surrounding pixel values.
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)


# Show some augmented image examples
img_id = 2500
dog_generator = train_datagen.flow(train_imgs[img_id:img_id+1], train_labels[img_id:img_id+1], batch_size=1)
dog = [next(dog_generator) for i in range(0, 6)]
fig, ax = plt.subplots(1, 6, figsize=(16, 6))
print('Labels:', [item[1][0] for item in dog])
l = [ax[i].imshow(dog[i][0][0]) for i in range(0, 6)]
plt.show()


#For our test generator, we need to send the original test images to the model for evaluation. We just scale the image pixels between 0 and 1 and do not apply any transformations.
#We just apply image augmentation transformations only to our training set images and validation image
train_generator = train_datagen.flow(train_imgs, train_labels_enc,batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=30)


# Transfer learning using Pre-trained model as Feature Extractor
# ResNet50
# To implement Transfer learning, we will remove the last predicting layer of the pre-trained ResNet50 model and replace them with our own predicting layers. FC-T1 and FC_T2 as shown below
# Weights of ResNet50 pre-trained model is used as feature extractor
# Weights of the pre-trained model are frozen and are not updated during the training

# We do not want to load the last fully connected layers which act as the classifier. We accomplish that by using “include_top=False”. We do this so that we can add our own fully connected layers on top of the ResNet50 model for our task-specific classification.
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
input_shape=(IMG_HEIGHT,IMG_WIDTH,3)
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, output)
# We freeze the weights of the model by setting trainable as “False”. This stops any updates to the pre-trained weights during training
for layer in restnet.layers:
    layer.trainable = False

print(restnet.summary())


#create our model using Transfer Learning using Pre-trained ResNet50 by adding our own fully connected layer and the final classifier using sigmoid activation function
model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])
print(model.summary())


#run the model
history = model.fit(train_generator,
                              steps_per_epoch=100,
                              epochs=100,
                              validation_data=val_generator,
                              validation_steps=50,
                              verbose=1)

#Saving the trained weights
model.save('cats_dogs_tlearn_img_aug_cnn_restnet50.h5')