import glob
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from tensorflow import keras
import zipfile


with zipfile.ZipFile("train.zip",'r') as z:
    z.extractall(".")
with zipfile.ZipFile("train.zip",'r') as z:
    z.extractall(".")
files = glob.glob('train/*.jpg')


cat_files = [fn for fn in files if 'cat' in fn]
dog_files = [fn for fn in files if 'dog' in fn]
print('total amount of cat files:',len(cat_files),'\ntotal amount of dog files:', len(dog_files))



cat_train = np.random.choice(cat_files, size=1500, replace=False)
dog_train = np.random.choice(dog_files, size=1500, replace=False)
cat_files = list(set(cat_files) - set(cat_train))
dog_files = list(set(dog_files) - set(dog_train))

cat_val = np.random.choice(cat_files, size=500, replace=False)
dog_val = np.random.choice(dog_files, size=500, replace=False)
cat_files = list(set(cat_files) - set(cat_val))
dog_files = list(set(dog_files) - set(dog_val))

cat_test = np.random.choice(cat_files, size=500, replace=False)
dog_test = np.random.choice(dog_files, size=500, replace=False)

print('Cat datasets:', cat_train.shape, cat_val.shape, cat_test.shape)
print('Dog datasets:', dog_train.shape, dog_val.shape, dog_test.shape)



for i in ['training_data','validation_data','testing_data']:
    if os.path.exists(i):
        shutil.rmtree(i)
    os.makedirs(i)
# os.makedirs('training_data',exist_ok=True)
# os.makedirs('validation_data',exist_ok=True)
# os.makedirs('testing_data',exist_ok=True)

train_file_path =np.concatenate([cat_train,dog_train])
for f in train_file_path:
    print(f)
    shutil.move(os.path.join('/Users/jessica/workproject/custom_dataset',f), '/Users/jessica/workproject/custom_dataset/training_data/')

valid_file_path =np.concatenate([cat_val,dog_val])
for f in valid_file_path:
    shutil.move(os.path.join('/Users/jessica/workproject/custom_dataset',f), 'validation_data')

test_file_path =np.concatenate([cat_test,dog_test])
for f in test_file_path:
    shutil.move(f, '/Users/jessica/workproject/custom_dataset/testing_data')