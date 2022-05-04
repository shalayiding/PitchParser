#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' # Allow only 1 GPU visible to CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable annoying logs


import tensorflow as tf
from tensorflow.keras import datasets, layers, models , losses
# import simpleaudio as sa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# In[2]:


def _parseme(raw_audio_record):
	feature_description = {
	    'note': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'note_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
	    'instrument': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'instrument_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
	    'pitch': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'velocity': tf.io.FixedLenFeature([], tf.int64,default_value=0),
	    'sample_rate': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'audio': tf.io.FixedLenSequenceFeature([], tf.float32,  allow_missing=True, default_value=0.0),
	    'qualities': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
	    'qualities_str': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value=''),
	    'instrument_family': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'instrument_family_str': tf.io.FixedLenFeature([], tf.string, default_value=''),
	    'instrument_source': tf.io.FixedLenFeature([], tf.int64, default_value=0),
	    'instrument_source_str': tf.io.FixedLenFeature([], tf.string, default_value='')     
	}

	return tf.io.parse_single_example(raw_audio_record, feature_description)



def extract_data(ds_data,input_size,test_size):
  x = []
  y = []
  count = 0
  for element in ds_data:
    count+=1
    if count > test_size:
        break;
    if int(element['instrument_source'].numpy()) == 9:
      continue
    else:
      temp = np.resize(element['audio'][:16000].numpy(),100*160)
      x.append(temp)
#       x.append(np.resize(element['audio'].numpy(),input_size*input_size))
      if int(element['instrument_source'].numpy()) == 10:
        y.append(element['instrument_source'].numpy()-1)
      else :
        y.append(element['instrument_source'].numpy())
   
  x = np.array(x)
  y = np.array(y)
  return x,y


# In[3]:


test_raw_data = tf.data.TFRecordDataset("/local/sandbox/nsynth-tf/nsynth-test.tfrecord")


# In[4]:


input_size = 16000
test_size = 22679
test_ar = test_raw_data.map(_parseme)
x,y = extract_data(test_ar,input_size,test_size)
x = np.expand_dims(x,axis=-1)


# In[5]:


complex_network_Model=load_model('project2_bonus_model.h5')
complex_network_Model.summary()


# In[6]:


result =  complex_network_Model.evaluate(x, y)


# In[7]:


prediction = complex_network_Model.predict(x[:,:])


# In[8]:


# print(x.shape)
# print(y.shape)
# x_rechange = []
# for i in x:    
#     tmp = i[:,:,0]
#     tmp = tmp.flatten()
#     x_rechange.append(tmp)

# x_rechange = np.array(x_rechange)
# print(x_rechange.shape)
# x = x_rechange


# In[13]:


print(prediction[0])
print(y[0])
print(len(y))
print(len(prediction))


# In[15]:


true_table = [[0,0],[0,0],[0,0]]
miss_classified = []
prediciton_by_class = []
right_classified = []
for i in range(len(prediction)):
    prediciton_by_class.append(np.argmax(prediction[i]))
    if y[i] == np.argmax(prediction[i]):
        true_table[y[i]][0]+=1
        right_classified.append([y[i],prediction[i],x[i],0])
    else :
        miss_classified.append([y[i],np.argmax(prediction[i])])
        true_table[y[i]][1]+=1


accuracy_set = []
for i in range(len(true_table)):
    right = true_table[i][0]
    wrong = true_table[i][1]
    accuracy = right/(right+wrong+1)
    accuracy_set.append(accuracy)
print(accuracy_set)
data = [0,1,2]
plt.hist(data,weights= accuracy_set,color = 'lightblue',ec="red")
plt.savefig('project2_bones_histogram.png')
plt.figure().clear()
print(accuracy_set)
disp = ConfusionMatrixDisplay(confusion_matrix(y, prediciton_by_class, labels=[i for i in range(3)]))
disp.plot()
plt.savefig('project2_bones_confusionmatrix.png')
plt.figure().clear()


# In[19]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
visited = []
for i in right_classified:
    visited.append( [i[1].max(),i[0],i[2]] )
visited = sorted(visited, key=lambda test: test[0],reverse=True)


for clas in range(0,3):
    r = clas // 2
    c = clas % 2
    ax = axes[r][c]
    count = 0
    for i in visited:
        if i[1] == clas:
          if count < 5:
            count+=1
            ax.plot(i[2])
            ax.set_title('classes:'+str(clas))
plt.show()
plt.savefig('project2_bones_high_prob_waveform.png')


# In[20]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
visited = []
for i in right_classified:
    visited.append( [i[1].max(),i[0],i[2]] )
visited = sorted(visited, key=lambda test: test[0])

for clas in range(0,3):
    r = clas // 2
    c = clas % 2
    ax = axes[r][c]
    count = 0
    for i in visited:
        if i[1] == clas:
          if count < 5:
            count+=1
            ax.plot(i[2])
            ax.set_title('classes:'+str(clas))
plt.show()
plt.savefig('project2_bones_low_prob_waveform.png')


# In[ ]:





# In[ ]:





# In[ ]:




