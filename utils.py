# helper functions in the IPYNB file.

# exporting result
import glob
import os
import cv2
files = glob.glob(path_to_folder + 'test/*')

results = []
i = 0
for file in files:
    print(i)
    i+=1
    img = cv2.imread(file,1)
    img = cv2.resize(img, (224, 224)) 
    img = img/255
    result = model.predict([[img]])
    label = labels[np.argmax(result)]
    filename = os.path.basename(file)
    results.append([filename, label])

headers = ['file', 'label']
df = pd.DataFrame(results, columns=headers)
df.head()

df = df.sort_values(['file'])

df.to_csv("submission.tsv", sep ='\t', index = False)

# save model

model_json = model.to_json()
with open("densnetmodel.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("densenet.model.h5")


# make model
from keras.applications.resnet50 import ResNet50
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16

def make_model(name):
  model = Sequential()
  if name == 'resnet':
    model.add(ResNet50(include_top=False, weights='imagenet', pooling='avg'))
  #if name == 'resnetxt':
  # model.add(ResNeXt50(include_top=False, weights='imagenet', pooling='avg'))
  elif name == 'mobilenet':
    model.add(MobileNetV2(include_top=False, weights='imagenet', pooling='avg'))
  elif name == 'nasnet':
    model.add(NASNetMobile(include_top=False, weights='imagenet', pooling='avg'))
  elif name == 'densenet':
    model.add(DenseNet121(include_top=False, weights='imagenet', pooling='avg'))
  elif name == 'vgg16':
    model.add(VGG16(include_top=False, weights='imagenet', pooling='avg'))
    
  model.add(Dense(25, activation = 'softmax'))
  model.summary()
  
  return model  

# train model
def train_model(model, epochs, generator = True):
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])
  if generator:
    history = model.fit_generator(train_generator, epochs= epochs, 
                                  steps_per_epoch= train_generator.n//train_generator.batch_size, 
                                  validation_steps=valid_generator.n//valid_generator.batch_size, 
                                  validation_data = valid_generator)
  else:
    history = model.fit(x_train, y_train, epochs= epochs, batch_size = 32, validation_data= (x_test,y_test))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.show()
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.show()

