# Data Science Things
import pandas as pd
import numpy as np

# fast.ai Library
import fastai
from fastai.vision import *
from fastai.vision.models import *
import torch

# Images & Paths
from PIL import ImageFile
from pathlib import Path
import glob

#other
from google.colab import drive
from datetime import date

# setting random seed
np.random.seed(42)
# make sure to change this to what you have, this path will be used for loading test images.
path_to_folder = 'gdrive/My Drive/Dataset/The Picnic Hackathon 2019/'
ImageFile.LOAD_TRUNCATED_IMAGES = True

# getting reference to our model file from drive.
drive.mount('/content/gdrive')

# Loading The Solution Model

# sometimes this cell cause error due Google Drive OSError, just re-run it, it happen only once.
# change this to what it convient for you. Where the model file is ?
path_to_model_file = 'gdrive/My Drive/'
# change this if you have renamed the file. What is the name of the file ?
file_name = 'densenet161_final_model.pkl'
model = load_learner(path = path_to_model_file, file = file_name)
print('Done')

# Predicting the Test Set

# first we get reference to all the fiels in the test set.
files = glob.glob(path_to_folder + 'test/*')
total = len(files)
print('Found {} images'.format(total))

# Lopping over all the file, load -> predict -> and Store the results.
# final array to hold the results.
results = []
# variable to track the progress.
i = 1

for file in files:  
    print("\rImage #{} of {} , Total Progress {}% .".format(i, total, int((i/total)*100)), end="")
    sys.stdout.flush()
    i+=1
    # open the image
    img = open_image(Path(file)).apply_tfms(None, size = 224)
    # predict
    predicted_class, idx, out = model.predict(img)
    # getting file name.
    filename = os.path.basename(file)
    results.append([filename, str(predicted_class)])

# Constructing The Submission file.
headers = ['file', 'label']
submission = pd.DataFrame(results, columns=headers)
submission = submission.sort_values(['file'])

# Make sure the right appearance
submission.head()

# saving the file into the desired format.
today = date.today()
name_file = today.strftime("%d-%m-%y") + '_1.tsv'
submission.to_csv(name_file, sep = '\t', index = False)
