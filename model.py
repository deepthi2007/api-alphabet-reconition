import pandas as pd
from PIL import Image,ImageOps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

#fetch the data
x = np.load("image.npz")['arr_0']
y = pd.read_csv("labels.csv")['labels']

#st the classes
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#no.of classes
nclasses = len(classes)

#split data to train and test
x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=3500, test_size=500, random_state=9)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf = LogisticRegression(solver='saga',multi_class='multinomial').fit(x_train_scaled,y_train)

#creating a function for api
def getPrediction(image):
    impil = Image.open(image)
    img_bw = impil.convert('L')
    img_bw_resized = img_bw.resize((22,30),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel =np.percentile(img_bw_resized,pixel_filter)
    img_bw_resized_inverted_scaled= np.clip(img_bw_resized-min_pixel,0,255)
    max_pixel = np.max(img_bw_resized)
    img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel

    test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1,660)
    test_pred = clf.predict(test_sample)

    return test_pred[0]