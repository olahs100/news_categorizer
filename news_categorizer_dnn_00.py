# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:01:14 2019

@author: olahs
"""

import os
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import pickle
from nltk.stem.porter import PorterStemmer
from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from keras.models import load_model
import numpy

path = "C:/Users/olahs/Documents/Python_scripts/news_categorizer/data_set/bbc"

entertain = os.listdir(os.chdir(os.path.join(path, 'entertainment')))

files_entertain = []
target_entertain = np.array([1] * len(entertain))
for sample in entertain:
    sl = open(sample, 'r').read()
    slnn = sl.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = sl.translate(table)
    files_entertain.append(stripped)


sport = os.listdir(os.chdir(os.path.join(path, 'sport')))

files_sport = []
target_sport = np.array([2] * len(sport))
for sample in sport:
    spr = open(sample, 'r').read()
    sprn = spr.split()
    table = str.maketrans('', '', string.punctuation)
    sport_stripped = spr.translate(table)
    files_sport.append(sport_stripped)



politics = os.listdir(os.chdir(os.path.join(path, 'politics')))

files_politics = []
target_politics = np.array([3] * len(politics))
for sample in politics:
    pol = open(sample, 'r').read()
    poln = pol.split()
    table = str.maketrans('', '', string.punctuation)
    pol_stripped = pol.translate(table)
    files_politics.append(pol_stripped)



business = os.listdir(os.chdir(os.path.join(path, 'business')))

files_business = []
target_business = np.array([4] * len(business))
for sample in business:
    buss = open(sample, 'r').read()
    bussn = buss.split()
    table = str.maketrans('', '', string.punctuation)
    buss_stripped = buss.translate(table)
    files_business.append(buss_stripped)


tech = os.listdir(os.chdir(os.path.join(path, 'tech')))

files_tech = []
target_tech = np.array([5] * len(tech))
for sample in tech:
    tec = open(sample, 'r').read()
    tecn = tec.split()
    table = str.maketrans('', '', string.punctuation)
    tec_stripped = tec.translate(table)
    files_tech.append(tec_stripped)
#files_tech.append(open(sample, 'r').read())
    

#fashion = os.listdir(os.chdir(os.path.join(path, 'fashion')))
#delete_list = ["HuffPost", "Reporter", "Lifestyle", "Canada"]
#files_fashion = []
#target_fashion = np.array([6] * len(fashion))
#for sample in fashion:
#    fash = open(sample, encoding = 'utf-8').read()
#    for word in delete_list: 
#        fash_line = fash.replace(word,"")
#    fashn = fash.split()
#    table = str.maketrans('', '', string.punctuation)
#    fash_stripped = fash_line.translate(table)
#    files_fashion.append(fash_stripped)
    
#    with open(sample, "r") as f:
#        if f in entern:
#            continue
#        f[sample] = sample.read()
print(files_entertain)
print(files_sport)
print(files_politics)
print(files_business)
print(files_tech)
#print(files_fashion)

datan = []
datan.extend(files_entertain)
datan.extend(files_sport)
datan.extend(files_politics)
datan.extend(files_business)
datan.extend(files_tech) 
#datan.extend(files_fashion) 

            
porter = PorterStemmer()
data = [porter.stem(word) for word in datan]

Target = np.concatenate((target_entertain, target_sport, target_politics, target_business, target_tech))


#### splitting data into training and testing set
X_trainn, X_testt, y_train, y_test = train_test_split(data, Target, random_state=0, test_size=0.35)

#features = tfidf.fit_transform(data) 
#print(features.shape)

######################################################################### 
#########################################################################
enc_length = 3000
tfidf = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english', max_features = enc_length)
tfidf_model = tfidf.fit(X_trainn)

pickle.dump(tfidf_model, open(r"C:\Users\olahs\Documents\Python_scripts\news_categorizer\tfidfmodel.pkl", "wb"))

# ------------------------------------------------------------------------------------------#
############################################################################
############################################################################ 
# fix random seed for reproducibility
#numpy.random.seed(2090)


# create model
epoch = 50
batch = 150

# alternative model
text_model = Sequential()
text_model.add(Dense(500, input_dim=enc_length, activation='relu'))
text_model.add(Dense(400, activation='relu')) 
text_model.add(Dense(200, activation='relu')) 
text_model.add(Dense(100, activation='relu')) 
text_model.add(Dense(5, activation='sigmoid'))

train_feat = tfidf_model.transform(X_trainn)
test_feat = tfidf_model.transform(X_testt)
Y = pd.get_dummies(y_train).values
Yt = pd.get_dummies(y_test).values
# Compile model
text_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
text_model.fit(train_feat, Y, epochs=epoch, batch_size=batch)
# evaluate the model
scores = text_model.evaluate(test_feat, Yt)
print("\n%s: %.2f%%" % (text_model.metrics_names[1], scores[1]*100))


text_model.save('C:/Users/olahs/Documents/Python_scripts/news_categorizer/News_Cate.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one

#model = load_model('C:/Users/olahs/Documents/Python_scripts/news_categorizer/News_Cate.h5')

################################################################### 
################################################################### 

path = "C:/Users/olahs/Documents/Python_scripts/news_categorizer/data_set/testt.txt"
#entertain = os.listdir(os.chdir(os.path.join(path, 'entertainment')))
testdata = open(path, 'r').read()

linessn = testdata.strip()


#    strippedn = word_tokenize(striptab) 
#    snh = '. '
#    stripped = snh.join(strippedn)
 
porter = PorterStemmer()
liness = porter.stem(linessn) 

Yb = tfidf_model.transform([liness])
prediction = text_model.predict(Yb)
print(np.argmax(prediction))

if np.argmax(prediction) == 0:
    print("This is", np.max(prediction)*100, "entertainment news")
elif np.argmax(prediction) == 1:
    print("This is", np.max(prediction)*100, " sport news")
elif np.argmax(prediction) == 2:
    print("This is", np.max(prediction)*100, " politics news")
elif np.argmax(prediction) == 3:
    print("This is", np.max(prediction)*100, " business news")
elif np.argmax(prediction) == 4:
    print("This is", np.max(prediction)*100, " tech news")
#elif np.argmax(prediction) == 5:
#    print("This is", np.max(prediction)*100, " fashion news")

print("The accuracy of prediction for DNN is: ", np.max(prediction)*100)
#print(prediction)
# ------------------------------------------------------------------------------------------#