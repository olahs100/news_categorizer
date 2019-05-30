# Dependencies 
import sys
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import tensorflow as tf
from urllib.request import Request, urlopen 
from nltk.stem.porter import PorterStemmer
from keras.models import load_model
from nltk.tokenize import word_tokenize


def category_output(prediction):
    if np.argmax(prediction) == 0:
        cate_predict = {"status": "success", "category": "entertainment", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    elif np.argmax(prediction) == 1:
        cate_predict = {"status": "success", "category": "sport", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    elif np.argmax(prediction) == 2:
        cate_predict = {"status": "success", "category": "politics", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    elif np.argmax(prediction) == 3:
        cate_predict = {"status": "success", "category": "business", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    elif np.argmax(prediction) == 4:
        cate_predict = {"status": "success", "category": "tech", "category id": str(np.argmax(prediction)), "prediction rate": str(np.max(prediction)*100)}
    return cate_predict

# Your API definition
app = Flask(__name__)

@app.route('/categorize', methods=['POST'])
def categorize():
    
    if textmodel:
        datan = []
        try:
            json_ = request.json
            print(json_)
#            query = pd.DataFrame(json_)
            query = json_
            
            for i in range(len(query)):
                url = (query[i])
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                webpage = urlopen(req).read()
                #    response = requests.get(url)   
                soup = BeautifulSoup(webpage, "html5lib")
                for a in soup.find_all('a'): 
                    a.decompose()
                contt = soup.findAll('p')
                testdata = [re.sub(r'<.+?>',r'',str(x)) for x in contt]
                snh = ' '
                testdata = snh.join(testdata)
                linessn = testdata.strip()
                porter = PorterStemmer()
                liness = porter.stem(linessn)                
                Yb = tfidf_model.transform([liness])
                with graph1.as_default():
                     prediction = textmodel.predict(Yb)            
                     print(np.argmax(prediction))
                     datan.append(category_output(prediction))
                
            return jsonify(datan)

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 6000 # If you don't provide any port the port will be set to 12345

    tfidf_model = joblib.load(r"C:\Users\olahs\Documents\Python_scripts\news_categorizer\tfidfmodel.pkl") 
    print ('text vectorizer loaded')
    
    textmodel = load_model('C:/Users/olahs/Documents/Python_scripts/news_categorizer/News_Cate.h5')
    graph1 = tf.get_default_graph()
    print ('Model loaded')

    app.run(port=port, debug=True)
