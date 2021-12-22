# Importing essential libraries
from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from pyspark.ml.feature import HashingTF
from pyspark.sql.functions import split
from pyspark.ml.classification import LogisticRegressionModel
import regex as re
import os
# Load the Logistic regression model and Tfidf-vectorizer object from disk
os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk1.8.0_261.jdk/Contents/Home"
spark = SparkSession.builder.appName('Read CSV File into DataFrame'). getOrCreate()
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
app = Flask(__name__)
def text_process(data): 
    if not data:
        return ''
    review_blob = TextBlob(data)
    words = review_blob.words
    sent = ' '.join(words)
    return sent 
def remove_junk(data): #function to keep only characters and remove 'user'- which is not required 
    words=[words for words in data.split() if words != 'user']    
    clean_tokens = [t for t in words if re.match(r'[^\W\d]*$', t)] # Remove punctuations')]
    sent_join  = ' '.join(clean_tokens)
    return sent_join

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        txt_len = len(message.split())
        review = text_process(message)
        review = remove_junk(review)
        review = ' '.join(word.lower() for word in review.split() if word not in stop)
        review = lemmatize_text(review)
        datadict = {'review': review}
        df = spark.createDataFrame([datadict])
        df = df.withColumn('review_array', split(df.review, ' '))
        ht = HashingTF.load("Model/hashing-tf")
        lrModel = LogisticRegressionModel.load("Model/lr_model")
        review = ht.transform(df)
        pred = lrModel.transform(review) 
        result = pred.select('prediction').collect()[0]['prediction']
        if txt_len>0:
            return render_template('result.html', prediction=result)
        else:
            return render_template('result.html', prediction='')
if __name__ == '__main__':
	app.run(debug=True)