import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data=open("data.txt","w")
post=open("postdata.txt","w")
f1=open("predictsvc.txt","w")
f2=open("predictdt.txt","w")



f=open("X_test.txt","w")
ft=open("tf.txt","w")


"""
**********
Read CSV
Drop unused columns
Rename columns
**********
"""
sms=pd.read_csv('spam.csv',encoding = 'latin-1',nrows=100)
sms=sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
sms=sms.rename(columns={'v1':'label','v2':'message'})


"""
**********
Add length of message to columns
**********
"""
sms['length']=sms['message'].apply(len)


text=sms['message'].copy()

for i in text:
    data.write(i)
    data.write("\n")


"""
**********
Remove punctuations and stopwords.
**********
"""
def text_process(text):

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    return " ".join(text)

tf=TfidfVectorizer("english")
text=text.apply(text_process)
features=tf.fit_transform(text)

for i in text:
    post.write(i)
    post.write("\n")


"""
**********
Find tf-idf of terms.
**********
"""


#"""
for i in features:
    ft.write(str(i))
    ft.write("\n")
#"""


"""
**********
Split the data into Train and Test Data.
**********
"""
X_train, X_test, Y_train, Y_test = train_test_split(features, sms['label'], test_size=0.2, random_state=111)

#"""
for i in (X_test):
    f.write(str(i))
    f.write("\n")
#"""


"""
*************Support Vector Classifier**************
"""

s=SVC(kernel='sigmoid', gamma=1.0)
s.fit(X_train,Y_train)

for i in (X_test):
    f1.write(str(s.predict(i)))
    f1.write("\n")
print("SVC:")
print(s.score(X_test,Y_test))

"""
*************Decision Classifier*************
"""

d=DecisionTreeClassifier(min_samples_split=7, random_state=111)
d.fit(X_train,Y_train)

for i in (X_test):
    f2.write(str(d.predict(i)))
    f2.write("\n")
print("DECISION TREE:")
print(d.score(X_test,Y_test))


#"""
f.close()
ft.close()
#"""


post.close()
data.close()
f1.close()
f2.close()
