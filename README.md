# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: TARANIKKA A
RegisterNumber:  212223220115
*/
```
```
import chardet

file='spam.csv'

with open (file,'rb') as rawdata:

result = chardet.detect(rawdata.read(100000))
    
result

import pandas as pd

data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

x_train=cv.fit_transform(x_train)

x_test=cv.transform(x_test)

from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

y_pred

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
```

## Output:

### ENCODING

![445905030-d3e2f384-d6f3-4a6d-a133-1aa63ca1a0c3](https://github.com/user-attachments/assets/6f142d95-2f99-4c17-8616-2133a2357fc6)

### HEAD():

![445905102-a704d324-ac39-4cd7-a035-81a562b7d785](https://github.com/user-attachments/assets/3466f4ee-e21d-4800-899d-add02a9e7294)

### isnul().sum

![445905277-352a0781-826d-4cf3-840d-e796c0da0685](https://github.com/user-attachments/assets/119ced1d-d41c-49ad-9a39-a1455487c620)

### Prediction of Y

![445905344-c5d1c084-23f3-4a63-9a25-da2425f702fe](https://github.com/user-attachments/assets/e8c35a43-b33f-4210-8f53-cde5818b5bc6)

### Accuracy

![445905400-799a522e-1ead-4353-b879-f1a257de489b](https://github.com/user-attachments/assets/18880cc2-58d0-464f-90e3-2d7e49fadf08)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
