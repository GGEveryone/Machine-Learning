import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split,cross_val_score,KFold
from sklearn.metrics import accuracy_score
from time import time
bank_data = pd.read_csv("bank-full.csv",delimiter=';')

#numeric columns
numeric_col = ['age','balance','day','duration','campaign','pdays','previous']
x_num_train = bank_data[numeric_col].as_matrix()

#categorical columns
bank_train = bank_data.drop(numeric_col+['y'],axis = 1)
x_bank_train = bank_train.T.to_dict().values()

#vectorize
vectorizer = DV( sparse = False )
vec_x_bank_train = vectorizer.fit_transform( x_bank_train )
feature_train = np.hstack(( x_num_train, vec_x_bank_train ))

#y
label_train = bank_data.as_matrix(columns=['y'])


x_train,x_test,y_train,y_test = train_test_split(feature_train,label_train,test_size=0.3,random_state=0)
##parameter tuning

x_train = x_train[:len(feature_train)/100]
y_train = y_train[:len(feature_train)/100]

clf= MLPClassifier()

t0=time()
clf.fit(x_train,y_train)
print "training time is " ,round(time()-t0,3),"s"

t1=time()
cv = KFold(x_train.shape[0], 5)
scores = cross_val_score(clf,x_train,y_train,cv=cv)

print scores
print "Average accuracy score is " + str(round(scores.mean(),3))
print "Cross validation training time",round(time()-t1,3),"s"
print "Current parameters are "+ str(clf.get_params())
t2 = time()
pred = clf.predict(x_test)
print "prediction time",round(time()-t2,3),"s"
accuracy = accuracy_score(pred,y_test)
print "accuracy is" + str(round(accuracy,3))
print()
