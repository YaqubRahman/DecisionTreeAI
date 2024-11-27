import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'

my_data = pd.read_csv(url)

print(my_data.head())

X = my_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values
print(X[0:5])

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1]= le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW','NORMAL','HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])


print(X[0:5])



y = my_data["Drug"]
print(y[0:5])


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)


print(drugTree)



drugTree.fit(X_trainset, y_trainset)

predTree = drugTree.predict(X_testset)




print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


plt.figure(figsize=(12, 8))
tree.plot_tree(drugTree, filled=True, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], class_names=drugTree.classes_)
plt.show()













X_trainset, X_testset, y_trainset, y_testset = train_test_split(x,y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

drugTree.fit
