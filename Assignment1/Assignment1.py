import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

data = pd.read_csv("titanic_train.csv")
data.head()

survival_count = Counter(data['Survived'])
                  #38% true and 61% false
print("\nDataSet 1\nImbalance: ", (survival_count[0]/(survival_count[0]+survival_count[1]))*100, "%")
print(survival_count) 

fig, ax = plt.subplots(1, 3, sharey=True, tight_layout=True, figsize=(9,3))
ax[0].hist(data['Survived'])

#sns.pairplot(data, hue='Survived');

malesData = data[data['Sex'] == 'male']             #Comparision of survival Rate between Males and Females
femalesData = data[data['Sex'] == 'female']

ax[1].hist(malesData['Survived'])
ax[2].hist(femalesData['Survived'])

data2 = pd.read_csv("creditcard.csv")
data.head()
class_count = Counter(data2['Class'])
print("\nDataSet 2\nImbalance: ", (class_count[0]/(class_count[0]+class_count[1]))*100, "%")
print(class_count)

#downsampling 0 class in dataset2
indexNames = data2[(data2['Class'] == 0)].index
percentageToRemove = 0.98 #Removing 98% of 0 class elements
toRemove = int(class_count[0] * percentageToRemove)
indexesToRemove = random.sample(indexNames.tolist(), toRemove)
data2 = data2.drop(indexesToRemove)
class_count = Counter(data2['Class'])
print("\nDataSet 2 after downsampling\nImbalance: ", (class_count[0]/(class_count[0]+class_count[1]))*100, "%")
print(class_count)

fig2, ax2 = plt.subplots(tight_layout=True)
ax2.hist(data2['Class'])

data3 = pd.read_csv("income_evaluation.csv")
income_count = Counter(data3['income>50k'])
print("\nDataSet 3\nImbalance: ", (income_count[0]/(income_count[0]+income_count[1]))*100, "%")
print(income_count)

fig3, ax3 = plt.subplots(tight_layout=True)
ax3.hist(data3['income>50k'])

#Encoding categorical variables in dataSet1
encoder1 = LabelEncoder()
data['Sex'] = encoder1.fit_transform(data['Sex'])

#Encoding categorical variables in dataSet3
encoder2 = LabelEncoder()
data3['workclass'] = encoder2.fit_transform(data3['workclass'])
encoder3 = LabelEncoder()
data3['fnlwgt'] = encoder3.fit_transform(data3['fnlwgt'])
encoder4 = LabelEncoder()
data3['education'] = encoder4.fit_transform(data3['education'])
encoder5 = LabelEncoder()
data3['marital-status'] = encoder5.fit_transform(data3['marital-status'])
encoder6 = LabelEncoder()
data3['occupation'] = encoder6.fit_transform(data3['occupation'])
encoder7 = LabelEncoder()
data3['relationship'] = encoder7.fit_transform(data3['relationship'])
encoder8 = LabelEncoder()
data3['race'] = encoder8.fit_transform(data3['race'])
encoder9 = LabelEncoder()
data3['sex'] = encoder9.fit_transform(data3['sex'])
encoder10 = LabelEncoder()
data3['native-country'] = encoder10.fit_transform(data3['native-country'])
