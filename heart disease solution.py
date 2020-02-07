#!/usr/bin/env python
# coding: utf-8

# In[31]:


##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle



# In[2]:


#we are reading our data
df = pd.read_csv('heart.csv')


# In[3]:


# First 5 rows of our data
df.head()


# In[4]:


#check for total no of data
df.shape


# ## Data contains;
# 
# 1. age - age in years
# 2. sex - (1 = male; 0 = female)
# 3. cp - chest pain type
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# 5. chol - serum cholestoral in mg/dl
# 6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. restecg - resting electrocardiographic results
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest
# 11. slope - the slope of the peak exercise ST segment
# 12. ca - number of major vessels (0-3) colored by flourosopy
# 13. thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. target - have disease or not (1=yes, 0=no)

# In[ ]:





# ## Data Exploration

# In[5]:


df.target.value_counts()


# In[6]:


#visualize target variable
sns.countplot(x="target", data=df, palette="bwr")
plt.show()


# In[7]:


#check for percentages of pateints and non_patients
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))


# In[8]:


#visualize gender of our data
sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()


# In[9]:


#check for gender percentages 
countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))


# In[10]:


df.groupby('target').mean()


# In[11]:


#visualize if sick or not for each age
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In[12]:


# visualize if sick or not for each gender
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[13]:


#scatter plot for age against age
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[14]:


pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()


# In[15]:


#visualize heart disease according to fasting blood sugar
pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()


# In[16]:


#visualize heart disease according to chest pain
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()


# # Convert our categorial data to numerical(dummies)

# In[17]:


a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")


# In[18]:


#concat the data and check it
frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()


# In[19]:


#drop our initial categorical data
df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()


# In[20]:


df.shape


# ## separate our target and features data 

# In[21]:


y = df.target.values
x_data = df.drop(['target'], axis = 1)


# ## normalize our data

# In[22]:


# Normalize
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values


# ## We will split our data. 80% of our data will be train data and 20% of it will be test data.

# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)


# In[24]:


clf = LogisticRegression(C=1e40, solver='newton-cg')


# In[25]:


model = clf.fit(x, y)


# In[26]:


y_pred=model.predict(x_test)


# In[27]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[28]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[32]:


#save model to disk
pickle.dump(model, open('model.pkl', 'wb'))


# In[33]:


#loading the model to compare results
saved_model = pickle.load(open('model.pkl', 'rb'))


# In[34]:


y_pred_savedmodel = saved_model.predict(x_test)


# In[35]:


cnf_matrix2 = metrics.confusion_matrix(y_test, y_pred_savedmodel)
cnf_matrix2


# In[ ]:




