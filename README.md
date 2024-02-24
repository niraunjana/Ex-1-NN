<H3>NAME        : NIRAUNJANA GAYATHRI G R</H3>
<H3>REGISTER NO : 212222230096</H3>
<H3>EX. NO.1</H3>
<H3>DATE        : 24-02-24</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:

DATASET:
![307528672-6258c482-3ac4-42a7-bbbe-4d77d1805ae8](https://github.com/niraunjana/Ex-1-NN/assets/119395610/38fa9ae2-a4c1-4011-a775-fa13b997ffdd)
DROPPING THE UNWANTED DATASET:
![307528812-b8a01314-62ba-40bf-a710-44acddc07ebb](https://github.com/niraunjana/Ex-1-NN/assets/119395610/7a5b3937-aa9b-4e36-a838-33e0814368fb)
CHECKING NULL VALUES:
![307528893-bf72ec6b-6df3-4937-9110-d53e72dbc670](https://github.com/niraunjana/Ex-1-NN/assets/119395610/abb6f434-0187-41c6-b6c7-17c1aea6408a)
CHECKING FOR DUPLICATION:
![307528982-567efbe2-458d-4746-9f98-569af1c17045](https://github.com/niraunjana/Ex-1-NN/assets/119395610/21e21c0f-66cc-46e8-9d3d-b436e1ba9749)
DESCRIBING THE DATASET:
![307529026-8a0a5a6f-9b51-447f-8ebc-58bf1985bd58](https://github.com/niraunjana/Ex-1-NN/assets/119395610/bc069d78-4233-4062-a518-52fd4553550e)
SCALING THE DATASET:
![307529078-e0f4884c-b7c6-4eaf-9212-c80e360ce5d2](https://github.com/niraunjana/Ex-1-NN/assets/119395610/622a1c99-4ea2-4bb3-b29d-d8c0b4c95fcc)
X FEATURES:
![307529270-34973f72-79bf-47ee-b363-9c5ede485e08](https://github.com/niraunjana/Ex-1-NN/assets/119395610/8351440b-3346-4236-b131-16e208ae84bc)
Y FEATURES:
![307529299-e15a0d22-a9ef-4c67-b83b-f537ad5e942a](https://github.com/niraunjana/Ex-1-NN/assets/119395610/b3c1fb39-165d-4933-9f06-6bc0c061f843)
SPLITTING THE TRAINING AND TESTING DATASET:
![307529356-75a0e3ed-65e8-4795-a112-0324ac96295e](https://github.com/niraunjana/Ex-1-NN/assets/119395610/66cbcbb6-d473-4d85-b16b-42cd87067584)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


