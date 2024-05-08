<H3>NAME        : NIRAUNJANA GAYATHRI G R</H3>
<H3>REGISTER NO : 212222230096</H3>
<H3>EX. NO.1</H3>
<H3>DATE        : </H3>
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

![307528672-6258c482-3ac4-42a7-bbbe-4d77d1805ae8](https://github.com/niraunjana/Ex-1-NN/assets/119395610/d6b8d242-a266-41f9-9f08-eacb96428bbe)

DROPPING THE UNWANTED DATASET:

![307528812-b8a01314-62ba-40bf-a710-44acddc07ebb](https://github.com/niraunjana/Ex-1-NN/assets/119395610/d4d85de8-b6c4-4703-a594-2f93c30fb514)

CHECKING NULL VALUES:

![307528893-bf72ec6b-6df3-4937-9110-d53e72dbc670](https://github.com/niraunjana/Ex-1-NN/assets/119395610/16b35620-2a95-4f8a-bbca-fff22e0a83e6)

CHECKING FOR DUPLICATION:

![307528982-567efbe2-458d-4746-9f98-569af1c17045](https://github.com/niraunjana/Ex-1-NN/assets/119395610/af0d6e5d-8a52-410d-901f-236887b512e6)

DESCRIBING THE DATASET:

![307529026-8a0a5a6f-9b51-447f-8ebc-58bf1985bd58](https://github.com/niraunjana/Ex-1-NN/assets/119395610/7efcd58e-5b06-4ab3-9e9a-df7d9a26d0f3)

SCALING THE DATASET:

![307529078-e0f4884c-b7c6-4eaf-9212-c80e360ce5d2](https://github.com/niraunjana/Ex-1-NN/assets/119395610/1381b730-fb62-44cd-861a-7d260a7a55d2)

X FEATURES:

![307529270-34973f72-79bf-47ee-b363-9c5ede485e08](https://github.com/niraunjana/Ex-1-NN/assets/119395610/42e2f068-d798-4721-96ec-555c95d97cd0)

Y FEATURES:

![307529299-e15a0d22-a9ef-4c67-b83b-f537ad5e942a](https://github.com/niraunjana/Ex-1-NN/assets/119395610/6df83942-4ebf-470e-85a7-71109e9d6214)

SPLITTING THE TRAINING AND TESTING DATASET:

![307529356-75a0e3ed-65e8-4795-a112-0324ac96295e](https://github.com/niraunjana/Ex-1-NN/assets/119395610/87107db2-18d0-4ed3-a541-4a8d631bec35)

### RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


