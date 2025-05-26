import sklearn
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
# Load dataset
df = pd.read_csv('Titanic-Dataset.csv')

#show first few rows
 #first few rows
print(df.head()) 
 #Basic info
print(df.info()) 
 #Statistical summary
print(df.describe()) 

# Handling missing values
  #Checking missing values
print("Null values count:\n")  
print(df.isnull().sum())

#Dropping column with excessive missing data
df.drop(columns=['Cabin'], inplace=True)

# Fill with mean or median using imputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['Age']] = imputer.fit_transform(df[['Age']])
  
embarked_imputer = SimpleImputer(strategy='most_frequent')
df[['Embarked']] = embarked_imputer.fit_transform(df[['Embarked']])  

# converting categorical features to numerical
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])           # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])

# Normalizing data
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
df[['Age','Fare']]= sc.fit_transform(df[['Age','Fare']])

# Visualize and remove outliers using box plot
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot before removing outliers
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Age'])
plt.title('Age Boxplot')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Fare'])
plt.title('Fare Boxplot')
plt.tight_layout()
plt.show()

# remove outliers
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


df_cleaned = remove_outliers_iqr(df, 'Age')
df_cleaned = remove_outliers_iqr(df_cleaned, 'Fare')

# new box plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

# Age Boxplot
plt.subplot(1, 2, 1)
sns.boxplot(x=df_cleaned['Age'], notch=True, color='skyblue')
plt.title('Age Boxplot (No Outliers)')

# Fare Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=df_cleaned['Fare'], notch=True, color='lightgreen')
plt.title('Fare Boxplot (No Outliers)')

plt.tight_layout()
plt.show()

# box plot for other variables
plt.figure(figsize=(12, 5))

# Age by Pclass
plt.subplot(1, 2, 1)
sns.boxplot(x='Pclass', y='Age', data=df_cleaned, palette='pastel', notch=True)
plt.title('Age by Pclass')

# Age by Survived
plt.subplot(1, 2, 2)
sns.boxplot(x='Survived', y='Age', data=df_cleaned, palette='Set2', notch=True)
plt.title('Age by Survival')

plt.tight_layout()
plt.show()
