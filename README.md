# Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
# Explanation

An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

# ALGORITHM
# STEP 1
Read the given Data

# STEP 2
Get the information about the data

# STEP 3
Detect the Outliers using IQR method and Z score

# STEP 4
Remove the outliers

# STEP 5
Plot the datas using Box Plot

# CODE
(1) & (2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
```
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\bhp.csv")
df

df.head()

df.describe()

df.info()

df.isnull().sum()

df.shape

sns.boxplot(x="price_per_sqft",data=df)
q1 = df['price_per_sqft'].quantile(0.25)
q3 = df['price_Aper_sqft'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df1 =df[((df['price_per_sqft']>=ll)&(df['price_per_sqft']<=ul))]
df1

df1.shape

sns.boxplot(x="price_per_sqft",data=df1)
(3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
from scipy import stats

z = np.abs(stats.zscore(df['price_per_sqft']))
df2 = df[(z<3)]
df2

print(df2.shape)
sns.boxplot(x="price_per_sqft",data=df2)
(4)(i) For the data set height_weight.csv detect weight outliers using IQR method
df3 = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\height_weight.csv")
df3

df3.head()

df3.info()

df3.describe()

df3.isnull().sum()

df3.shape
sns.boxplot(x="weight",data=df3)

q1 = df3['weight'].quantile(0.25)
q3 = df3['weight'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df4 =df3[((df3['weight']>=ll)&(df3['weight']<=ul))]
df4

df4.shape

sns.boxplot(x="weight",data=df4)
(4)(ii) For the data set height_weight.csv detect height outliers using IQR method
sns.boxplot(x="height",data=df3)

q1 = df3['height'].quantile(0.25)
q3 = df3['height'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df5 =df3[((df3['height']>=ll)&(df3['height']<=ul))]
df5

df5.shape

sns.boxplot(x="height",data=df5)
```
# OUTPUT
(1)(2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
# Dataset

![image](https://user-images.githubusercontent.com/127847210/229714412-33d08c4e-a9dc-4785-976f-f658be188cfa.png)

# Dataset Head

![image](https://user-images.githubusercontent.com/127847210/229714484-2dc7aa4f-ca94-4d10-86f0-749ab254277e.png)


# Dataset Info

![image](https://user-images.githubusercontent.com/127847210/229714526-91b26005-65ab-469a-bc7c-4ce4692e790e.png)


# Dataset Describe

![image](https://user-images.githubusercontent.com/127847210/229732272-2b74c3ac-4a37-4df7-b03d-629c82b778d6.png)


# Null Values!

![image](https://user-images.githubusercontent.com/127847210/229765336-ee01ca0f-c2b2-41ad-9e52-b478544d1ae2.png)


# Dataset Shape

![image](https://user-images.githubusercontent.com/127847210/229732429-3e0861aa-38d4-4612-99b7-933b96706163.png)


# Box plot of price_per_sqft column with outliers

![image](https://user-images.githubusercontent.com/127847210/229734212-1531dc8c-c7ee-4e08-82d0-1a16779704b0.png)


# price_per_sqft - Dataset after removing outliers

![image](https://user-images.githubusercontent.com/127847210/229763216-4adbfa58-cb1f-42fa-a3e6-a64ee07bbcd6.png)


# price_per_sqft - Shape of Dataset after removing outliers

![image](https://user-images.githubusercontent.com/127847210/229763297-b660bbb9-965f-4342-aad7-d06f958704a2.png)


# Box Plot of price_per_sqft column without outliers

![image](https://user-images.githubusercontent.com/127847210/229763429-5808f32d-44f4-46a4-b414-1048035b01f6.png)


# (3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
# Dataset after removal of outlier using z score

![image](https://user-images.githubusercontent.com/127847210/229763861-25d63656-5c1a-43b1-990e-39afd25be897.png)


# Shape of Dataset after removal of outlier using z score

![image](https://user-images.githubusercontent.com/127847210/229763958-d572bf50-c0ed-40fa-9c31-1766a4e98889.png)


# price_per_sqft column after removing outliers

![image](https://user-images.githubusercontent.com/127847210/229764060-3e2d8fc9-5db7-4858-92db-5f190bacfffc.png)


# (4) For the data set height_weight.csv detect weight and height outliers using IQR method
# Dataset

![image](https://user-images.githubusercontent.com/127847210/229764166-fd7f7599-4ff6-441b-b68b-280258c8adcf.png)


# Dataset Head

![image](https://user-images.githubusercontent.com/127847210/229764243-906c5724-475e-44fb-92e0-c7b3e927dd77.png)

# Dataset Info

![image](https://user-images.githubusercontent.com/127847210/229764289-477b6eba-d9cd-492a-8900-296c0e1c3951.png)


# Dataset Describe

![image](https://user-images.githubusercontent.com/127847210/229764328-45122825-4494-4004-9d43-53508963cbb3.png)


# Null Values

![image](https://user-images.githubusercontent.com/127847210/229764396-2f44be2d-cb4e-4ad8-811e-08b8912dedcc.png)


# Dataset Shape

![image](https://user-images.githubusercontent.com/127847210/229764502-82265b9c-9547-437d-b511-d7ca61400f5d.png)


# Weight - With outliers

![image](https://user-images.githubusercontent.com/127847210/229764579-9279d038-a2bd-45ac-b3f8-cf61144b8861.png)


# Weight - Dataset after removing Outliers using IQR method

![image](https://user-images.githubusercontent.com/127847210/229764657-6003b88f-d0e6-44e4-ad9f-466b33d6b223.png)


# Weight - Shape of Dataset after removing Outliers using IQR method

![image](https://user-images.githubusercontent.com/127847210/229764744-2de57f67-fd7d-4bf3-a7ed-201e3adc9d09.png)


# Weight - Without Outliers using IQR method

![image](https://user-images.githubusercontent.com/127847210/229764824-166a42eb-0a45-4f44-b2c2-a01f3267d150.png)

# Height - With outliers

![image](https://user-images.githubusercontent.com/127847210/229764894-c423a9aa-7553-4d80-80f3-43f68e17e228.png)

# Height - Dataset after removing Outliers using IQR method

![image](https://user-images.githubusercontent.com/127847210/229764987-4302643a-8241-4c69-a756-3892141d9bc8.png)

# Height - Shape of Dataset after removing Outliers using IQR method

![image](https://user-images.githubusercontent.com/127847210/229765049-ba9764ef-ac61-4f62-a9a0-bce494fc8ef1.png)

# Height - Without Outliers using IQR method

![image](https://user-images.githubusercontent.com/127847210/229765098-1be2c765-10dc-44c7-9592-348792bfca84.png)


# RESULT

The given datasets are read and outliers are detected and are removed using IQR and z-score methods.
