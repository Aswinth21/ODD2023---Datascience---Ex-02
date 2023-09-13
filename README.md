# Ex02-Outlier
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

(i) Using IQR detect weight outliers and print them

(ii) Using IQR, detect height outliers and print them

# Aim:
TO detect and remove the outliers in the given data set and save the final data.

# EXPLANATION
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

## ALGORITHM
STEP 1 Read the given Data

STEP 2 Get the information about the data

STEP 3 Detect the Outliers using IQR method and Z score

STEP 4 Remove the outliers

# CODE 
### bhp.csv
```
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("bhp.csv")
q1 = df['price_per_sqft'].quantile(0.25)
q2 = df['price_per_sqft'].quantile(0.5)
q3 = df['price_per_sqft'].quantile(0.75)
iqr = q3-q1
iqr
low = q1-1.5*iqr
low
high = q3+1.5*iqr
high
df = df[((df['price_per_sqft']>=low) & (df['price_per_sqft']<=high))]
df
z = np.abs(stats.zscore(df['price_per_sqft']))
z
df1 = df[z<3]
df1
```
### height_weight.CSV
```
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("height_weight.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
low = q1 - 1.5*iqr
low
high = q3 + 1.5*iqr
high
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
z = np.abs(stats.zscore(df['height']))
z
df1 = df[z<3]
df1
```
# OUTPUT
### bhp.csv
![266902284-2bb4bf97-b308-435c-917a-502bd1ed4f88](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/041e0103-4387-4e79-b13a-50817df88e0b)
![266902325-e356ac1e-be13-4309-98a4-4acc0e040e12](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/7edea532-d1c2-4a7c-a530-c0e11aec470c)
![266902372-41045de4-50c6-4438-84d9-d60cfde78716](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/47c60cc9-23d0-4edc-9e28-98d32d3096e4)
![266902392-1205400e-72eb-4534-8400-fd24928d6b08](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/25327604-830d-412e-8df4-a9ccde724e02)
![266902424-c3fbf939-9606-4560-9ff3-b2c5b9441a0f](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/86716206-d0d0-4376-b36f-fa8b98c6919b)
![266902449-70a38680-c475-4eea-b086-be320bc6dc36](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/7c1cc268-f503-45ba-aedd-358e8ac38914)




### height_weight
![266902509-1b5054b5-66ec-451b-b44e-2e6ad997a59e](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/fb546842-6b6c-4991-b8db-ec848fbf7ef8)
![266902526-8a07a5d1-f27c-4cbb-b647-39e5b44cf9b7](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/8a337d6e-ab86-4b5a-bd81-87e9b30abfc1)
![266902539-5ba40b64-0a20-4d11-88bf-15591ea80026](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/602903bb-f38c-47d7-aa8b-2166df8f63c0)
![266902873-89072b3b-da25-4614-b429-b93c664c4ffb](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/1fdb8cfc-b82a-435c-bb60-b534f0096703)
![266902916-2f95f644-4442-4664-ad5d-5942c3e5dc74](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/e796dda5-1d66-4fc2-a5b9-4f2886152671)
![266902934-86bd7416-d02a-45e7-9128-6da5b9413c1c](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/c6aac1be-aaf7-462b-aa6a-c499e80ef0d0)
![266902966-8221b6f6-b55e-4226-8f6f-96bb6409a033](https://github.com/Aswinth21/ODD2023---Datascience---Ex-02/assets/120236638/d37a6353-9f20-4070-b63d-080308262089)


# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.
