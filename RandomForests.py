# Using magic commands
%matplotlib inline
%load_ext autoreload
%autoreload 2

# Imports of the functions and the libraries

import turicreate as tc
import turicreate.aggregate as agg
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import math
import dateutil.parser

tc.visualization.set_target('browser')
Path = "FastAI/ML/bluebook-for-bulldozers/"

!ls {Path}

# Import data
sf = tc.SFrame.read_csv(Path+'/train.csv', header = True)
sf.head(10)
sf.shape

# Metric is RMSLE, so we will take a log of the sales price
sf['Saleprice_log'] = sf['SalePrice'].apply(lambda x: math.log(x))
sf = sf.remove_column('SalePrice')

# Handle dates
sf['saledate'] = sf['saledate'].apply(lambda x: dateutil.parser.parse(x))
sf = sf.split_datetime('saledate', limit=['year','month','day', 'weekday', 'isoweekday', 'tmweekday'])
sf.show()

# Split train and test
sf['label'] = sf['Saleprice_log']
sf = sf.remove_column('Saleprice_log')
len(sf)
train_data = sf[0:320000]
test_data = sf[320000:len(sf)]
train_data.shape
test_data.shape
seed = 34532
# Let's use the turicreate's internal methods

model = tc.random_forest_regression.create(train_data, target='label', max_iterations= 100, random_seed =seed, verbose=True, max_depth= 40, metric = 'rmse')

model.predict(test_data)

test_data['label']
model.predict(train_data)

train_data['label']

results = model.evaluate(test_data, metric='rmse')
results


# Further iterations

sf.shape
sf_copy = sf
sf_tmp = tc.SFrame()
lst = []
for col in sf_copy.column_names():
    lst.append(((len(sf_copy)- tc.SArray.nnz(sf_copy[col]))/len(sf_copy))*100)

sf_tmp = sf_tmp.add_column(sf.column_names())
sf_tmp = sf_tmp.add_column(lst)

sf_tmp.sort('X2',ascending = False).print_rows(58)

# Many columns have missing values
## Let's treat the numeric missing values but before let's change the SFrame to DataFrame as SFrame isn't letting me work with the dtype

df = tc.SFrame.to_dataframe(sf_copy)

#Numerical missing values
df_1 = df.select_dtypes(include = ['int64', 'float64'])
for col in df_1.columns:
    df_1[col] = df_1[col].fillna(df_1[col].mean())
# or do this
df_1 = df_1.fillna(df_1.mean())

# Categorical data missing values
df_2 = df.select_dtypes(include=['object']).copy()
for col in df_2.columns:
    df_2[col] = df_2[col].astype('category')
print(df_2.dtypes)

for col in df_2.columns:
    df_2[col] = df_2[col].cat.codes


def fill_cat_na(df):
    for dt, col in zip(df.dtypes, df):
        if str(dt) == 'category':
            df[col] = df[col].fillna(df[col].mode().iloc[0])

fill_cat_na(df_2)

pd.DataFrame((df_1.isna().sum()/len(df_1)).sort_values(ascending=False), columns=['Null percent'])
df_f = pd.concat([df_1, df_2], axis =1)
df_f.shape
#Calculate age of the machine
df_f.columns
df_f['age'] = df['YearMade'] - df['saledate.year']

## Train test split

train_f = df_f[0:320000]
test_f = df_f[320000:len(df_f)]
#train_data
y = train_f['label']
df_m = train_f.drop(['label'], axis = 1)

#test_data
y_test = test_f['label']
df_m_test = test_f.drop(['label'], axis = 1)

#Model
m = RandomForestRegressor(n_jobs=-1)
m.fit(df_m, y)
m.score(df_m,y)


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

# Training predictions (to demonstrate overfitting)
train_rf_predictions = m.predict(df_m)

def rmse(x,y):
     return math.sqrt(((x-y)**2).mean())
#train rmse
rmse(train_rf_predictions, y)

# Testing predictions (to determine performance)
rf_predictions = m.predict(df_m_test)
#test score
m.score(df_m_test, y_test)
#test rmse
rmse(rf_predictions, y_test)
