
```python
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics 
```


```python
# dataset link - https://raw.githubusercontent.com/johnmyleswhite/ML_for_Hackers/master/02-Exploration/data/01_heights_weights_genders.csv
```


```python
df= pd.read_csv("lr_dataset.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Male</td>
      <td>73.847017</td>
      <td>241.893563</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Male</td>
      <td>68.781904</td>
      <td>162.310473</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Male</td>
      <td>74.110105</td>
      <td>212.740856</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Male</td>
      <td>71.730978</td>
      <td>220.042470</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Male</td>
      <td>69.881796</td>
      <td>206.349801</td>
    </tr>
  </tbody>
</table>
</div>




```python
# test data distribution  
df['Gender'].value_counts()
```




    Male      5000
    Female    5000
    Name: Gender, dtype: int64




```python
df.columns
```




    Index(['Gender', 'Height', 'Weight'], dtype='object')




```python
# Mapping data related to male/female
mapping = {'Male': 1,'Female': 0}
df['Gender'] = df['Gender'].map(mapping)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>73.847017</td>
      <td>241.893563</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>68.781904</td>
      <td>162.310473</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>74.110105</td>
      <td>212.740856</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>71.730978</td>
      <td>220.042470</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>69.881796</td>
      <td>206.349801</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df[['Height', 'Weight']]
y = df['Gender']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, test_preds)
print(f"Accurracy = {accuracy}")

print(f"\n----------------------------Training --------------------------------------------\n")
print(metrics.classification_report(y_train, train_preds))
print(f"\n----------------------------Testing ---------------------------------------------\n")
print(metrics.classification_report(y_test, test_preds))
```

    Accurracy = 0.9226666666666666
    
    ----------------------------Training --------------------------------------------
    
                  precision    recall  f1-score   support
    
               0       0.92      0.92      0.92      3537
               1       0.92      0.92      0.92      3463
    
        accuracy                           0.92      7000
       macro avg       0.92      0.92      0.92      7000
    weighted avg       0.92      0.92      0.92      7000
    
    
    ----------------------------Testing ---------------------------------------------
    
                  precision    recall  f1-score   support
    
               0       0.93      0.91      0.92      1463
               1       0.92      0.93      0.93      1537
    
        accuracy                           0.92      3000
       macro avg       0.92      0.92      0.92      3000
    weighted avg       0.92      0.92      0.92      3000
    


    /home/nitin/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)



```python
# For b0:
b0 = model.intercept_
b0=  b0[0]
print(b0)

# For b1 and b2:
b = model.coef_
b1 = b[0][0]
b2 = b[0][1]
print(b1, b2)
```

    -0.00956930980444511
    -0.4716648774161416 0.19413023855020817



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>73.847017</td>
      <td>241.893563</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>68.781904</td>
      <td>162.310473</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>74.110105</td>
      <td>212.740856</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>71.730978</td>
      <td>220.042470</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>69.881796</td>
      <td>206.349801</td>
    </tr>
  </tbody>
</table>
</div>



## Let's take positive example 


```python
# 1	73.847017	241.893563

x1 = 73.847017
x2 = 241.893563
```


```python
y=b0+b1*x1+b2*x2
z = 1/(1+np.exp(-y))
```


```python
z
```




    0.9999945410113663



## Let's take negative example 


```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9995</td>
      <td>0</td>
      <td>66.172652</td>
      <td>136.777454</td>
    </tr>
    <tr>
      <td>9996</td>
      <td>0</td>
      <td>67.067155</td>
      <td>170.867906</td>
    </tr>
    <tr>
      <td>9997</td>
      <td>0</td>
      <td>63.867992</td>
      <td>128.475319</td>
    </tr>
    <tr>
      <td>9998</td>
      <td>0</td>
      <td>69.034243</td>
      <td>163.852461</td>
    </tr>
    <tr>
      <td>9999</td>
      <td>0</td>
      <td>61.944246</td>
      <td>113.649103</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 0	66.172652	136.777454

x1 = 66.172652
x2 = 136.777454
```


```python
y=b0+b1*x1+b2*x2
z = 1/(1+np.exp(-y))
print(z)
```

    0.00930140086616673



```python
if y>=0:
then z>0.5 

means class 1
```


```python
# 0	60.634246	136.879127
y = 0 + (-0.48009418*60.634246)+ (0.19746617*136.879127)
y
```




    -2.081151651654686




```python
if y>=0:
then z>0.5 

means class 1
```


      File "<ipython-input-45-7926e63e1fdb>", line 2
        then z>0.5
           ^
    IndentationError: expected an indented block



## K-fold validation


```python
df["kfold"]=-1
df = df.sample(frac=1).reset_index(drop=True)
y =df.Gender.values
kf= model_selection.StratifiedKFold(n_splits=5)
for f, (t_,v_) in enumerate(kf.split(X=df, y=y)):
    print(f, len(t_), len(v_))
    df.loc[v_, 'kfold'] = f
```

    0 8000 2000
    1 8000 2000
    2 8000 2000
    3 8000 2000
    4 8000 2000



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Height</th>
      <th>Weight</th>
      <th>kfold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>67.719814</td>
      <td>171.731928</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>71.353209</td>
      <td>209.309880</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>64.130341</td>
      <td>121.292597</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>62.570682</td>
      <td>126.912850</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>64.548972</td>
      <td>148.787471</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
for fold_ in range(5):
    train_df = df[df.kfold!=fold_].reset_index(drop=True)
    test_df = df[df.kfold==fold_].reset_index(drop=True)
    
    print(f"**10")
    print(f"Fold: {fold_}")
    
    model = linear_model.LogisticRegression()
    model.fit(train_df[['Height','Weight']], train_df['Gender'])
    
    train_preds = model.predict(train_df[['Height','Weight']])
    test_preds = model.predict(test_df[['Height','Weight']])
    
    accuracy = metrics.accuracy_score(test_df.Gender, test_preds)
    print(f"Accurracy = {accuracy}")
    print("")
    
    print(f"\n------------Training -------\n")
    print(metrics.classification_report(train_df.Gender, train_preds))
    print(f"\n------------Testing -------\n")
    print(metrics.classification_report(test_df.Gender, test_preds))
    
```

    **10
    Fold: 0
    Accurracy = 0.914
    
    
    ------------Training -------
    
                  precision    recall  f1-score   support
    
               0       0.92      0.92      0.92      4000
               1       0.92      0.92      0.92      4000
    
        accuracy                           0.92      8000
       macro avg       0.92      0.92      0.92      8000
    weighted avg       0.92      0.92      0.92      8000
    
    
    ------------Testing -------
    
                  precision    recall  f1-score   support
    
               0       0.91      0.92      0.91      1000
               1       0.92      0.91      0.91      1000
    
        accuracy                           0.91      2000
       macro avg       0.91      0.91      0.91      2000
    weighted avg       0.91      0.91      0.91      2000
    
    **10
    Fold: 1
    Accurracy = 0.9175
    
    
    ------------Training -------
    
                  precision    recall  f1-score   support
    
               0       0.92      0.92      0.92      4000
               1       0.92      0.92      0.92      4000
    
        accuracy                           0.92      8000
       macro avg       0.92      0.92      0.92      8000
    weighted avg       0.92      0.92      0.92      8000
    
    
    ------------Testing -------
    
                  precision    recall  f1-score   support
    
               0       0.93      0.90      0.92      1000
               1       0.91      0.93      0.92      1000
    
        accuracy                           0.92      2000
       macro avg       0.92      0.92      0.92      2000
    weighted avg       0.92      0.92      0.92      2000
    
    **10
    Fold: 2
    Accurracy = 0.922
    
    
    ------------Training -------
    
                  precision    recall  f1-score   support
    
               0       0.92      0.92      0.92      4000
               1       0.92      0.92      0.92      4000
    
        accuracy                           0.92      8000
       macro avg       0.92      0.92      0.92      8000
    weighted avg       0.92      0.92      0.92      8000
    
    
    ------------Testing -------
    
                  precision    recall  f1-score   support
    
               0       0.92      0.92      0.92      1000
               1       0.92      0.92      0.92      1000
    
        accuracy                           0.92      2000
       macro avg       0.92      0.92      0.92      2000
    weighted avg       0.92      0.92      0.92      2000
    
    **10
    Fold: 3
    Accurracy = 0.9175
    
    
    ------------Training -------
    
                  precision    recall  f1-score   support
    
               0       0.92      0.92      0.92      4000
               1       0.92      0.92      0.92      4000
    
        accuracy                           0.92      8000
       macro avg       0.92      0.92      0.92      8000
    weighted avg       0.92      0.92      0.92      8000
    
    
    ------------Testing -------
    
                  precision    recall  f1-score   support
    
               0       0.91      0.93      0.92      1000
               1       0.93      0.91      0.92      1000
    
        accuracy                           0.92      2000
       macro avg       0.92      0.92      0.92      2000
    weighted avg       0.92      0.92      0.92      2000
    


    /home/nitin/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /home/nitin/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /home/nitin/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /home/nitin/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    **10
    Fold: 4
    Accurracy = 0.927
    
    
    ------------Training -------
    
                  precision    recall  f1-score   support
    
               0       0.92      0.92      0.92      4000
               1       0.92      0.92      0.92      4000
    
        accuracy                           0.92      8000
       macro avg       0.92      0.92      0.92      8000
    weighted avg       0.92      0.92      0.92      8000
    
    
    ------------Testing -------
    
                  precision    recall  f1-score   support
    
               0       0.93      0.92      0.93      1000
               1       0.92      0.93      0.93      1000
    
        accuracy                           0.93      2000
       macro avg       0.93      0.93      0.93      2000
    weighted avg       0.93      0.93      0.93      2000
    


    /home/nitin/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)



```python
class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
```
