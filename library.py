import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce

titanic_variance_based_split = 107

customer_variance_based_split = 113

def dataset_setup(original_table, label_column_name:str, the_transformer, rs, ts=.2):
    features = original_table.drop(label_column_name, axis=1)
    labels = original_table[label_column_name].to_list()
    
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=ts, shuffle=True, random_state=rs, stratify=labels)
    
    x_train_transformed = the_transformer.fit_transform(x_train, y_train)
    x_test_transformed = the_transformer.transform(x_test)
    
    x_train_numpy = x_train_transformed.to_numpy()
    x_test_numpy = x_test_transformed.to_numpy()
    y_train_numpy = np.array(y_train)
    y_test_numpy = np.array(y_test)
    
    return x_train_numpy, x_test_numpy, y_train_numpy, y_test_numpy

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X:pd.core.frame.DataFrame):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns, f'{self.__class__.__name__}.target column "{self.target_column}" not found in dataframe'

    X_ = X.copy()

    dummies = pd.get_dummies(
        X_[self.target_column],
        prefix=self.target_column,
        drop_first=self.drop_first,
        dummy_na=self.dummy_na
    )

    X_.drop(self.target_column, axis=1, inplace=True)

    X_ = pd.concat([X_, dummies], axis=1)

    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X:pd.core.frame.DataFrame):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    column_values = X.columns.to_list()
    column_set = set(column_values)

    keys_values = self.mapping_dict.keys()
    keys_set = set(keys_values)

    keys_not_found = keys_set - column_set
    assert not keys_not_found, f"\n{self.__class__.__name__} these mapping keys do not appear as columns: {keys_not_found}\n"

    X_ = X.copy()
    X_.rename(columns=self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

class CustomMappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #now check to see if all keys are contained in column.
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #do actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    self.target_column = target_column
    self.lower_bound = None
    self.upper_bound = None

  def fit(self, X):
      assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
      assert self.target_column in X.columns, f'{self.__class__.__name__}.fit unknown column "{self.target_column}"'

      mean = X[self.target_column].mean()
      std = X[self.target_column].std()

      self.lower_bound = mean - 3*std
      self.upper_bound = mean + 3*std

      return self

  def transform(self, X):
      assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
      assert self.target_column in X.columns, f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'
      assert self.lower_bound is not None and self.upper_bound is not None, f'{self.__class__.__name__} not fitted yet'

      # Clip values outside the 3-sigma range and reset the index
      X_ = X.copy()
      X_[self.target_column] = X_[self.target_column].clip(self.lower_bound, self.upper_bound)
      return X_

  def fit_transform(self, X):
      self.fit(X)
      return self.transform(X)

class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
        assert fence in ['inner', 'outer'], f'Invalid fence type: {fence}. Use "inner" or "outer".'
        self.target_column = target_column
        self.fence = fence
        self.lower_bound = None
        self.upper_bound = None

  def fit(self, X, y=None):
      assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
      assert self.target_column in X.columns, f'{self.__class__.__name__}.fit unknown column "{self.target_column}"'

      Q1 = X[self.target_column].quantile(0.25)
      Q3 = X[self.target_column].quantile(0.75)
      IQR = Q3 - Q1

      if self.fence == 'inner':
          self.lower_bound = Q1 - 1.5 * IQR
          self.upper_bound = Q3 + 1.5 * IQR
      elif self.fence == 'outer':
          self.lower_bound = Q1 - 3.0 * IQR
          self.upper_bound = Q3 + 3.0 * IQR

      return self

  def transform(self, X):
      assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
      assert self.target_column in X.columns, f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'
      assert self.lower_bound is not None and self.upper_bound is not None, f'{self.__class__.__name__} not fitted yet'

      # Clip values outside the Tukey fence range and reset the index
      X_ = X.copy()
      X_[self.target_column] = X_[self.target_column].clip(self.lower_bound, self.upper_bound)
      return X_

  def fit_transform(self, X, y=None):
      self.fit(X, y)
      return self.transform(X)

class CustomRobustTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.target_column = column
        self.median = None
        self.iqr = None

    def fit(self, X, y=None):
        column_data = X[self.target_column].dropna()
        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)

        self.median = np.median(column_data)
        self.iqr = q3 - q1

        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy[self.target_column] = (X[self.target_column] - self.median) / self.iqr
        return X_copy

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

def find_random_state(features_df, labels, n=200):
  model = KNeighborsClassifier(n_neighbors=5) 
  var = []

  for i in range(1, n):
    train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,random_state=i, stratify=labels)

    model.fit(train_X, train_y)  

    train_pred = model.predict(train_X)     
    test_pred = model.predict(test_X)    

    train_f1 = f1_score(train_y, train_pred)   
    test_f1 = f1_score(test_y, test_pred)     
    f1_ratio = test_f1/train_f1   

    var.append(f1_ratio)
  
  rs_value = sum(var)/len(var)
  return np.array(abs(var - rs_value)).argmin()

titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', ce.TargetEncoder(cols=['Joined'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer('Age')),  #from chapter 5
    ('scale_fare', CustomRobustTransformer('Fare')),  #from chapter 5
    ('imputer', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False))  #from chapter 6
    ], verbose=True)

customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', ce.TargetEncoder(cols=['ISP'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer('Age')), #from 5
    ('scale_time spent', CustomRobustTransformer('Time Spent')), #from 5
    ('impute', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False)),
    ], verbose=True)


def titanic_setup(titanic_table, transformer=titanic_transformer, rs=titanic_variance_based_split, ts=.2):
  return dataset_setup(titanic_table, 'Survived', transformer, rs=rs, ts=ts)

def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
  return dataset_setup(customer_table, 'Rating', transformer, rs=rs, ts=ts)
class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
