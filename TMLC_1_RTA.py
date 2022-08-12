#!/usr/bin/env python
# coding: utf-8


import warnings
from collections import Counter

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold, GridSearchCV

warnings.filterwarnings('ignore')

#  Read CSV
df = pd.read_csv("./Data/RTA Dataset.csv")
df.info()
print(df.columns)
print(df.shape)
df.nunique()
print(df.describe(include=['object']).T)

# In[119]:


df.isna().sum()

# In[120]:


df.duplicated().sum()

# In[121]:


df['Time'] = pd.to_datetime(df['Time'])

# In[122]:


df.info()

# In[123]:


df['hour'] = df['Time'].dt.hour
df['minute'] = df['Time'].dt.minute
df.drop("Time", inplace=True, axis=1)

# In[124]:

print(df.shape)

# In[125]:


print(df.info())

# In[126]:
print(df.columns)

# In[127]:


df.columns = df.columns.str.lower()

# In[128]:


df.describe()

# In[129]:


df.head()

# In[130]:


# df2=df.groupby(['day_of_week']).size()
# df2=df2.sort_values()
# fig= plt.figure(figsize=(10,10))
# plt.title("Number of casualities vs Day of week")
# plt.grid(visible=False)
# colors = ["red" if i == df2.shape[0]-1 else "Grey" for i in range(df2.shape[0])]
# plt.bar(df2.index,df2,color=colors)
# for i, v in enumerate(df2):
#     plt.text( i-.1,v+.1 , str(v), color='Black', fontweight='bold')
#
# plt.show()


# In[131]:


plt.figure(figsize=(20, 100))
plot_number = 1
for col in df.drop(['hour', 'minute', 'road_allignment', 'pedestrian_movement', 'number_of_casualties',
                    'number_of_vehicles_involved'], axis=1):
    df5 = df.groupby(col).size()
    df5 = df5.sort_values()
    ax = plt.subplot(16, 2, plot_number)
    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.grid(visible=False)
    plt.title('distribution of accidents by ' + col)
    indexlist = df5.index.tolist()
    ix = indexlist.index(df5.idxmax())
    colors = ["red" if i == ix else "grey" for i in range(df5.shape[0])]
    plt.barh(df5.index, df5, color=colors)
    for i, v in enumerate(df5):
        plt.text(v, i - 0.05, str(v), color="Black", fontweight='bold')
    plot_number += 1
plt.tight_layout()

# In[132]:


df2 = df.groupby(['hour']).size()
fig = plt.figure(figsize=(12, 12))
plt.title("distribution of accidents by hour")
plt.grid(visible=False)
indexlist = df2.index.tolist()
ix = indexlist.index(df2.idxmax())
colors = ["red" if i == ix else "grey" for i in range(df2.shape[0])]
plt.barh(df2.index, df2, color=colors)
plt.yticks(df2.index)

for i, v in enumerate(df2):
    plt.text(v, i - 0.05, str(v), color="Black", fontweight='bold')

plt.show()

# In[133]:


df2 = df.groupby(['road_allignment']).size()
fig = plt.figure(figsize=(10, 10))
plt.title("distribution of accidents by road_allignment")
plt.grid(visible=False)
indexlist = df2.index.tolist()
ix = indexlist.index(df2.idxmax())
colors = ["red" if i == ix else "grey" for i in range(df2.shape[0])]
plt.barh(df2.index, df2, color=colors)
plt.yticks(df2.index)

for i, v in enumerate(df2):
    plt.text(v, i - 0.05, str(v), color="Black", fontweight='bold')

plt.show()

# In[134]:


df2 = df.groupby(['minute']).size()
fig = plt.figure(figsize=(12, 12))
plt.title("distribution of accidents by minute")
plt.grid(visible=False)
indexlist = df2.index.tolist()
ix = indexlist.index(df2.idxmax())
colors = ["red" if i == ix else "grey" for i in range(df2.shape[0])]
plt.barh(df2.index, df2, color=colors)
plt.yticks(df2.index)

for i, v in enumerate(df2):
    plt.text(v, i - 0.05, str(v), color="Black", fontweight='bold')

plt.show()

# In[135]:


df2 = df.groupby(['pedestrian_movement']).size()
fig = plt.figure(figsize=(12, 12))
plt.title("distribution of accidents by pedestrian_movement")
plt.grid(visible=False)
indexlist = df2.index.tolist()
ix = indexlist.index(df2.idxmax())
colors = ["red" if i == ix else "grey" for i in range(df2.shape[0])]
plt.barh(df2.index, df2, color=colors)
plt.yticks(df2.index)

for i, v in enumerate(df2):
    plt.text(v, i - 0.05, str(v), color="Black", fontweight='bold')

plt.show()

# In[136]:


categorical_columns = []
numerical_columns = []
for col in df.columns.tolist():
    if df[col].dtype == 'object':
        categorical_columns.append(col)
    elif df[col].dtype == 'int64':
        numerical_columns.append(col)
    else:
        pass
print(categorical_columns)
print(40 * '*')
print(numerical_columns)


# In[138]:


df2 = df.groupby(['accident_severity']).size()
fig = plt.figure(figsize=(12, 12))
plt.title("distribution of accidents by accident_severity")
myexplode = [0.2, 0, 0]
plt.pie(df2, labels=list(df2.index),
        autopct='%1.2f%%',
        pctdistance=0.8,
        shadow=False,
        explode=myexplode)
plt.show()

# In[139]:


df.hist(figsize=(8, 8), xrot=45)
plt.show()


# In[140]:


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# In[141]:


missing_values_table_df = missing_values_table(df)
print(missing_values_table_df)

# In[142]:


for col in categorical_columns:
    # display(pd.crosstab(df['accident_severity'], df[col], normalize='index'))
    g = sns.catplot(x=col, kind='count', col='accident_severity', data=df, sharey=False)
    g.set_xticklabels(rotation=60)

# In[143]:


df.drop(columns=['defect_of_vehicle', 'vehicle_driver_relation', 'work_of_casuality', 'fitness_of_casuality',
                 'service_year_of_vehicle'], inplace=True)

# In[144]:


print(df)

# In[145]:


msno.bar(df)

# In[146]:


missing_values_table_df = missing_values_table(df)

for col in missing_values_table_df.index.tolist():
    mode = df[col].mode()[0]
    df[col].fillna(mode, inplace=True)

# In[147]:


msno.bar(df)

# In[148]:
X = df.drop('accident_severity', axis=1)
print(type(X))
X_COLS = X.columns.tolist()
y = df['accident_severity']
enc = OrdinalEncoder()
X = enc.fit_transform(X)
X = pd.DataFrame(X, columns=X_COLS)
print(type(X))

# In[149]:


print(X)

# In[150]:


plt.figure(figsize=(22, 17))
sns.set(font_scale=0.8)
sns.heatmap(X.corr(), annot=True)

# In[151]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=32, shuffle=True)


# In[152]:


def calc_class_percentage(y_train):
    counter = Counter(y_train)
    print("=============================")
    for k, v in counter.items():
        per = 100 * v / len(y_train)
        print(f"Class= {k}, n={v} ({per:.2f}%)")


# In[153]:


calc_class_percentage(y_train)
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)
calc_class_percentage(y_train)

# In[154]:


enc = OrdinalEncoder()
y_train = enc.fit_transform(np.array(y_train).reshape(-1, 1))
y_test = enc.fit_transform(np.array(y_test).reshape(-1, 1))

# In[175]:


def plot_confusion_perf(y_test, y_pred, model):
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1_scr = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy Score: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1_score: {f1_scr}")
    return disp
# In[176]:


model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = model.predict(X_test)
disp = plot_confusion_perf(y_test, y_pred, model)
disp.plot()
plt.grid(False)
plt.show()

# In[177]:


rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
disp = plot_confusion_perf(y_test, y_pred, rf)
disp.plot()
plt.grid(False)
plt.show()

# In[178]:


ab = AdaBoostClassifier(n_estimators=200)
ab.fit(X_train, y_train)
y_pred = ab.predict(X_test)
disp = plot_confusion_perf(y_test, y_pred, ab)
disp.plot()
plt.grid(False)
plt.show()

# In[179]:


gtb = GradientBoostingClassifier(n_estimators=200)
gtb.fit(X_train, y_train)
y_pred = gtb.predict(X_test)
disp = plot_confusion_perf(y_test, y_pred, gtb)
disp.plot()
plt.grid(False)
plt.show()

# In[180]:


dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
disp = plot_confusion_perf(y_test, y_pred, dtc)
disp.plot()
plt.show()

# In[181]:


etc = ExtraTreesClassifier(random_state=0)
etc.fit(X_train, y_train)
y_pred = etc.predict(X_test)
disp = plot_confusion_perf(y_test, y_pred, etc)
disp.plot()
plt.show()

# In[182]:


rf.get_params()

# In[185]:


gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=y_train)

# A parameter grid for ETrees
params = {
    'n_estimators': range(100, 500, 100),
    'ccp_alpha': [0.0, 0.1],
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 11],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [2, 3],
    'class_weight': ['balanced', None]
}

rf_estimator = RandomForestClassifier()

gsearch = GridSearchCV(
    estimator=rf_estimator,
    param_grid=params,
    scoring='f1_weighted',
    n_jobs=-1,
    cv=gkf,
    verbose=3,
)

rf_model = gsearch.fit(X=X_train, y=y_train)
# final_model = gsearch.best_estimator_(gsearch.best_params_, gsearch.best_score_)

# In[188]:


rf_tuned = RandomForestClassifier(ccp_alpha=0.0,
                                  criterion='entropy',
                                  min_samples_split=2,
                                  min_samples_leaf=2,
                                  class_weight=None,
                                  max_depth=11,
                                  n_estimators=400)

rf_tuned.fit(X_train, y_train)
y_pred_tuned = rf_tuned.predict(X_test)
disp = plot_confusion_perf(y_test, y_pred_tuned, rf_tuned)
disp.plot()
plt.grid(False)
plt.show()

# In[ ]:
