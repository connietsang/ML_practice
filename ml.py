import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import seaborn as sns

training_file_path = '../input/titanic/train.csv'
test_file_path = '../input/titanic/test.csv'

train = pd.read_csv(training_file_path, index_col='PassengerId')
test = pd.read_csv(test_file_path, index_col='PassengerId')

y = train['Survived']

# Tests
# sns.countplot(x='Survived', hue='Sex', data=train)
# sns.countplot(x='Survived', hue='Pclass', data=train)
# sns.countplot(x='Survived', hue='Age', data=train)
# sns.catplot(x='Survived', y='Age', kind='swarm',data=train)

# sns.countplot(x='Survived',hue='Embarked',data=train)
sns.catplot(x='Survived', y='SibSp', kind='strip', data=train)

features = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Age', 'Fare']
training_data = train.loc[:, features]
# replace categorical features with numerical representations
training_data['Sex'] = training_data['Sex'].replace('male', 0)
training_data['Sex'] = training_data['Sex'].replace('female', 1)

median_age = training_data['Age'].median()

training_data['Age'] = training_data['Age'].fillna(median_age)

# # check that there are no null values in dataset
# print(training_data)
# training_data.isnull().sum()

train_X, validation_X, train_y, validation_y = train_test_split(training_data, y,
                                                                train_size=0.6, test_size=0.4,
                                                                random_state=1)

model = RandomForestClassifier()
model.fit(train_X, train_y)
preds = model.predict(validation_X)

score = mean_absolute_error(validation_y, preds)

print('MAE:', score)

testing_data = test.loc[:, features]
testing_data['Sex'] = testing_data['Sex'].replace('male', 0)
testing_data['Sex'] = testing_data['Sex'].replace('female', 1)
median_age = testing_data['Age'].median()
median_fare = testing_data['Fare'].median()
testing_data['Age'] = testing_data['Age'].fillna(median_age)
testing_data['Fare'] = testing_data['Fare'].fillna(median_fare)

testing_data.isnull().sum()
preds_test = model.predict(testing_data)
output = pd.DataFrame({'PassengerId': test.index,
                       'Survived': preds_test})
output.to_csv('submission.csv', index=False)
