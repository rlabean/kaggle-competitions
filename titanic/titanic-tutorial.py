# Load in our packages

import numpy as np
import pandas as pd


# Load in the training dataset

train_data = pd.read_csv("~/kaggle-competitions/titanic/train.csv")
print(train_data.head())


# Load in the test dataset 

test_data = pd.read_csv("~/kaggle-competitions/titanic/test.csv")
print(test_data.head)


# Finding the rate of women who survived the Titanic

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


# Finding the rate of men who survived the Titanic

men= train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men= sum(men)/len(men)

print("% of men who survived:", rate_men)


# Build a random forest model to estimate who died

# Import packages

from sklearn.ensemble import RandomForestClassifier

# Assign variables

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Build our model

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# Output

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
        'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfullysuccessfully succesfully  saved")


