# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# # Load the data
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
# data = pd.read_csv(url, header=None, delimiter=', ', engine='python')

# # Rename the columns
# columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
#            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
#            'hours-per-week', 'native-country', 'income']
# data.columns = columns

# # Convert categorical variables using LabelEncoder
# label_encoder = LabelEncoder()
# data['workclass'] = label_encoder.fit_transform(data['workclass'])
# data['education'] = label_encoder.fit_transform(data['education'])
# data['marital-status'] = label_encoder.fit_transform(data['marital-status'])
# data['occupation'] = label_encoder.fit_transform(data['occupation'])
# data['relationship'] = label_encoder.fit_transform(data['relationship'])
# data['race'] = label_encoder.fit_transform(data['race'])
# data['sex'] = label_encoder.fit_transform(data['sex'])
# data['native-country'] = label_encoder.fit_transform(data['native-country'])
# data['income'] = label_encoder.fit_transform(data['income'])

# # Split the data into training and testing sets
# X = data.drop('income', axis=1)
# y = data['income']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model's performance
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy score: {accuracy}")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE

# Load the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
data = pd.read_csv(url, header=None, delimiter=', ', engine='python')
# Rename the columns
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country', 'income']
data.columns = columns

# Convert categorical variables using LabelEncoder
label_encoder = LabelEncoder()
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])
data['income'] = label_encoder.fit_transform(data['income'])

# Select a subset of features using RFE
X = data.drop('income', axis=1)
y = data['income']
selector = RFE(LogisticRegression(max_iter=10000), n_features_to_select=10)
X_new = selector.fit_transform(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Train the logistic regression model with regularization
model = LogisticRegression(max_iter=10000, C=0.1)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")
