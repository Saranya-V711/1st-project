import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
# # load the dataset
df = pd.read_csv("bank-additional-full.csv",sep=';')
# print(df.head(10))
# Display basic information about the dataset
print(df.info())
print(df.describe())
print(df.head())

print(df.columns)
# df.columns = df.columns.str.strip()
# print(df.columns)

if 'y' in df.columns:
    print(df['y'].unique())

else:
    print("Target variable 'y' not found in the dataset")
import matplotlib.pyplot as plt
# Analyze the target variable
print(df['y'].value_counts())
sns.countplot(x='y', data=df)
plt.title('Target Variable Distribution')
plt.savefig('target_variable_distribution.png')  # Save as a PNG file
plt.close()

# Analyze numerical features
num_features = df.select_dtypes(include=['int64', 'float64']).columns
df[num_features].hist(bins=20, figsize=(10, 8))
plt.suptitle('Numerical Feature Distributions')
plt.savefig('numerical_feature_distributions.png')  # Save as a PNG file
plt.close()

# Analyze categorical features
cat_features = df.select_dtypes(include=['object']).drop(columns=['y']).columns
for col in cat_features:
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f'{col} Distribution')
    plt.savefig('distributions.png')  # Save as a PNG file
    plt.close()


# 3. Data Preprocessing

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Split features and target
X = df.drop(columns=['y'])
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Define preprocessing
cat_features = X.select_dtypes(include=['object']).columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ]
)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Development

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create a pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# 5. Model Evaluation

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Logistic Regression
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l2']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best AUC Score:", grid_search.best_score_)

import pickle

# Save the model
with open("bank_marketing_model.pkl", "wb") as l:
     pickle.dump(model,l)


