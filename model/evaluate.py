import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('data/customer_churn.csv')

# Drop CustomerID (not useful)
data = data.drop('CustomerID', axis=1)

# Ensure categorical features are encoded in the same way as training
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['Subscription Type'] = label_encoder.fit_transform(data['Subscription Type'])
data['Contract Length'] = label_encoder.fit_transform(data['Contract Length'])

# Preprocess the dataset
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the saved model
model = joblib.load('model/customer_churn.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')


