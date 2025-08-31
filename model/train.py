import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('data/customer_churn.csv')

# Drop CustomerID (not useful for prediction)
data = data.drop('CustomerID', axis=1)

# Encode categorical variables
label_encoder = LabelEncoder()
data['Subscription Type'] = label_encoder.fit_transform(data['Subscription Type'])
data['Contract Length'] = label_encoder.fit_transform(data['Contract Length'])

# Define features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model/customer_churn.pkl')



