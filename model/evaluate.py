# Load the dataset
data = pd.read_csv('data/customer_churn')

# Preprocess the dataset
X = data.drop('churn', axis=1)
y = data['churn']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, andom_state=42)

# Load the saved model
model = joblib.load('model/customer_churn.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

