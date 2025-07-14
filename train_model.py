
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load Iris dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Features & target
X = df.drop(columns=['species'])
y = df['species'].astype('category').cat.codes  # encode species: 0,1,2

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(multi_class='ovr')
model.fit(X_train_scaled, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Test Accuracy: {acc:.4f}")

# Save model & scaler
joblib.dump(model, 'iris_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("💾 Model and scaler saved to disk.")
