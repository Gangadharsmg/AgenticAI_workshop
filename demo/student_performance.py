
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample student dataset
data = {
    "attendance": [90, 85, 60, 45, 95, 70, 50, 80, 65, 92,
                   55, 88, 75, 40, 98, 68, 82, 58, 93, 72],
    "study_hours": [4, 3, 2, 1, 5, 2, 1, 3, 2, 4,
                    1, 4, 3, 1, 5, 2, 3, 1, 5, 2],
    "previous_score": [85, 78, 55, 40, 92, 65, 45, 75, 58, 88,
                       50, 82, 70, 35, 95, 60, 80, 48, 90, 68],
    "assignments": [95, 90, 60, 40, 100, 70, 45, 85, 65, 95,
                    50, 90, 80, 30, 100, 65, 88, 55, 98, 75],
    "result": [
        "Pass", "Pass", "Fail", "Fail", "Pass",
        "Pass", "Fail", "Pass", "Fail", "Pass",
        "Fail", "Pass", "Pass", "Fail", "Pass",
        "Fail", "Pass", "Fail", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

# Separate input features and target
X = df.drop("result", axis=1)
y = df["result"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train the ML model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))

# Predict for a new student
new_student = pd.DataFrame([{
    "attendance": 85,
    "study_hours": 4,
    "previous_score": 78,
    "assignments": 90
}])

result = model.predict(new_student)

print("Predicted Student Result:", result[0])
