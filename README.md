# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare Dataset
Import the required Python libraries and load the food dataset. Select nutritional features (Calories, Total Fat, Saturated Fat, Sugars, Dietary Fiber, Protein) and define the target class.

2. Split and Normalize Data
Divide the dataset into training and testing sets using train_test_split. Apply StandardScaler to normalize the feature values for better SVM performance.

3. Train SVM with Hyperparameter Tuning
Create an SVM classifier and use GridSearchCV to test different hyperparameters such as C, kernel, and gamma to find the best model.

4. Evaluate the Model
Use the optimized model to predict test data and calculate accuracy. Generate a classification report and display a confusion matrix using a heatmap.

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: POOJA U
RegisterNumber:  212225230209

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class'

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm = SVC()

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("Name: POOJA U")
print("Register Number: 212225230209")
print("Best Parameters:", grid_search.best_params_)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Name: POOJA U")
print("Register Number: 212225230209")
print("Accuracy:", accuracy)

print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
*/
```

## Output:
<img width="1075" height="709" alt="Screenshot 2026-03-10 190055" src="https://github.com/user-attachments/assets/43923350-63b3-4bf3-be55-7cc99ff459ba" />
<img width="1392" height="602" alt="Screenshot 2026-03-10 190115" src="https://github.com/user-attachments/assets/ea3a733a-18cb-4656-b421-06e8ce0ed589" />
<img width="1100" height="454" alt="Screenshot 2026-03-10 190127" src="https://github.com/user-attachments/assets/650667a9-148c-4850-badb-2261042966ea" />
<img width="1052" height="765" alt="Screenshot 2026-03-10 190136" src="https://github.com/user-attachments/assets/ea65d1b4-55a8-4365-9109-c4a75da457b3" />


## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
