# Key Machine Learning Algorithms

A summary of important machine learning algorithms, including definitions, use cases, and Python code examples.

---

## 1. Linear Regression

**Definition:**  
Linear Regression is a **supervised learning algorithm** used to predict a continuous outcome. It models the relationship between dependent and independent variables by fitting a straight line through the data points.

**Use Cases:**
- Predicting house prices based on size and location.
- Estimating stock prices based on historical data.
- Forecasting sales based on marketing spend.

**Key Insights:**
- **Simple and interpretable:** The coefficients represent the change in the dependent variable for a one-unit change in the independent variable.
- **Limitation:** Assumes a linear relationship between variables, which might not always be true in real-world data.
- **Overfitting risk:** With too many variables, it might overfit, leading to poor generalization on new data.

**Code Example:**
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 3, 5, 7])

# Model training
model = LinearRegression().fit(X, y)

# Prediction
pred = model.predict([[5]])
print(f"Predicted value for x=5: {pred}")
```

---

## 2. Support Vector Machine (SVM)

**Definition:**  
SVM is a **supervised learning algorithm** used for classification and regression tasks. It finds an optimal hyperplane that maximizes the margin between classes in a high-dimensional space.

**Use Cases:**
- Image classification (e.g., face detection).
- Text classification (e.g., spam detection).
- Handwriting recognition.

**Key Insights:**
- **High-dimensional spaces:** SVMs work well in high-dimensional spaces (e.g., text classification).
- **Choice of kernel:** The kernel trick allows SVM to classify non-linearly separable data by mapping it to a higher-dimensional space.
- **Sensitive to parameters:** SVM requires careful tuning of the regularization and kernel parameters.

**Code Example:**
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model training
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Prediction
accuracy = svm.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
```

---

## 3. Naive Bayes

**Definition:**  
Naive Bayes is a **probabilistic classifier** that applies Bayes’ theorem, assuming that the features are conditionally independent given the class label.

**Use Cases:**
- Spam detection.
- Sentiment analysis.
- Document classification.

**Key Insights:**
- **Fast and scalable:** Naive Bayes can be applied to large datasets, making it efficient in real-time applications.
- **Independence assumption:** Its simplicity is also its limitation; the assumption of feature independence can lead to poor performance with highly correlated data.
- **Works well with small datasets:** Despite its simplicity, Naive Bayes often performs well even with small datasets or noisy data.

**Code Example:**
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model training
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
```

---

## 4. Logistic Regression

**Definition:**  
Logistic Regression is a **supervised learning algorithm** used for binary classification. It uses the logistic (sigmoid) function to model the probability of a binary outcome.

**Use Cases:**
- Disease prediction (e.g., whether a patient has a certain disease).
- Customer churn prediction.
- Marketing response prediction.

**Key Insights:**
- **Interpretable probabilities:** It provides the probability that a data point belongs to a specific class, which can be useful for decision-making.
- **Assumes linear decision boundary:** Logistic Regression assumes that the relationship between features and the log-odds of the outcome is linear.
- **Multi-class extension:** Logistic Regression can be extended to multi-class problems using techniques like one-vs-all (OvA).

**Code Example:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load data
X, y = datasets.load_iris(return_X_y=True)
y = (y == 2)  # Make it binary classification

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
```

---

## 5. K-Nearest Neighbors (KNN)

**Definition:**  
KNN is a **non-parametric classification algorithm** that assigns a class to a data point based on the majority class of its nearest neighbors.

**Use Cases:**
- Image recognition.
- Recommendation systems.
- Anomaly detection.

**Key Insights:**
- **Intuitive and simple:** KNN is easy to understand and implement.
- **Sensitive to distance metric:** The choice of distance metric (e.g., Euclidean distance) can significantly affect performance.
- **Curse of dimensionality:** KNN struggles with high-dimensional data because the concept of "nearest neighbors" becomes less meaningful as the number of features increases.

**Code Example:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model training
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediction
accuracy = knn.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
```

---

## 6. Random Forest

**Definition:**  
Random Forest is an **ensemble learning method** that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.

**Use Cases:**
- Predicting customer behavior (e.g., churn prediction).
- Classifying medical diagnoses.
- Feature selection and importance analysis.

**Key Insights:**
- **Reduces overfitting:** By averaging multiple decision trees, Random Forest is less prone to overfitting compared to a single decision tree.
- **Works well with complex data:** It performs well with large, complex datasets that have many features.
- **Feature importance:** Random Forest can give insights into the importance of different features for prediction.

**Code Example:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model training
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Prediction
accuracy = rf.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")
```

---

### Key Takeaways

1. **Algorithm Selection:** Choosing the right algorithm depends on data type, problem structure, and computation requirements.
2. **Interpretability vs. Complexity:** Simpler models (e.g., Linear Regression, Naive Bayes) are easier to interpret, while complex models (e.g., Random Forest) provide better accuracy but are harder to interpret.
3. **Feature Independence Assumption:** Naive Bayes assumes feature independence, making it fast but less suitable for correlated data.
4. **Hyperparameter Tuning:** Algorithms like SVM, KNN, and Random Forest are sensitive to hyperparameters, which require careful tuning to avoid overfitting.

---
