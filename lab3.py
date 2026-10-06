import pandas as r100
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

Breast_cancer = load_breast_cancer()

x = r100.DataFrame(Breast_cancer.data, columns=Breast_cancer.feature_names)
y = Breast_cancer.target

#print(x.head())

missing_values = x.isnull().sum()
#print (missing_values)

Class_distributuin = r100.Series(y).value_counts()
#print(Class_distributuin)

plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, legend=False, palette="Set2")

plt.xticks(ticks=[0, 1], labels=Breast_cancer.target_names)
plt.title("Class Distribution (Malignant vs. Benign)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
standard_scaler = StandardScaler()
x_train_scaled = standard_scaler.fit_transform(x_train)
x_test_scaled = standard_scaler.transform(x_test)

model = LogisticRegression(random_state=42)

model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)