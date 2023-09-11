import pandas as pd
from sklearn.model_selection import train_test_split
from autokeras import StructuredDataClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '../resources/titanic.csv'
data = pd.read_csv(file_path)

X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = StructuredDataClassifier(max_trials=10, overwrite=True)
clf.fit(X_train, y_train, verbose=2)

accuracy = clf.evaluate(X_test, y_test)[1]
print(f'Final Accuracy: {accuracy:.4f}')

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Survived', 'Survived'], yticklabels=['No Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

