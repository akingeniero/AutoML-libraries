import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '../resources/diabetes.csv'
data = pd.read_csv(file_path)

X = data.drop('Class variable', axis=1)
y = data['Class variable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tpot = TPOTClassifier(generations=5,max_time_mins=3,verbosity=2, random_state=42, config_dict='TPOT sparse')
tpot.fit(X_train, y_train)

modelo_actual = tpot.fitted_pipeline_

print("Información sobre el modelo actual:")
print(modelo_actual)

accuracy = tpot.score(X_test, y_test)
print(f'Final Accuracy: {accuracy:.4f}')

y_pred = tpot.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

