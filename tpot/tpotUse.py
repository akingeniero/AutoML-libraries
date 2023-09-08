import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
""""
pot es una biblioteca de Python que automatiza el proceso de optimización de modelos de aprendizaje automático, incluida la selección de algoritmos y la configuración de hiperparámetros. Para usar Tpot, debes asegurarte de tener instaladas las siguientes dependencias:

numpy: Una biblioteca fundamental para la manipulación de matrices y cálculos numéricos en Python.

scipy: Una biblioteca científica que proporciona funcionalidades adicionales, como optimización numérica.

pandas: Utilizado para la manipulación y análisis de datos.

joblib: Utilizado para la paralelización y el almacenamiento en caché de operaciones costosas.

scikit-learn: Una biblioteca de aprendizaje automático que proporciona algoritmos y herramientas para la construcción de modelos.

deap: El "Distributed Evolutionary Algorithms in Python" es una biblioteca utilizada por Tpot para la optimización evolutiva.

update_checker: Utilizado para comprobar si hay actualizaciones disponibles para Tpot.

stopit: Una biblioteca utilizada para detener la ejecución de código en caso de que se exceda el tiempo de espera.

xgboost (opcional): Tpot puede aprovechar la biblioteca XGBoost para mejorar el rendimiento de los modelos.

lightgbm (opcional): Al igual que XGBoost, Tpot puede utilizar la biblioteca LightGBM para mejorar el rendimiento de los modelos.


"""
file_path = '../resources/diabetes.csv'
data = pd.read_csv(file_path)

X = data.drop('Class variable', axis=1)
y = data['Class variable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tpot = TPOTClassifier(generations=5,max_time_mins=3,verbosity=2, random_state=42, config_dict='TPOT sparse')
tpot.fit(X_train, y_train)

modelo_actual = tpot.fitted_pipeline_

# Mostrar información sobre el modelo actual
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

