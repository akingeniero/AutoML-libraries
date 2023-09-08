from pycaret.datasets import get_data
from pycaret.classification import *

data = get_data('titanic')
exp_clf = setup(data, target='Survived', session_id=123)

best_model = compare_models(budget_time=3, fold=30)
trained_model = finalize_model(best_model)

evaluate_model(trained_model)
interpret_model(trained_model)

plot_model(trained_model, plot='confusion_matrix')
plot_model(trained_model, plot='auc')
plot_model(trained_model, plot='feature')
plot_model(trained_model, plot='learning')
plot_model(trained_model, plot='calibration')
plot_model(trained_model, plot='error')

# predictions = predict_model(trained_model, data=new_data)
