from helper_function import *
import time
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# 1 - Load data and Convert DataFrame to NumPy Array
PATH = "cleaned_data"
X_train, X_train_array = data_loading(PATH + "/" + "X_train_cleaned.csv")
y_train, y_train_array = data_loading(PATH + "/" + "y_train_cleaned.csv")
X_test, X_test_array = data_loading(PATH + "/" + "X_test_cleaned.csv")
y_test, y_test_array = data_loading(PATH + "/" + "y_test_cleaned.csv")

# 2 - Grid Search
hyper_params1 = {"n_estimators": np.arange(5,301),
                 "max_leaves": np.append(np.arange(5,30001), [None])}

hyper_params2 = {"n_estimators": np.arange(5, 301),
                 "max_leaves": np.append(np.arange(5,3001,50), [None])}

hyper_params = {"n_estimators": np.array([5,6]),
                "max_leaves": np.array([5,6,7])}

start = time.time()
np.random.seed(121)
xg_model = XGBRegressor()
xg_model_gs = GridSearchCV(estimator=xg_model,
                           param_grid=hyper_params,
                           cv=5,
                           verbose=3,
                           refit=True,
                           scoring="neg_root_mean_squared_error")
xg_model_gs.fit(X_train_array, y_train_array.ravel())
end = time.time()

print(f"Best tunned hyperparameters: {xg_model_gs.best_params_}\n")
print(f"Mean cross-validated score of best model: {xg_model_gs.best_score_}\n")
xg_model_gs_best = xg_model_gs.best_estimator_
print(f"Model parameters of best model: {xg_model_gs_best.get_params()}\n")

# 3 - Model Evaluation
rmse_train_gs, r2_train_gs = model_evaluation(xg_model_gs_best, X_train_array, y_train_array,
                                              print_dataset="Training Set")
rmse_test_gs, r2_test_gs = model_evaluation(xg_model_gs_best, X_test_array, y_test_array,
                                              print_dataset="Test Set")

tree_feature_importance(xg_model_gs_best, X_train.columns, print_top=20)

plot_predictedVSreal(xg_model_gs_best, X_train_array, y_train_array, print_title="Training Set")
plt.savefig("image/train_predictedVSreal_xg.png")
# plt.show()

plot_predictedVSreal(xg_model_gs_best, X_test_array, y_test_array, print_title="Test Set")
plt.savefig("image/test_predictedVSreal_xg.png")
# plt.show()

# 4 - Prediction for test data
raw_data = pd.read_csv("raw_data.csv", encoding="ISO-8859-1")
plot_prediction(xg_model_gs_best, X_test, y_test, raw_data)
plt.savefig("image/prediction_xg.png")

comp_time = end - start
print(f"Computation Time: {comp_time} sec")