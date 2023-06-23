from helper_function import *
import time
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1 - Load data and Convert DataFrame to NumPy Array
PATH = "cleaned_data"
X_train, X_train_array = data_loading(PATH + "/" + "X_train_cleaned.csv")
y_train, y_train_array = data_loading(PATH + "/" + "y_train_cleaned.csv")
X_test, X_test_array = data_loading(PATH + "/" + "X_test_cleaned.csv")
y_test, y_test_array = data_loading(PATH + "/" + "y_test_cleaned.csv")

start = time.time()
np.random.seed(121)
xg_model = XGBRegressor()
xg_model.fit(X_train_array, y_train_array.ravel())
# xg_model.fit(X_train_array, y_train_array)
end = time.time()

# 3 - Model Evaluation
rmse_train, r2_train = model_evaluation(xg_model, X_train_array, y_train_array,
                                              print_dataset="Training Set")

rmse_test, r2_test = model_evaluation(xg_model, X_test_array, y_test_array,
                                            print_dataset="Test Set")

tree_feature_importance(xg_model, X_train.columns, print_top=20)

plot_predictedVSreal(xg_model, X_train_array, y_train_array, print_title="Training Set")
plt.savefig("image/train_predsVSreal_xg_notune.png")
# plt.show()

plot_predictedVSreal(xg_model, X_test_array, y_test_array, print_title="Test Set")
plt.savefig("image/test_predsVSreal_xg_notune.png")
# plt.show()

# 4 - Prediction for testing data
raw_data = pd.read_csv("raw_data.csv", encoding="ISO-8859-1")
plot_prediction(xg_model, X_test, y_test, raw_data)
plt.savefig("image/prediction_xg_notune.png")

comp_time = end - start
print(f"Computation Time: {comp_time} sec => {comp_time/60} min => {comp_time/60/60} hr")