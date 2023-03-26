"""
Explore how accurately you can predict whether someone will have an optimal vitamin D
result before testing their levels, from other information about them in the dataset.
What are your conclusions? Do not spend too much time cleaning and transforming data,
feel free to write a few notes about what more you would do given time/the limitations
of your exploration.
"""
# %%
# %cd ..
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier

# %%
from thriva.model import ModelSelection, encoded_logistic_regression, prepare_data

# %%
X_train, X_test, y_train, y_test = prepare_data()
model_list = [encoded_logistic_regression(), LGBMClassifier(), DummyClassifier()]
# %%
models = ModelSelection(models=model_list)
models.fit(X_train, y_train)
models.score(X_test, y_test)
# %%
# Clean NaN values from the training data
X_train_clean, X_test_clean, y_train_clean, y_test_clean = prepare_data(clean_na=True)
models_clean = ModelSelection(models=model_list)
models_clean.fit(X_train_clean, y_train_clean)
models_clean.score(X_test_clean, y_test_clean)
# %%
models.roc_plot(X_test, y_test)
plt.savefig("reports/roc_curve.png", dpi=300)
plt.close()
models_clean.roc_plot(X_test_clean, y_test_clean)
plt.savefig("reports/roc_curve_clean.png", dpi=300)
plt.show()
# %%
importances = models_clean.importance_plot()
importances["LGBMClassifier"]
plt.savefig("LGBM_feature_importance.png", dpi=300)
