import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from thriva.clean_data import load_clean_data
from thriva.model import ModelSelection, prepare_data
from thriva.plots import (
    plot_categorical_percentages,
    plot_hist_percentages,
    plot_roc_curve,
)


@pytest.fixture(scope="module")
def vitamin_d():
    yield load_clean_data()


@pytest.fixture(scope="module")
def models_X_test_y_test():
    X_train, X_test, y_train, y_test = prepare_data()
    models = ModelSelection()
    models.fit(X_train, y_train)
    yield models, X_test, y_test


def test_plot_categorical_percentages(vitamin_d):
    target = "Vitamin D Supplement"
    findings_cols_0 = ["Fatigued Rating", "Diet Rating", "Exercise Rating"]
    fig = plot_categorical_percentages(vitamin_d, findings_cols_0, target, bottom=0.0)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_hist_percentages(vitamin_d):
    target = "Low Vitamin D"
    col_list = ["Fatigued Rating", "Diet Rating", "Exercise Rating"]
    vitamin_d[target] = vitamin_d["Vitamin D Level"] < 50
    vitamin_d[target] = vitamin_d[target].map({True: "Yes", False: "No"})
    vitamin_d[target] = vitamin_d[target].astype("category")
    fig = plot_hist_percentages(vitamin_d, col_list, target_col="Low Vitamin D")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_roc_curve(models_X_test_y_test):
    models, X_test, y_test = models_X_test_y_test
    fig = plot_roc_curve(models, X_test, y_test)
    assert isinstance(fig, Figure)
