import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series

from thriva.model import ModelSelection, prepare_data


@pytest.fixture(scope="module")
def data():
    X_train, X_test, y_train, y_test = prepare_data()
    yield X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def models():
    X_train, _, y_train, _ = prepare_data()
    models = ModelSelection()
    models.fit(X_train, y_train)
    yield models


def test_prepare_data():
    X_train, X_test, y_train, y_test = prepare_data()
    assert isinstance(X_train, DataFrame)
    assert isinstance(X_test, DataFrame)
    assert isinstance(y_train, Series)
    assert isinstance(y_test, Series)


def test_prepare_data_clean_na():
    X_train, X_test, y_train, y_test = prepare_data(clean_na=True)
    assert isinstance(X_train, DataFrame)
    assert isinstance(X_test, DataFrame)
    assert isinstance(y_train, Series)
    assert isinstance(y_test, Series)
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0
    assert y_train.isna().sum() == 0
    assert y_test.isna().sum() == 0


def test_fit_model(data):
    X_train, X_test, y_train, y_test = data
    models = ModelSelection()
    models.fit(X_train, y_train)
    models.score(X_test, y_test)
    assert isinstance(models.results_, DataFrame)


def test_ModelSelection_has_importace_plot(models):
    # Check that the interpretation is a dictiona
    # Check that the elements in interpretation are matplotlib figures
    importances = models.importance_plot()
    assert isinstance(importances["LGBMClassifier"], Axes)


def test_ModelSelection_roc_plot(models, data):
    # Check that the interpretation is a dictiona
    # Check that the elements in interpretation are matplotlib figures
    X_train, X_test, y_train, y_test = data
    roc = models.roc_plot(X_test, y_test)
    assert isinstance(roc, Figure)
    plt.close()
