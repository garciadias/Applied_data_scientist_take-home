import pytest

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import seed as default_rng
from pandas import DataFrame
from pytest import fixture
from sklearn.datasets import load_diabetes
from sklearn.utils.validation import check_is_fitted
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from shap_clustering.metrics import REGRESSION_METRICS, get_model_type
from shap_clustering.model import ModelSelection

# set up fixed global random seed for reproducibility
default_rng(14032023)


@fixture(scope="module")
def df():
    return load_diabetes(as_frame=True).frame.sample(100)


@fixture(scope="module")
def selection(df):
    target = "target"
    selection = ModelSelection()
    selection.fit(df, target)
    return selection


def test_ModelSelection_is_trained(selection):
    assert selection.models is not None
    assert len(selection.models) == 3
    assert selection.models[0].__class__.__name__ == "LinearRegression"
    assert all([check_is_fitted(model) is None for model in selection.models])
    assert isinstance(selection.metrics, DataFrame)
    assert selection.metrics.shape == (len(selection.models), len(REGRESSION_METRICS))


def test_ModelSelection_has_pdp(selection, df):
    # Check that the interpretation is a dictionary
    # Check that the elements in interpretation are matplotlib figures
    first_var = df.columns[0]
    explanation = selection.explain()
    assert isinstance(explanation.pdp_, dict)
    assert isinstance(explanation.pdp_[first_var]["LinearRegression"], Axes)


def test_ModelSelection_has_shap_values(selection):
    # Check that the interpretation is a dictionary
    # Check that the elements in interpretation are matplotlib figures
    explanation = selection.explain()
    assert isinstance(explanation.shap_values_, dict)
    assert isinstance(explanation.shap_values_["LinearRegression"], ndarray)
    assert isinstance(explanation.shap_values_["LGBMRegressor"], ndarray)
    assert isinstance(explanation.shap_values_["ElasticNet"], ndarray)


def test_ModelSelection_has_shap_importance(selection):
    # Check that the interpretation is a dictionary
    # Check that the elements in interpretation are matplotlib figures
    explanation = selection.explain()
    assert isinstance(explanation.shap_importance_, DataFrame)
    assert explanation.shap_importance_.shape == (
        selection.X_train.shape[1],
        len(selection.models),
    )


def test_ModelSelection_has_importace_plot(selection):
    # Check that the interpretation is a dictionary
    # Check that the elements in interpretation are matplotlib figures
    explanation = selection.explain()
    assert isinstance(explanation.importance_plot(), Figure)


def test_model_type(selection):
    for model in selection.models:
        assert isinstance(get_model_type(model), str)
        assert get_model_type(model) == "regressor"


def test_classification_metrics(df):
    target = "target"
    models = ModelSelection(models=[SVC()])
    models.fit(df, target)


def test_clustering_metrics(df):
    target = "target"
    models = ModelSelection(models=[KMeans()])

    with pytest.raises(ValueError) as excinfo:
        models.fit(df, target)
    