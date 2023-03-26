from dataclasses import dataclass, field

import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
from matplotlib.figure import Figure
from numpy.random import seed as default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from thriva.clean_data import load_clean_data
from thriva.plots import plot_roc_curve

default_rng(25032023)

classification_metrics = {
    "accuracy": accuracy_score,
    "auc": roc_auc_score,
    "precision": precision_score,
    "recall": recall_score,
}


def prepare_data(clean_na=False):
    """Prepare data for model training.

    Returns
    -------
    X_train : pandas.DataFrame
        Training data.

    X_test : pandas.DataFrame
        Testing data.

    y_train : pandas.DataFrame
        Training labels.

    y_test : pandas.DataFrame
        Testing labels.

    """
    vitamin_d = load_clean_data()
    exclude_columns = ["Vitamin D Level", "Tests Completed Month"]
    if clean_na:
        vitamin_d = vitamin_d.dropna()
    X = vitamin_d.drop(columns=exclude_columns, axis=1)
    y = vitamin_d["Vitamin D Level"].lt(75)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


@dataclass
class ModelSelection:
    """Class for selecting the best model from a list of models.

    Parameters
    ----------
    models : list, optional, default=[LinearRegression(), RandomForestRegressor(), lgb.LGBMRegressor()]
        List of models to be trained and evaluated.

    """

    models: list = field(
        default_factory=lambda: [
            LGBMClassifier(),
        ]
    )

    def fit(self, X, y) -> "ModelSelection":
        """Fit models to data and create a list of trained models, `trained_models`.

        Parameters
        ----------
        X : pandas.DataFrame
            Training data.
        y : pandas.DataFrame
            Training labels.

        Returns
        -------
        self : ModelSelection
            Trained models.
        """
        for model in self.models:
            model.fit(X, y)
        return self

    def score(self, X_test, y_true) -> pd.DataFrame:
        """Score the model on the test data.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Testing data.
        y_true : pandas.DataFrame
            Testing labels.

        Returns
        -------
        results_ : pd.DataFrame
            Classification report for each model.
        """
        self.results_ = pd.DataFrame()
        for model in self.models:
            model_name = model.__class__.__name__
            report = {}
            for metric_name, metric in classification_metrics.items():
                if metric_name == "auc":
                    report[metric_name] = metric(
                        y_true, model.predict_proba(X_test)[:, 1]
                    )
                else:
                    report[metric_name] = metric(y_true, model.predict(X_test))
            report = pd.DataFrame(report, index=[model_name], columns=report.keys())
            self.results_ = pd.concat([self.results_, report])
        self.results_ = self.results_.T.round(3)
        return self.results_

    def importance_plot(self) -> Figure:
        """Plot the importance of each feature in the model."""
        self.importance_plots_ = {}
        for model in self.models:
            model_name = model.__class__.__name__
            if model_name == "LGBMClassifier":
                self.importance_plots_[model_name] = plot_importance(model)
        return self.importance_plots_

    def roc_plot(self, X_test, y_test) -> Figure:
        """Plot the ROC curve for each model."""
        self.roc_plot_ = plot_roc_curve(self, X_test, y_test)
        return self.roc_plot_


def encoded_logistic_regression():
    """Return a LogisticRegression model with OneHotEncoder."""
    pipeline = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("logistic", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.__class__.__name__ = "LogisticRegression"
    return pipeline
