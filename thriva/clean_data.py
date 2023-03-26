import pandas as pd


def age_clean(age: int) -> int:
    """If age is wrongly entered as year of birth, convert to age.
    If age is greater than 100, return 100.

    Parameters
    ----------
    age : int
        Age of the user

    Returns
    -------
    age : int
        Age of the user corrected if necessary
    """
    age = int(age)
    if age > 1900:
        return 2022 - age
    elif age > 100:
        return 100
    else:
        return age


def clean_vitamin_d(df):
    df["Users Age"] = df["Users Age"].apply(age_clean)
    df["Users Bmi"] = df["Users Bmi"].mask(~df["Users Bmi"].between(10, 60))
    df["Users Inactive Time"] = df["Users Inactive Time"].apply(
        lambda x: x.replace(" hours", "")
    )
    rename_dict = {col: col.replace("Users ", "") for col in df.columns}
    rename_dict["Users Bmi"] = "BMI"
    rename_dict["Users Vitamin D Supplement (Yes / No)"] = "Vitamin D Supplement"
    rename_dict["Analyte Results Avg Numeric Result"] = "Vitamin D Level"
    df = df.rename(columns=rename_dict)
    return df


def load_clean_data():
    dtypes = {
        "Active Days Walking": "float64",
        "Diet Rating": "float64",
        "Exercise Rating": "float64",
        "Inactive Time": "category",
        "BMI": "float64",
        "Fatigued Rating": "float64",
        "Sleep Hours": "category",
        "Stressed Rating": "category",
        "Main Goal": "category",
        "Age": "int64",
        "Sex": "category",
        "Vitamin D Supplement": "category",
        "Vitamin D Level": "float64",
    }
    date_cols = ["Tests Completed Month"]

    vitamin_d = pd.read_csv(
        "data/vitamin_d_test_results_2022_cleaned.csv",
        dtype=dtypes,
        parse_dates=date_cols,
    )
    return vitamin_d


if __name__ == "__main__":
    dtypes = {
        "Users Active Days Walking": "float64",
        "Users Diet Rating": "float64",
        "Users Exercise Rating": "float64",
        "Users Inactive Time": "category",
        "Users Bmi": "float64",
        "Users Fatigued Rating": "category",
        "Users Sleep Hours": "category",
        "Users Stressed Rating": "category",
        "Users Main Goal": "category",
        "Users Age": "int64",
        "Sex": "category",
        "Users Vitamin D Supplement (Yes / No)": "category",
        "Analyte Results Avg Numeric Result": "float64",
    }
    date_cols = ["Tests Completed Month"]
    vitamin_d = pd.read_csv(
        "data/vitamin_d_test_results_2022.csv", dtype=dtypes, parse_dates=date_cols
    )
    vitamin_d = clean_vitamin_d(vitamin_d)
    vitamin_d.to_csv("data/vitamin_d_test_results_2022_cleaned.csv", index=False)
