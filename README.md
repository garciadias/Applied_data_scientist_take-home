# **Applied data scientist take-home task Code**

This repository contains the solution for the code test for an Applied data scientist position.
The presentation of the results are found in [this page](https://garcia-dias.notion.site/Applied-data-scientist-take-home-task-da54105abad9431e84bd90bca1cf3921?pvs=4).

To run the code, you need to download the data and add it to the path: `data/vitamin_d_test_results_2022_cleaned.csv`

I am not sharing the data here because I am not the owner, and I want to make this repository public.

To start, you need to install the Python virtual environment, here I am using `conda`:

```bash
conda env create -f conda.yml
```

Then, you need to activate the environment:

```bash
conda activate thriva
```

To run the code, you need to start by cleaning the data:

```bash
python thriva/clean_data.py
```
Before running the analysis, you can take a look at the data by using pandas profiling:

```bash
pandas_profiling --title "Clean data" ./data/vitamin_d_test_results_2022_cleaned.csv reports/report.html
```

Then you can run the analysis:

```bash
python thriva/task_1.py
```

```bash
python thriva/task_2.py
```
If you prefer to run the code in a Jupyter notebook, you can use the following commands:

```bash
python -m ipykernel install --user --name thriva --display-name "thriva"
jupyter-lab
```
This will open a Jupyter lab session in your browser. You can then open the notebooks in the `notebooks` folder. To run the code, you must select the `thriva` kernel.

To run the tests:

```bash
python -m pytest
```

To run the tests with coverage:

```bash
python -m pytest --cov=thriva --cov-report=html
```
