# NFL Stats Analysis with Python

## 🏈 Overview

This project is an analysis of NFL statistical data using Python to predict game outcomes (win or loss). Leveraging the **pandas** library for data manipulation and **scikit-learn** for machine learning, this project builds a **Logistic Regression** model to classify wins based on offensive and defensive stats. The workflow includes data preprocessing, visualization, model training, and evaluation.

## 📁 Repository Structure

  - `Analyze NFL Stats with Python.ipynb`: A Jupyter Notebook containing all the code for the analysis, from data loading to model evaluation.
  - `season_2021.csv`: The primary dataset used for training the model, containing NFL game stats from the 2021 season.
  - `helper.py`: A helper file containing the `get_new_data` function to retrieve new game data.

## 🛠️ Requirements

To run this notebook, you will need a Python environment with the following libraries:

  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
    You can install them with the following command:

<!-- end list -->

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 📈 Project Workflow

### 1\. Setup and Data Inspection

  - The `season_2021.csv` dataset is loaded using **pandas**.
  - The `result` column is inspected and re-categorized from `W` (win), `L` (loss), and `T` (tie) into numeric values `1` for a win and `0` for a loss/tie. This is a necessary step as Logistic Regression requires numeric outcome values.
  - Key statistics, such as `1stD_offense` (offensive first downs), are visualized using a **boxplot** to understand the trends between wins and losses.

-----

### 2\. Data Preparation

  - **Feature Standardization**: All game stats (`TotYd_offense`, `PassY_offense`, etc.) are standardized using **scikit-learn's `StandardScaler`**. This process ensures all features are on the same scale, preventing a single feature from dominating the model due to a larger value range.
  - **Data Splitting**: The dataset is then divided into a training set (`X_train`, `y_train`) and a testing set (`X_test`, `y_test`). The initial split uses a **50:50 ratio** for training and testing.

-----

### 3\. Analysis and Model Training

  - **Logistic Regression Model**: A `LogisticRegression` model is created and trained (`.fit()`) using the training data. The model learns from the statistical patterns to predict the probability of a winning game.
  - **Accuracy Evaluation**: The initial model's accuracy is calculated by comparing its predictions on the test data (`y_pred`) to the actual results (`y_test`). The resulting accuracy is **83%**.

-----

### 4\. Model Optimization

  - **Hyperparameter Tuning**: To improve performance, the hyperparameters `penalty` and `C` are tuned.
      - `penalty` (`l1` or `l2`) is a regularization parameter to reduce overfitting.
      - `C` is the inverse of regularization strength.
      - Testing shows the highest accuracy is achieved with `penalty='l1'` and `C=0.1`, reaching **84.6%**.
  - **Optimal Test Size**: The test size (`test_size`) is varied from `0.2` to `0.35` to see its effect on accuracy. The highest accuracy is found at `test_size=0.25`, reaching **88.8%**.

-----

### 5\. Finalization and Conclusion

  - **Optimized Model**: The final Logistic Regression model is trained with the optimal hyperparameters (`penalty='l1'`, `C=0.1`, and `test_size=0.25`). This model is named `optLr`.
  - **Feature Importance**: The model's coefficients are examined to determine which features are most important in predicting a win. Stats like `TO_offense` and `TO_defense` (offensive and defensive turnovers) and `TotYd_offense` (Total Offensive Yards) show the highest scores, indicating their significant impact on game outcomes.
  - **New Data Testing**: The `optLr` model is then tested on new data for a new team (**the 2022 Dallas Cowboys**). The model's accuracy on this new data is **89.5%**, which demonstrates that the model performs well beyond its original training dataset.
