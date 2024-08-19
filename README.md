# Video Game Sales Prediction using Decision Tree Regression

This project implements a machine learning model using a Decision Tree Regressor to predict global video game sales based on various regional sales data and the release year. The dataset used is `vgsales.csv`, which contains information about video game sales across different regions, including North America, Europe, Japan, and others.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Installation

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/7irelo/regressor_model.git
```

Install the necessary Python libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## Dataset

The dataset used in this project is `vgsales.csv`. The dataset includes the following columns:

- **Year**: The release year of the video game.
- **NA_Sales**: Sales in North America (in millions).
- **EU_Sales**: Sales in Europe (in millions).
- **JP_Sales**: Sales in Japan (in millions).
- **Other_Sales**: Sales in other regions (in millions).
- **Global_Sales**: Total global sales (in millions).

Make sure the dataset is placed in the correct path as specified in the script.

## Model

The model is built using the `DecisionTreeRegressor` from scikit-learn. It predicts global video game sales based on the following features:

- `Year`
- `NA_Sales`
- `EU_Sales`
- `JP_Sales`
- `Other_Sales`

## Usage

1. **Preprocessing**: The dataset is preprocessed by dropping any rows with missing values.
2. **Feature Selection**: We select relevant features for training the model.
3. **Model Training**: The data is split into training and testing sets, and the `DecisionTreeRegressor` is trained on the training data.
4. **Model Evaluation**: The performance of the model is evaluated using Mean Squared Error (MSE).
5. **Visualization**: The trained decision tree is visualized using Matplotlib.

To run the code, simply execute the script:

```bash
python main.py
```

## Model Evaluation

The model's performance is evaluated using the Mean Squared Error (MSE) metric, which calculates the average squared difference between the predicted and actual values:

```python
Mean Squared Error: [MSE Value]
```

## Visualization

The trained decision tree is visualized using the `plot_tree` function from scikit-learn, which provides a graphical representation of the tree.

```python
plt.figure(figsize=(20,10))
tree.plot_tree(regressor, filled=True)
plt.show()
```

## Saving and Loading the Model

The trained model is saved using `joblib` for future use:

```python
joblib.dump(regressor, 'regressor_model.pkl')
```

You can load the saved model and use it for predictions:

```python
loaded_model = joblib.load('regressor_model.pkl')
new_predictions = loaded_model.predict(X_test)
```

## Future Enhancements

- Implement more advanced regression techniques such as Random Forest or Gradient Boosting.
- Perform hyperparameter tuning for better model performance.
- Explore feature engineering and scaling to improve the model's accuracy.
- Handle missing data more robustly, possibly using imputation techniques.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

