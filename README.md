# House Price Prediction using Machine Learning

This project aims to predict house prices using various machine learning algorithms. By analyzing a dataset of housing attributes, we develop a model that can estimate the price of a house based on its features like location, size, number of rooms, and other relevant characteristics.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Results](#results)
9. [Contributors](#contributors)
10. [License](#license)
11. [Let's Connect](#lets-connect)

## Project Overview

Predicting house prices is a common use case in real estate. By applying machine learning techniques to housing data, this project builds a predictive model to estimate house prices based on attributes like size, number of bedrooms, location, and more.

The goal is to:
- Build a machine learning model to predict house prices.
- Analyze and preprocess the data to handle missing values and outliers.
- Compare different machine learning algorithms for the best performance.
- Evaluate the model using various metrics.

## Dataset

The dataset consists of several features that describe a house and its associated price. It includes:
- **Features**: Number of bedrooms, number of bathrooms, square footage, lot size, year built, and location.
- **Target Variable**: `Price` (continuous variable)

The dataset used for this project is publicly available and can be found in the `data/` directory.

## Data Preprocessing

Data preprocessing includes:
- Handling missing data by imputing values or removing incomplete records.
- Scaling numerical features to ensure all variables are on a similar scale.
- Encoding categorical features using one-hot encoding.
- Removing outliers that might distort model performance.

These steps ensure that the dataset is clean and ready for model training.

## Modeling

Several machine learning models were applied and compared, including:
- Linear Regression
- Random Forest
- Gradient Boosting (XGBoost)
- Support Vector Regressor (SVR)
- K-Nearest Neighbors (KNN)

We used cross-validation and grid search for hyperparameter tuning to achieve optimal results.

## Evaluation

The model performance was evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²)**

These metrics help in assessing the accuracy of the predictions and how well the model generalizes to unseen data.

## Installation

Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/3m0r9/House_Price_Dataset_ML.git
   ```
2. Navigate to the project directory:
   ```bash
   cd House_Price_Dataset_ML
   ```
3. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the model for predictions:

1. Ensure the dataset is in the correct format and placed in the `data/` directory.
2. Run the preprocessing and model training script:
   ```bash
   python train_model.py
   ```
3. For predicting house prices based on new data, use the following:
   ```bash
   python predict.py --input new_house_data.csv
   ```

## Results

The final model achieved good predictive accuracy on the test set with the following results:
- **Mean Absolute Error (MAE)**: 15,000
- **Root Mean Squared Error (RMSE)**: 25,000
- **R-squared (R²)**: 0.85

These results indicate that the model is able to make fairly accurate predictions, although improvements can still be made by further tuning or using more complex algorithms.

## Contributors

- **Imran Abu Libda** - [3m0r9](https://github.com/3m0r9)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Let's Connect

- **GitHub** - [3m0r9](https://github.com/3m0r9)
- **LinkedIn** - [Imran Abu Libda](https://www.linkedin.com/in/imran-abu-libda/)
- **Email** - [imranabulibda@gmail.com](mailto:imranabulibda@gmail.com)
- **Medium** - [Imran Abu Libda](https://medium.com/@imranabulibda_23845)
