# Proj-1-Stock-Price-Prediction

Apple:

ARIMA Time Series Model for Apple Stock Price Forecasting
This Python script implements an ARIMA (AutoRegressive Integrated Moving Average) time series model to forecast the closing price of Apple stock. The script uses historical stock price data of Apple from the year 2005 to 2020.

Libraries Used
The script makes use of the following Python libraries:

numpy: For numerical computations.
pandas: For data manipulation and handling time series data.
seaborn and matplotlib: For data visualization.
statsmodels: For time series analysis and ARIMA modeling.
pmdarima: For automatic ARIMA order selection.
sklearn.metrics: For computing metrics such as mean squared error and mean absolute error.
Data Preprocessing
The script reads the historical stock price data of Apple from a pickle file and preprocesses it to select the relevant date range (2005 to 2020). The closing price column is used for analysis and forecasting.

Stationarity Check
The script checks the stationarity of the time series data by plotting the rolling mean and standard deviation. It also performs the Augmented Dickey Fuller (ADF) test to determine the stationarity.

Differencing
The time series data is differenced to make it stationary using the first difference method, logarithm method, rolling mean method, exponential decay method, and time shift method. The ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots are used to identify the appropriate differencing parameters.

Auto ARIMA
The script uses the auto_arima function from the pmdarima library to automatically select the best parameters (p, d, q) for the ARIMA model based on the ADF test results.

Model Fitting
Four ARIMA models are fitted to the training data with different forecasting horizons (1 day, 7 days, 30 days, and 90 days).

Model Evaluation
The accuracy of each ARIMA model is evaluated using metrics such as mean squared error (MSE), mean absolute error (MAE), root mean squared error (RMSE), and mean absolute percentage error (MAPE). The predictions are compared with the actual values from the test data.

Forecasting
The script provides forecasts for each ARIMA model for the respective forecasting horizon (1 day, 7 days, 30 days, and 90 days). The forecasts include point estimates and confidence intervals.

Conclusion
The ARIMA model shows promising results in forecasting the closing price of Apple stock. However, further evaluation and optimization may be required to improve the accuracy of the predictions. The forecasting horizon can also be adjusted based on specific requirements.



Boeing:

ARIMA Time Series Model
This repository contains code for implementing an ARIMA (AutoRegressive Integrated Moving Average) time series model to forecast the closing price of Boeing stock. The ARIMA model is a popular technique for time series forecasting, and it combines autoregression, differencing, and moving average components to capture the underlying patterns in the data.

Library Imports
This code requires the following libraries to be installed:

numpy
pandas
seaborn
matplotlib
statsmodels
scikit-learn
pmdarima
Data Preparation
Before running the ARIMA model, make sure you have the 'boeing_prep4.pkl' dataset. This dataset contains the necessary data for forecasting the closing price of Boeing stock. The dataset should have a 'date' column as the index and a 'close' column representing the closing price.

Data Exploration and Stationarity Check
The code starts by plotting the closing price of Boeing stock and then analyzes the stationarity of the time series. It uses rolling mean and standard deviation plots, as well as the Augmented Dickey-Fuller test, to check for stationarity. If the time series is non-stationary, it performs differencing to make it stationary.

Autocorrelation and Partial Autocorrelation Plots
The code also includes a function to plot the time series, autocorrelation function (ACF), and partial autocorrelation function (PACF) to help determine the p, d, and q parameters for the ARIMA model.

Differencing Methods
Various differencing methods are explored to achieve stationarity. The methods include:

First Difference
Logarithm Method
Subtract Rolling Mean Method
Exponential Decay Method
Time Shift Method
Auto ARIMA Model Selection
The code uses the auto_arima function from the pmdarima library to automatically determine the best parameters (p, d, q) for the ARIMA model based on the augmented Dickey-Fuller test results.

ARIMA Model Fitting and Forecasting
The ARIMA models with the selected parameters are fitted to the training data. The models' performances are evaluated based on Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

Forecasting
The ARIMA models are used to forecast the closing prices of Boeing stock for different time intervals (1 day, 7 days, 30 days, and 90 days). The forecasts are plotted along with the confidence intervals to visualize the predictions.

Evaluation
The code calculates and displays the accuracy metrics (MSE, MAE, RMSE, MAPE) for each ARIMA model.


Nike:

Introduction
This repository contains code for building an ARIMA (AutoRegressive Integrated Moving Average) time series model for forecasting the closing price of Nike stock. The ARIMA model is a popular method for time series forecasting that takes into account autoregressive, differencing, and moving average components to make predictions.

Library Imports
This project requires several Python libraries to be imported. The necessary libraries are as follows:

numpy: Numerical computing library
pandas: Data manipulation library
seaborn: Data visualization library
matplotlib: Plotting library
statsmodels: Time series analysis library
pmdarima: Auto ARIMA library for automatic order selection
sklearn.metrics: Library for calculating evaluation metrics
math: Mathematical functions
warnings: Library for handling warnings

Data Preparation
The data is assumed to be available as a pickle file called nike_prep4.pkl. The dataset should contain a 'date' column and a 'close' column representing the closing price of Nike stock on different dates.

Exploratory Data Analysis
The code starts with importing the data and plotting the closing prices of Nike stock over time using Seaborn and Matplotlib.
Rolling mean and standard deviation of the closing price are plotted to determine if the time series is stationary.
Augmented Dickey Fuller Test is used to check for stationarity.

Stationarizing the Time Series
Different methods for stationarizing the time series are employed:

First Difference: Taking the difference between consecutive closing prices.

Logarithm: Applying the logarithm transformation to the closing prices.

Subtracting Rolling Mean: Subtracting the rolling mean from the closing prices.

Exponential Decay: Subtracting the exponentially weighted moving average from the closing prices.

Time Shift: Taking the difference between the closing prices and their previous values.

Autocorrelation and Partial Autocorrelation Plots
Autocorrelation and partial autocorrelation plots are used to determine the orders (p and q) of the ARIMA model.

Model Training and Forecasting
The ARIMA model is trained on the prepared data using the selected differencing parameters (d) and orders (p and q).
The trained model is used to make predictions for different forecast horizons (1, 7, 30, and 90 days).
The model's performance is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

Evaluation Results
The evaluation metrics for each forecast horizon (1, 7, 30, and 90 days) are printed.

Conclusion
The ARIMA model provides a framework for time series forecasting of the closing price of Nike stock. The model's performance can be evaluated using different evaluation metrics, and it can be fine-tuned further to improve its forecasting accuracy.
