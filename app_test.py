import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
import base64
import datetime
import os
from PIL import Image
from datetime import date
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import os
import numpy as np
import math
from matplotlib import pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, explained_variance_score, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from pickle import TRUE
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
# Class thống kê mô tả
class DESCRIPTIVE_STATISTICS:
    def __init__(self, df):
        self.df = df
        self.closedf = self.df[['date', 'close']].copy()
        self.close_stock_2023 = self.closedf[self.closedf['date'] > '2023-01-01'].copy()

    def analyze_monthly_average(self):
        monthvise = self.df.groupby(self.df['date'].dt.strftime('%B'))[['open', 'close']].mean()
        new_order = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
        monthvise = monthvise.reindex(new_order, axis=0)
        return monthvise

    def plot_monthly_average_bar_chart(self, data):
        fig = px.bar(data, x=data.index, y=['open', 'close'], labels={'value': 'Price', 'variable': 'Metric'},
                    title='Trung bình giá giao dịch theo tháng')
        fig.update_layout(
            autosize=False,
            width=12*80,
            height=6*80,
            barmode='group',
            title_x=0.4
        )
        return fig

    def plot_quarterly_average_bar_chart(self):
        fig = px.bar(self.df.groupby(self.df['date'].dt.quarter)[['open', 'close']].mean().reset_index(),
                     x='date', y=['open', 'close'], labels={'value': 'Price', 'variable': 'Metric'},
                     title='Giá trung bình theo quý')
        fig.update_layout(
            autosize=False,
            width=12*80,
            height=6*80,
            barmode='group',
            title_x=0.4
        )
        return fig
    def plot_yearly_average_line_chart(self):
        fig = px.line(self.df.groupby(self.df['date'].dt.year)[['open', 'close', 'High', 'low']].mean().reset_index(),
                      x='date', y=['open', 'close', 'High', 'low'], labels={'value': 'Price', 'variable': 'Metric'},
                      title='Giá trung bình theo năm')
        fig.update_layout(
            autosize=False,
            width=12*80,
            height=6*80,
            title_x=0.4
        )
        return fig

    def relationship_correlation(self):
        columns_to_corr = ['close', 'open', 'High', 'low', 'volume']
        corr = self.df[columns_to_corr].corr()

        fig_heatmap = ff.create_annotated_heatmap(z=corr.values, x=columns_to_corr, y=columns_to_corr)
        fig_heatmap.update_layout(title='Tương quan giữa các biến ')

        fig_pairplot = px.scatter_matrix(self.df[columns_to_corr], title='Mối quan hệ giữa các biến')
        fig_heatmap.update_layout(
            autosize=False,
            width=12*80,
            height=6*80,
            title_x=0.4
        )
        fig_pairplot.update_layout(
            autosize=False,
            width=12*80,
            height=6*80,
            title_x=0.4
        )
        return fig_heatmap, fig_pairplot

    def distribution_closeprice(self):
        fig = px.histogram(self.df, x='close', nbins=30, title='Phân phối close Prices ')
        skewness = self.df['close'].skew()
        fig.update_layout(
            autosize=False,
            width=12*80,
            height=6*80,
            title_x=0.4,
            bargap=0.03
        )
        return fig, skewness

    def plot_close_price_comparision(self):
        self.closedf['Year'] = self.closedf['date'].dt.year
        self.closedf['Month'] = self.closedf['date'].dt.month
        years_of_interest = [2021, 2022, 2023]
        df_filtered = self.closedf[self.closedf['Year'].isin(years_of_interest)]
        grouped_data = df_filtered.groupby(['Month', 'Year'])['close'].mean().reset_index()

        fig = go.Figure()

        for year in years_of_interest:
            year_data = grouped_data[grouped_data['Year'] == year]
            fig.add_trace(go.Bar(x=year_data['Month'], y=year_data['close'], name=str(year)))

        trendline_data = df_filtered[df_filtered['Year'] == 2022].groupby(['Month'])['close'].mean().reset_index()
        fig.add_trace(go.Scatter(x=trendline_data['Month'], y=trendline_data['close'],
                                mode='lines', line=dict(color='red'), name='Trendline 2022'))

        fig.update_layout(
            barmode='group',
            title='Trung bình giá đóng cửa theo tháng và năm (2021, 2022, 2023)',
            xaxis_title='Month',
            yaxis_title='Close Price Average',
            legend_title='Year',
            autosize=False,
            width=12*80,
            height=6*80,
            title_x=0.3
        )

        return fig
    
    def plot_close_prices_2023_bymonth(self):
        fig = px.bar(self.close_stock_2023.groupby(self.close_stock_2023['date'].dt.month)['close'].mean().reset_index(),
                    x='date', y='close', title='Trung bình close prices theo tháng năm 2023',
                    labels={'close': 'Price'})
        x_values = np.unique(self.close_stock_2023['date'].dt.month)
        y_values = self.close_stock_2023.groupby(self.close_stock_2023['date'].dt.month)['close'].mean()
        slope, intercept = np.polyfit(x_values, y_values, 1)
        fig.add_trace(go.Scatter(x=x_values, y=slope * x_values + intercept,
                                mode='lines', line=dict(color='red')))
        fig.update_layout(
            autosize=False,
            width=12*80,
            height=6*80,
            barmode='group',
            title_x=0.4
        )
        return fig

    def plot_profit_margin_comparison(self):
        self.closedf['Return'] = self.closedf['close'].pct_change() * 100

        fig = go.Figure()

        for year in [2021, 2022, 2023]:
            data_year = self.closedf[self.closedf['date'].dt.year == year]
            fig.add_trace(go.Scatter(x=data_year['date'], y=data_year['Return'], mode='lines', name=str(year)))

        fig.update_layout(
            title='So sánh tỷ suất lợi nhuận giữa các năm',
            xaxis_title='Date',
            yaxis_title='Profit margin (%)',
            legend_title='Year',
            autosize=False,
            width=12*80,
            height=6*80,
            title_x=0.4
        )

        return fig
# Class model
class TRAIN_MODELS:
    def __init__(self, df):
        self.df = df
        self.closedf = self.df[['date', 'close']].copy()
        self.best_alpha_optuna = None
        self.best_beta_optuna = None
        self.best_gamma_optuna = None
        self.best_seasonal_optuna = None
        self.mae = None
        self.progress_bar = None

    # Moving average
    def dynamic_moving_average(self, window_size=1):
        prediction_column = self.closedf['close'].rolling(
            window=window_size).mean()
        return prediction_column
    #     # Các chỉ số đánh giá:
    def plot_dynamic_moving_average(self, window_size=1):
        prediction_column = self.dynamic_moving_average(window_size=window_size)
        prediction_column = prediction_column[-window_size:]
        prediction_column.index = np.arange(
            max(self.closedf.index), max(self.closedf.index) + len(prediction_column))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['close'],
                                mode='lines', name='Actual Close Price'))
        fig.add_trace(go.Scatter(x=prediction_column.index, y=prediction_column,
                                mode='lines', line=dict(color='red'),
                                name=f' Moving Average (Window = {window_size})'))
        fig.update_layout(
            title=f'Dynamic Moving Average Model (Window = {window_size})',
            xaxis_title='Date',
            yaxis_title='Close Price',
            legend_title='Legend',
            autosize=False,
            width=13*80,
            height=6*80,
            title_x=0.3
        )

        return fig
    def plot_dynamic_moving_average_acuracy(self, window_size=1):
        # Tính cột predict cho window_size cụ thể
        prediction_column = self.dynamic_moving_average(window_size=window_size)

        # Tạo đồ thị Plotly
        fig = go.Figure()

        # Vẽ đường cho Actual Close Price
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['close'],
                                mode='lines', name='Actual Close Price'))

        # Vẽ đường cho Dynamic Moving Average
        fig.add_trace(go.Scatter(x=prediction_column.index, y=prediction_column,
                                mode='lines', line=dict(color='red'),
                                name=f'Moving Average (Window = {window_size})'))

        fig.update_layout(
            title=f'Dynamic Moving Average Model (Window = {window_size})',
            xaxis_title='Date',
            yaxis_title='Close Price',
            legend_title='Legend',
            autosize=False,
            width=13*80,
            height=6*80,
            title_x=0.3
        )

        return fig
    def evaluate_dynamic_moving_average(self, window_size=1):
        mae = mean_absolute_error(
            self.closedf['close'][window_size-1:], self.closedf['Predict'][window_size-1:])
        mape = mean_absolute_percentage_error(
            self.closedf['close'][window_size-1:], self.closedf['Predict'][window_size-1:])
        mse = mean_squared_error(
            self.closedf['close'][window_size-1:], self.closedf['Predict'][window_size-1:])
        r2 = r2_score(self.closedf['close'][window_size-1:],
                      self.closedf['Predict'][window_size-1:])
        st.write(
            f"mean_absolute_error (Dynamic_MA, Window = {window_size}):", mae)
        st.write(
            f"mean_absolute_percentage_error (Dynamic_MA, Window = {window_size}):", mape)
        st.write(
            f"mean_square_error (Dynamic_MA, Window = {window_size}):", mse)
        st.write(f"r2_score (Dynamic_MA, Window = {window_size}):", r2)
    # Exponential Smoothing

    def optimize_alpha_optuna(self, n_trials=100):
        self.progress_bar = st.progress(0)

        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.01, 0.99)
            model = SimpleExpSmoothing(
                self.closedf['close']).fit(smoothing_level=alpha)
            predictions = model.fittedvalues
            MAE = mean_absolute_error(
                self.closedf['close'], predictions.dropna())
            return MAE

        study = optuna.create_study(direction='minimize')
        total_trials = n_trials
        for i in range(n_trials):
            progress = (i + 1) / total_trials
            self.progress_bar.progress(progress)
            study.optimize(objective, n_trials=1)
        # study.optimize(objective, n_trials=n_trials)
        self.best_alpha_optuna = study.best_params['alpha']
        self.mae = study.best_value
        return self.best_alpha_optuna, self.mae

    def fit_optimal_ses_model(self, alpha, steps):
        model = SimpleExpSmoothing(
            self.closedf['close']).fit(smoothing_level=alpha)
        self.closedf['SES_Optimal_Optuna'] = model.fittedvalues
        forecast_values = model.forecast(steps=steps)
        return forecast_values

    def plot_ses_results(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['SES_Optimal_Optuna'],
                                mode='lines', name=f'SES (Optimal Alpha = {self.best_alpha_optuna:.3f})'))
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['close'],
                                mode='lines', name='Actual Close Price'))
        fig.update_layout(
            title=f'Simple Exponential Smoothing (Optimal Alpha = {self.best_alpha_optuna:.3f})',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            autosize=False,
            width=13*80,
            height=6*80,
            title_x=0.3
        )
        return fig

    def plot_ses_forecast_results(self, forecast_values):
        fig = go.Figure()
        forecast_column = forecast_values
        forecast_column.index = np.arange(
            max(self.closedf.index), max(self.closedf.index) + len(forecast_column))
        fig.add_trace(go.Scatter(x=forecast_column.index, y=forecast_column,
                                mode='lines', name=f'SES (Optimal Alpha = {self.best_alpha_optuna:.3f})'))
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['close'],
                                mode='lines', name='Actual Close Price'))

        fig.update_layout(
            title=f'Simple Exponential Smoothing (Optimal Alpha = {self.best_alpha_optuna:.3f})',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            autosize=False,
            width=13*80,
            height=6*80,
            title_x=0.3
        )

        st.write("Optimal Alpha:", self.best_alpha_optuna)
        st.write("Best MAE - Optuna:", self.mae)
        self.progress_bar.empty()

        return fig
    # Holt model

    def optimize_alpha_beta_optuna(self, n_trials=200):
        self.progress_bar = st.progress(0)

        def objective_holt(trial):
            alpha = trial.suggest_float('alpha', 0.01, 0.99)
            beta = trial.suggest_float('beta', 0.01, 0.99)
            model = ExponentialSmoothing(self.closedf['close'], trend='add', damped=True).fit(
                smoothing_level=alpha, smoothing_slope=beta)
            predictions = model.fittedvalues
            MAE = mean_absolute_error(
                self.closedf['close'], predictions.dropna())
            return MAE

        study = optuna.create_study(direction='minimize')
        total_trials = n_trials
        for i in range(n_trials):
            progress = (i + 1) / total_trials
            self.progress_bar.progress(progress)
            study.optimize(objective_holt, n_trials=1)
        # study.optimize(objective_holt, n_trials=n_trials)
        self.best_alpha_optuna = study.best_params['alpha']
        self.best_beta_optuna = study.best_params['beta']
        self.mae = study.best_value
        return self.best_alpha_optuna, self.best_beta_optuna, self.mae

    def fit_optimal_holt_model(self, alpha, beta, steps):
        model = ExponentialSmoothing(self.closedf['close'], trend='add', damped=True).fit(
            smoothing_level=alpha, smoothing_slope=beta)
        self.closedf['Holt_Optimal_Optuna'] = model.fittedvalues
        forecast_values = model.forecast(steps=steps)
        return forecast_values
    def plot_holt_results(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['Holt_Optimal_Optuna'],
                                mode='lines',
                                name=f'Holt (Alpha= {self.best_alpha_optuna:.3f}, Beta= {self.best_beta_optuna:.3f})'))
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['close'],
                                mode='lines', name='Actual Close Price'))
        fig.update_layout(
            title='Holt Model (Optimal Alpha/Beta)',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            autosize=False,
            width=13*80,
            height=6*80,
            title_x=0.3
        )
        return fig
    def plot_forecast_holt_results(self, forecast_values):
        fig = go.Figure()
        forecast_column = forecast_values
        forecast_column.index = np.arange(
            max(self.closedf.index), max(self.closedf.index) + len(forecast_column))
        fig.add_trace(go.Scatter(x=forecast_column.index, y=forecast_column,
                                mode='lines',
                                name=f'Holt (Alpha= {self.best_alpha_optuna:.3f}, Beta= {self.best_beta_optuna:.3f})'))
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['close'],
                                mode='lines', name='Actual Close Price'))
        fig.update_layout(
            title='Holt Model (Optimal Alpha/Beta)',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            autosize=False,
            width=13*80,
            height=6*80,
            title_x=0.3
        )

        st.write("Optimal Alpha:", self.best_alpha_optuna)
        st.write("Optimal Beta:", self.best_beta_optuna)
        st.write("Best MAE - Optuna:", self.mae)
        self.progress_bar.empty()

        return fig
    # Holt winter model

    def optimize_holtwinter(self, seasonal_periods=60, n_trials=200):
        self.progress_bar = st.progress(0)

        def objective_holtwinter(trial):
            alpha = trial.suggest_float('alpha', 0.01, 0.99)
            beta = trial.suggest_float('beta', 0.01, 0.99)
            gamma = trial.suggest_float('gamma', 0, 0.99)
            seasonal = trial.suggest_categorical(
                'seasonal', ['add', 'multiplicative'])

            model = ExponentialSmoothing(self.closedf['close'], trend='add', seasonal=seasonal, seasonal_periods=seasonal_periods, damped=True).fit(
                smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
            predictions = model.fittedvalues
            MAE = mean_absolute_error(
                self.closedf['close'], predictions.dropna())
            return MAE

        study = optuna.create_study(direction='minimize')
        # Chạy thanh tiến trình
        total_trials = n_trials
        for i in range(n_trials):
            progress = (i + 1) / total_trials
            self.progress_bar.progress(progress)
            study.optimize(objective_holtwinter, n_trials=1)
        # study.optimize(objective_holtwinter, n_trials=n_trials)

        self.best_alpha_optuna = study.best_params['alpha']
        self.best_beta_optuna = study.best_params['beta']
        self.best_gamma_optuna = study.best_params['gamma']
        self.best_seasonal_optuna = study.best_params['seasonal']
        self.mae = study.best_value

        return self.best_alpha_optuna, self.best_beta_optuna, self.best_gamma_optuna, self.best_seasonal_optuna, self.mae

    def fit_optimal_holtwinter_model(self, alpha, beta, gamma, seasonal_periods, seasonal, steps):
        model = ExponentialSmoothing(self.closedf['close'], trend='add', seasonal=seasonal, seasonal_periods=seasonal_periods, damped=True).fit(
            smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
        self.closedf['Holt_Winters_Optimal'] = model.fittedvalues
        forecast_values = model.forecast(steps=steps)
        return forecast_values
    def plot_holtwinter_results(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['Holt_Winters_Optimal'],
                                mode='lines',
                                name=f'Holt-Winters (Alpha:{self.best_alpha_optuna:.3f},Beta:{self.best_beta_optuna:.3f}, Gamma:{self.best_gamma_optuna:.3f}, Seasonal:{self.best_seasonal_optuna})'))
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['close'],
                                mode='lines', name='Actual Close Price'))

        fig.update_layout(
            title='Holt-Winters (Optimal Alpha/Beta/Gamma/Seasonal)',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            autosize=False,
            width=13*80,
            height=6*80,
            title_x=0.3
        )
        return fig

    def plot_forecast_holtwinter_results(self, forecast_values):
        fig = go.Figure()
        forecast_column = forecast_values
        forecast_column.index = np.arange(
            max(self.closedf.index), max(self.closedf.index) + len(forecast_column))
        fig.add_trace(go.Scatter(x=forecast_column.index, y=forecast_column,
                                mode='lines',
                                name=f'Holt-Winters (Alpha:{self.best_alpha_optuna:.3f},Beta:{self.best_beta_optuna:.3f}, Gamma:{self.best_gamma_optuna:.3f}, Seasonal:{self.best_seasonal_optuna})'))
        fig.add_trace(go.Scatter(x=self.closedf.index, y=self.closedf['close'],
                                mode='lines', name='Actual Close Price'))

        fig.update_layout(
            title='Holt-Winters (Optimal Alpha/Beta/Gamma/Seasonal)',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend',
            autosize=False,
            width=13*80,
            height=6*80,
            title_x=0.3
        )

        st.write("Optimal Alpha:", self.best_alpha_optuna)
        st.write("Optimal Beta:", self.best_beta_optuna)
        st.write("Optimal Gamma:", self.best_gamma_optuna)
        st.write("Optimal Seasonal:", self.best_seasonal_optuna)
        st.write("Best MAE:", self.mae)
        self.progress_bar.empty()

        return fig   
    # Đánh giá Model

    def evaluate_model(self, columns):
        mape_sesop = mean_absolute_percentage_error(
            self.closedf['close'], columns)
        mse_sesop = mean_squared_error(self.closedf['close'], columns)
        r2_sesop = r2_score(self.closedf['close'], columns)

        st.write("Mean Absolute Percentage Error:", mape_sesop)
        st.write("Mean Squared Error:", mse_sesop)
        st.write("R-squared:", r2_sesop)
# Class Time Series 
class TIME_SERIES:
    def __init__(self,df):
        self.df = df
        self.closedf = self.df[['close','date']].copy()
    
    # def plot_seasonal_analysis(self,forecast_values):
        

# Thêm mã HTML để căn giữa tiêu đề
st.markdown("<h1 style='text-align: center;'>Stock Price Predictions</h1>",
            unsafe_allow_html=True)
st.sidebar.markdown("""
    <h1 style='font-size:35px;text-align:center'>TEAM 1</h1>
""", unsafe_allow_html=True)    
st.sidebar.markdown("""
        <div style="display: flex; justify-content: center;">
            <img src='https://scontent.fsgn5-9.fna.fbcdn.net/v/t39.30808-6/242489593_405101811147345_1733417058228090429_n.jpg?_nc_cat=102&ccb=1-7&_nc_sid=173fa1&_nc_ohc=bhYKiwdAEyAAX_l1-bz&_nc_ht=scontent.fsgn5-9.fna&oh=00_AfDuU3ENr2x7o-ugDanKzy0A7Xuz2cKIdHbuQTyQjAKRGg&oe=65983BAE' alt='Ten_Hinh_Anh' width='60%' style='border-radius:50%;margin-bottom:12%;'>
        </div>
        """, unsafe_allow_html=True)    
st.sidebar.info(
    'Welcome to the Stock Price Prediction App. Choose your options below')
st.sidebar.info(
    "Created and designed by [Team Data Science - QuocChienDuc](https://github.com/davisduccopny/Stock-Prediction-with-Python-project/)")


def main():
    st.sidebar.title('Lựa chọn')
    option = st.sidebar.radio('Chọn một tab:', [
                              'Trực quan', 'Thống kê mô tả', 'Phân tách Times Series', 'Prediction'])

    if option == 'Trực quan':
        introduction_stock()
    elif option == 'Thống kê mô tả':
        statistical_des()
    elif option == 'Phân tách Times Series':
        dataframe()
    else:
        predict()


# Tăng tốc tải bằng việc lưu dữ liệu vào cache
@st.cache_resource
# Hàm download data
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df


# Danh sách mã cổ phiếu
stock_options = ['TSLA', 'BMW.DE', '7203.T', 'VOW3.DE', 'F']

# Lựa chọn mã cổ phiếu thông qua select box
option_stock_name = st.sidebar.selectbox(
    'Select a Stock Symbol', stock_options)

# Chuyển đổi mã cổ phiếu thành chữ hoa để đảm bảo tính nhất quán
option_stock_name = option_stock_name.upper()
today = datetime.date.today()
# Nhập dữ liệu:
with st.sidebar.container():
    expander = st.expander("Times Select")
    with expander:
        duration = st.number_input('Enter the duration', value=1824)
        before = today - datetime.timedelta(days=duration)
        start_date = st.date_input('Start Date', value=before)
        end_date = st.date_input('End date', today)
        duration_2 = (end_date - start_date).days
        st.info(f"Final duration: {duration_2} days")

if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %
                           (start_date, end_date))
        data = download_data(option_stock_name, start_date, end_date)
        # Bây giờ bạn có thể sử dụng biến "data" để làm những việc khác
        st.write(data.head())  # In ra vài dòng đầu của dữ liệu
    else:
        st.sidebar.error('Error: End date must fall after start date')


data = download_data(option_stock_name, start_date, end_date)
scaler = StandardScaler()
# Hàm cleaning data


def clean_dataframe(dataframe):
    dataframe = dataframe.rename(columns={'Date': 'date', 'Open': 'open', 'Hight': 'hight',
                                 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'})
    dataframe['date'] = pd.to_datetime(dataframe.index, errors='coerce')
    dataframe = dataframe.dropna()
    dataframe = dataframe.drop_duplicates()
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


# Cleaning data
data = clean_dataframe(data)


def display_file_content(file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Kết hợp đường dẫn tuyệt đối với đường dẫn tương đối của tệp tin
    full_file_path = os.path.join(current_dir, file_path)

    if os.path.exists(full_file_path):
        with open(full_file_path, "r", encoding="utf-8") as file:
            try:
                # Đọc từng dòng và thêm ký tự xuống dòng
                lines = file.readlines()
                content = "\n".join(lines).strip()
                st.info(f"### Giới thiệu\n{content}")
            except UnicodeDecodeError:
                st.error(
                    f"Tệp tin '{full_file_path}' không thể đọc với encoding utf-8.")
    else:
        st.error(f"Tệp tin '{full_file_path}' không tồn tại.")


def embed_image(file_path):
    # Lấy đường dẫn thư mục hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Kết hợp đường dẫn tới tệp ảnh
    full_file_path = os.path.join(current_dir, file_path)

    # Đọc nội dung hình ảnh và chuyển thành chuỗi base64
    with open(full_file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Nhúng vào mã HTML và hiển thị trên Streamlit
    html_code = f"""
    <div style="display: flex; justify-content: center;">
        <img src='data:image/jpeg;base64,{encoded_image}' alt='Ten_Hinh_Anh' width='100%' style='border-radius:60%; margin-bottom:5%;'>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def introduction_stock():
    if option_stock_name == 'TSLA':
        st.header("Tesla, Inc. (TSLA)")
    elif option_stock_name == '7203.T':
        st.header("Toyota Motor Corporation (7203.T)")
    elif option_stock_name == 'BMW.DE':
        st.header("BMW AG (BMW.DE)")
    elif option_stock_name == 'VOW3.DE':
        st.header("Volkswagen AG (VOW3.DE)")
    else:
        st.header("Ford Motor Company (F)")
    option = st.radio(
        'Choose a Technical Indicator to Visualize', ['Close', 'BB'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['close', 'bb_h', 'bb_l']]
    total_volume = data['volume'].sum()
    max_price = data['close'].max()
    min_price = data['close'].min()

    # Layout dashboard
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Total Volume:**\n\n{total_volume} ")

    with col2:
        st.success(f"**Max Price:**\n\n{max_price} USD")

    with col3:
        st.warning(f"**Min Price:**\n\n{min_price} USD")

    if option == 'Close':
        st.write('Close Price')
        st.line_chart(data.close)

    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)

    if option_stock_name == 'TSLA':
        embed_image('./asset/image/logo-tesla.jpg')
        display_file_content("./info_stock/tsla.txt")

    elif option_stock_name == '7203.T':
        embed_image('./asset/image/logo-toyota.png')
        display_file_content("./info_stock/toyota.txt")
    elif option_stock_name == 'BMW.DE':
        embed_image('./asset/image/logo-bmw.jpg')
        display_file_content("./info_stock/bmw.txt")
    elif option_stock_name == 'VOW3.DE':
        embed_image('./asset/image/logo-wow.jpg')
        display_file_content("./info_stock/wow3.txt")
    else:
        embed_image('./asset/image/logo-ford.jpg')
        display_file_content("./info_stock/ford.txt")


def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))


def statistical_des():
    st.header("Thống kê mô tả")
    st.subheader("Các chỉ số cơ bản")
    st.dataframe(data.describe())
    stock_statistic_dv = DESCRIPTIVE_STATISTICS(data)
    st.subheader("Kiểm tra sự khác biệt các biến")
    with st.expander("Trung bình giá giao dịch theo tháng"):
        sub_month = stock_statistic_dv.analyze_monthly_average()
        st.plotly_chart(stock_statistic_dv.plot_monthly_average_bar_chart(sub_month))
    with st.expander("Trung bình giá giao dịch theo năm"):
        st.plotly_chart(stock_statistic_dv.plot_yearly_average_line_chart())
        fig_heatmap, fig_pairplot = stock_statistic_dv.relationship_correlation()
    with st.expander("Heatmap"):
        st.plotly_chart(fig_heatmap)
    with st.expander("Pairplot"):
        st.plotly_chart(fig_pairplot)
    st.subheader("Phân phối cổ phiếu")
    with st.expander("Histogram"):
        fig_distribution, skewness = stock_statistic_dv.distribution_closeprice()
        st.plotly_chart(fig_distribution)
        st.write("Độ xiên close price:", skewness)
    st.subheader("Cổ phiếu 2023")
    with st.expander("Trung bình close prices theo tháng năm 2023"):
        st.plotly_chart(stock_statistic_dv.plot_close_prices_2023_bymonth())
    st.subheader("So sánh với các năm")
    with st.expander("Trung bình giá đóng cửa theo tháng và năm (2021, 2022, 2023)"):
        st.plotly_chart(stock_statistic_dv.plot_close_price_comparision())
    with st.expander("Tỷ suất lợi nhuận"):
        st.plotly_chart(stock_statistic_dv.plot_profit_margin_comparison())


def predict():
    st.header("Dự báo giá cổ phiếu (Stock Price Prediction)")
    col_predict_1, col_predict_2 = st.columns(2)
    with col_predict_1:
        model = st.radio('Chọn mô hình', [
                         'Holt Winter', 'Holt', 'Exponential Smoothing', 'Simple Moving Average'])
    with col_predict_2:
        option_time = st.radio('Chọn thời gian dự đoán:', [
                               '1 ngày', '1 tuần', '1 tháng', 'Khác'])
    if option_time == '1 ngày':
        num = 1
    elif option_time == '1 tuần':
        num = 7
    elif option_time == '1 tháng':
        num = 30
    else:
        num = st.number_input('How many days forecast?', value=5)
    num = int(num)

    if st.button('Predict'):
        model_trainer = TRAIN_MODELS(data)
        if model == 'Holt Winter':
            best_alpha_optuna_hw, best_beta_optuna_hw, best_gamma_optuna_hw, best_seasonal_optuna_hw, mae_best_holtwinter_hw = model_trainer.optimize_holtwinter(
                seasonal_periods=60, n_trials=100)
            forecast_values_hw = model_trainer.fit_optimal_holtwinter_model(
                best_alpha_optuna_hw, best_beta_optuna_hw, best_gamma_optuna_hw, 60, best_seasonal_optuna_hw, steps=num)
            tab1, tab2, tab3 = st.tabs(
                ["📈 Chart train", "📈 Chart predict", "🗃 Data"])
            with tab1:
                st.plotly_chart(model_trainer.plot_holtwinter_results())
                model_trainer.evaluate_model(
                    columns=model_trainer.closedf['Holt_Winters_Optimal'])
            with tab2:
                st.plotly_chart(model_trainer.plot_forecast_holtwinter_results(
                    forecast_values=forecast_values_hw))
            with tab3:
                forecast_pred = forecast_values_hw.values
                day = 1
                for i in forecast_pred:
                    st.text(f'Day {day}: {i}')
                    day += 1
        elif model == 'Holt':
            best_alpha_optuna, best_beta_optuna, mae_best_holt = model_trainer.optimize_alpha_beta_optuna(
                n_trials=200)
            forecast_values_holt = model_trainer.fit_optimal_holt_model(
                best_alpha_optuna, best_beta_optuna, steps=num)
            tab1, tab2, tab3 = st.tabs(
                ["📈 Chart train", "📈 Chart predict", "🗃 Data"])
            with tab1:
                st.plotly_chart(model_trainer.plot_holt_results())
                model_trainer.evaluate_model(
                    columns=model_trainer.closedf['Holt_Optimal_Optuna'])
            with tab2:
                st.plotly_chart(model_trainer.plot_forecast_holt_results(
                    forecast_values=forecast_values_holt))
            with tab3:
                forecast_pred = forecast_values_holt.values
                day = 1
                for i in forecast_pred:
                    st.text(f'Day {day}: {i}')
                    day += 1
        elif model == 'Exponential Smoothing':
            best_alpha_optuna, mae = model_trainer.optimize_alpha_optuna(
                n_trials=100)
            forecast_values = model_trainer.fit_optimal_ses_model(
                best_alpha_optuna, steps=num)
            tab1, tab2, tab3 = st.tabs(
                ["📈 Chart train", "📈 Chart predict", "🗃 Data"])
            with tab1:
                st.plotly_chart(model_trainer.plot_ses_results())
                model_trainer.evaluate_model(
                    columns=model_trainer.closedf['SES_Optimal_Optuna'])
            with tab2:
                st.plotly_chart(model_trainer.plot_ses_forecast_results(
                    forecast_values=forecast_values))
            with tab3:
                forecast_pred = forecast_values.values
                day = 1
                for i in forecast_pred:
                    st.text(f'Day {day}: {i}')
                    day += 1
        else:
            model_trainer.closedf['Predict'] = model_trainer.dynamic_moving_average(
                window_size=num)
            tab1, tab2, tab3 = st.tabs(
                ["📈 Chart train", "📈 Chart predict", "🗃 Data"])
            with tab1:
                fig_movingaverage = model_trainer.plot_dynamic_moving_average_acuracy(window_size=num)
                st.plotly_chart(fig_movingaverage)
                model_trainer.evaluate_dynamic_moving_average(window_size=num)
            with tab2:
                fig_movingaverage2 = model_trainer.plot_dynamic_moving_average(
                    window_size=num)
                st.plotly_chart(fig_movingaverage2)
            with tab3:
                forecast_pred = model_trainer.closedf['Predict'][-num:]
                day = 1
                for i in forecast_pred:
                    st.text(f'Day {day}: {i}')
                    day += 1


if __name__ == '__main__':
    main()
