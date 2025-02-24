"""
Stock Price Prediction Model with Dynamic Ensemble
===============================================
This script implements a hybrid forecasting model combining Prophet and SARIMAX
with dynamic weighting based on prediction horizon. It incorporates multiple
factors including technical indicators, market sentiment, and fundamental data.

Model Weights and Factors:
-------------------------
1. Time Horizon Weighting:
   - Short-term (1-30 days): SARIMAX 70%, Prophet 30%
   - Medium-term (31-60 days): Equal weights (50-50)
   - Long-term (61-90 days): Prophet 70%, SARIMAX 30%

2. Additional Factors:
   - P/E Ratio: Fundamental factor indicating valuation
   - Analyst Target Price: Market expectations
   - Fear & Greed Index: Market sentiment indicator

3. Model Components:
   - Prophet: Handles seasonality, holidays, and trend
   - SARIMAX: Captures auto-regression and moving average patterns
   - Ensemble: Dynamically weighted combination of both models
"""

# =================
# Library Imports
# =================
import os
import yfinance as yf
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import ta  # Replace talib with ta library
from sklearn.impute import SimpleImputer
from scipy import stats

# Try to import bokeh, fallback to matplotlib if not available
try:
    from bokeh.plotting import figure, show, output_file
    from bokeh.models import ColumnDataSource, Band
    from bokeh.layouts import column, row
    from bokeh.models import DataTable, TableColumn, ColumnDataSource
    from bokeh.models import Button, Column, Row
    from bokeh.layouts import layout
    from bokeh.models import Toggle, CustomJS
    USE_BOKEH = True
except ImportError:
    USE_BOKEH = False
    import matplotlib.pyplot as plt
    print("Bokeh import failed. Using matplotlib for visualization.")

# ========================
# Data Collection Methods
# ========================
"""
The following functions handle data acquisition from various sources:
- Stock price data from Yahoo Finance
- P/E ratios and analyst targets from Yahoo Finance
- Fear & Greed Index from Alternative.me API
"""

# Function to download stock data using yfinance
def download_stock_data(ticker):
    print("Downloading stock data...")
    data = yf.download(ticker, start='2020-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    return data

# Function to download additional data using yfinance
def download_additional_data(ticker):
    print("Downloading additional data...")
    stock = yf.Ticker(ticker)
    pe_ratio = stock.info['trailingPE']
    analyst_target = stock.info['targetMeanPrice']
    return pe_ratio, analyst_target

# Function to fetch Fear & Greed Index from Alternative.me API
def fetch_fear_greed_index():
    """Fetch Fear & Greed Index from Alternative.me API"""
    print("Fetching Fear & Greed Index...")
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url)
        data = response.json()
        if data['data'][0]:
            return int(data['data'][0]['value'])
    except Exception as e:
        print(f"Warning: Could not fetch Fear & Greed Index: {e}")
        return 50  # Return neutral value as fallback

def get_earnings_dates(ticker):
    """Fetch historical and future earnings dates with improved error handling"""
    print("Fetching earnings dates...")
    stock = yf.Ticker(ticker)  # Use actual ticker symbol, not date string
    earnings_dates = []
    
    try:
        # Try getting from income statement first (more reliable)
        income_stmt = stock.income_stmt
        if income_stmt is not None and not income_stmt.empty:
            stmt_dates = pd.to_datetime(income_stmt.columns)
            earnings_dates.extend(stmt_dates)
        
        # Try getting from quarterly statements
        quarterly_stmt = stock.quarterly_income_stmt
        if quarterly_stmt is not None and not quarterly_stmt.empty:
            quarterly_dates = pd.to_datetime(quarterly_stmt.columns)
            earnings_dates.extend(quarterly_dates)
        
        # Remove duplicates and sort
        earnings_dates = sorted(list(set(earnings_dates)))
        
        if len(earnings_dates) > 0:
            print(f"Found {len(earnings_dates)} historical earnings dates")
            return earnings_dates
        
    except Exception as e:
        print(f"Warning: Error fetching earnings dates: {e}")
    
    # Fallback: estimate future dates based on typical reporting schedule
    print("Using estimated quarterly dates as fallback...")
    today = datetime.now()
    current_year = today.year
    fallback_dates = []
    
    # Include past 2 years and future 1 year
    for year in range(current_year - 2, current_year + 2):
        # UK companies typically report in March (full year) and September (half year)
        reporting_months = [3, 9]  # Adjust for RR.L reporting schedule
        for month in reporting_months:
            date = pd.Timestamp(f"{year}-{month:02d}-15")
            if date >= pd.Timestamp('2020-01-01'):  # Don't go earlier than data start
                fallback_dates.append(date)
    
    # Sort and filter out past dates for future predictions
    fallback_dates = sorted([d for d in fallback_dates if d > pd.Timestamp('2020-01-01')])
    return fallback_dates

# ==========================
# Data Preparation Methods
# ==========================
"""
These functions prepare the data for model training:
- Formats datetime index
- Handles missing values
- Combines multiple data sources
"""

def calculate_technical_indicators(data):
    """Calculate various technical indicators and return as DataFrame"""
    df = pd.DataFrame(index=data.index)
    
    # Ensure we're working with 1D Series objects by using .squeeze()
    close = data['Close'].squeeze()
    high = data['High'].squeeze()
    low = data['Low'].squeeze()
    volume = data['Volume'].squeeze()
    
    # Convert to float if needed
    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volume = volume.astype(float)
    
    try:
        # RSI calculation
        df['RSI'] = ta.momentum.rsi(close, window=14)
        
        # Moving averages
        df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
        df['SMA_50'] = ta.trend.sma_indicator(close, window=50)
        
        # MACD
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # ADX
        df['ADX'] = ta.trend.adx(high, low, close, window=14)
        
        # Volume indicators
        df['OBV'] = ta.volume.on_balance_volume(close, volume)
        
        # Volatility indicators
        df['ATR'] = ta.volatility.average_true_range(high, low, close)
        
        # Bollinger Bands
        df['Bollinger_Upper'] = ta.volatility.bollinger_hband(close)
        df['Bollinger_Middle'] = ta.volatility.bollinger_mavg(close)
        df['Bollinger_Lower'] = ta.volatility.bollinger_lband(close)
        
    except Exception as e:
        print(f"Warning: Error calculating technical indicators: {e}")
        # Return dummy values if calculation fails
        for col in ['RSI', 'SMA_20', 'SMA_50', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                   'ADX', 'OBV', 'ATR', 'Bollinger_Upper', 'Bollinger_Middle', 'Bollinger_Lower']:
            df[col] = 0.0
    
    # Replace fillna with forward fill then backward fill
    return df.ffill().bfill()

def perform_factor_analysis(data, technical_indicators, target_returns):
    """Perform factor analysis with enhanced importance calculation and robust handling of extreme values"""
    print("\nPerforming Factor Analysis on Technical Indicators...")
    
    # Calculate percentage changes with clipping to handle extreme values
    tech_pct_change = technical_indicators.pct_change()
    
    # Replace inf/-inf with nan, then fill with closest valid value
    tech_pct_change = tech_pct_change.replace([np.inf, -np.inf], np.nan)
    
    # Clip extreme values to 3 standard deviations
    for col in tech_pct_change.columns:
        series = tech_pct_change[col].dropna()
        if len(series) > 0:
            mean = series.mean()
            std = series.std()
            tech_pct_change[col] = tech_pct_change[col].clip(
                lower=mean - 3*std,
                upper=mean + 3*std
            )
    
    # Forward fill then backward fill remaining NaNs
    tech_pct_change = tech_pct_change.ffill().bfill()
    
    # Ensure target returns are also cleaned
    target_returns = target_returns.replace([np.inf, -np.inf], np.nan)
    target_returns = target_returns.ffill().bfill()
    
    try:
        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(tech_pct_change)
        scaled_features = pd.DataFrame(scaled_features, columns=technical_indicators.columns)
        
        # Calculate correlations with returns using cleaned data
        correlations = []
        for col in scaled_features.columns:
            mask = ~(np.isnan(scaled_features[col]) | np.isnan(target_returns))
            if mask.any():
                corr, _ = stats.pearsonr(scaled_features[col][mask], target_returns[mask])
                correlations.append(abs(corr))
            else:
                correlations.append(0)
        correlations = np.array(correlations)
        
        # Calculate rolling standard deviations to measure volatility impact
        volatility_impact = np.array([tech_pct_change[col].rolling(window=20).std().mean() 
                                 for col in tech_pct_change.columns])
    
        # Combine correlation and volatility for importance
        raw_importance = correlations * volatility_impact
    
        # Apply minimum threshold and normalize
        min_threshold = np.percentile(raw_importance, 25)  # Bottom 25% get filtered out
        raw_importance[raw_importance < min_threshold] = 0
        importance_scores = raw_importance / np.sum(raw_importance)
    
        # Create dictionaries with more meaningful scaling
        feature_importance = dict(zip(technical_indicators.columns, raw_importance))
        indicator_weights = dict(zip(technical_indicators.columns, importance_scores))
    
        # Print detailed analysis
        print("\nDetailed Factor Analysis:")
        print(f"{'Indicator':<15} {'Correlation':>10} {'Volatility':>10} {'Importance':>10}")
        print("-" * 50)
        for col, corr, vol, imp in zip(technical_indicators.columns, 
                                  correlations, volatility_impact, importance_scores):
            print(f"{col:<15} {corr:>10.4f} {vol:>10.4f} {imp:>10.4f}")
    
        return indicator_weights, feature_importance
        
    except Exception as e:
        print(f"Warning: Error in factor analysis: {e}")
        # Return equal weights if factor analysis fails
        n_features = len(technical_indicators.columns)
        equal_weight = 1.0 / n_features
        indicator_weights = {col: equal_weight for col in technical_indicators.columns}
        feature_importance = {col: equal_weight for col in technical_indicators.columns}
        return indicator_weights, feature_importance

def create_earnings_features(data, earnings_dates):
    """Create earnings-related features with proper dtype handling"""
    # Initialize earnings impact series with float dtype
    earnings_impact = pd.Series(0.0, index=data.index, dtype=float)
    
    for earning_date in earnings_dates:
        # Convert earning_date to pandas Timestamp if needed
        earning_date = pd.Timestamp(earning_date)
        
        # Mark days before earnings (anticipation effect)
        mask_before = (data.index >= earning_date - pd.Timedelta(days=5)) & (data.index < earning_date)
        earnings_impact.loc[mask_before] = 0.5
        
        # Mark earnings day and day after (reaction effect)
        mask_after = (data.index >= earning_date) & (data.index <= earning_date + pd.Timedelta(days=1))
        earnings_impact.loc[mask_after] = 1.0
    
    return earnings_impact

# Function to prepare data for Prophet
def prepare_prophet_data(data, pe_ratio, analyst_target, fear_greed_index):
    """Enhanced data preparation with improved technical scoring and robust data handling"""
    print("Preparing Prophet model data...")
    
    # Calculate technical indicators
    technical_indicators = calculate_technical_indicators(data)
    
    # Calculate daily returns with robust handling
    returns = data['Close'].pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    
    # Handle outliers more robustly using numpy operations
    returns_array = returns.values.ravel()  # Ensure 1D array
    valid_returns = returns_array[~np.isnan(returns_array)]
    
    if len(valid_returns) > 0:
        lower_bound = np.percentile(valid_returns, 1)
        upper_bound = np.percentile(valid_returns, 99)
        clipped_returns = np.clip(returns_array, lower_bound, upper_bound)
        returns = pd.Series(clipped_returns, index=returns.index)
    
    # Replace deprecated fillna method with bfill()
    returns = returns.bfill()
    
    # Perform factor analysis
    indicator_weights, feature_importance = perform_factor_analysis(
        data, technical_indicators, returns
    )
    
    # Print factor analysis results
    print("\nTechnical Indicator Importance:")
    for indicator, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{indicator}: {importance:.4f}")
    
    # Create weighted technical score with enhanced impact
    technical_score = pd.Series(0.0, index=data.index)
    for col, weight in indicator_weights.items():
        # Normalize the indicator
        normalized_indicator = (technical_indicators[col] - technical_indicators[col].mean()) / technical_indicators[col].std()
        score_component = normalized_indicator * weight * 10  # Amplify the effect
        technical_score += score_component
    
    # Apply sigmoid transformation for better scaling
    technical_score = 1 / (1 + np.exp(-technical_score))
    technical_score = technical_score.fillna(0.5)  # Handle any remaining NaNs with neutral value
    
    # Prepare Prophet data with enhanced features
    prophet_data = pd.DataFrame()
    prophet_data['ds'] = data.index
    prophet_data['y'] = data['Close'].values
    prophet_data['technical_score'] = technical_score.values
    prophet_data['pe_ratio'] = pe_ratio if pe_ratio is not None else data['Close'].mean()
    prophet_data['analyst_target'] = analyst_target if analyst_target is not None else data['Close'].mean()
    prophet_data['fear_greed_index'] = fear_greed_index

    # Get earnings dates with proper ticker
    earnings_dates = get_earnings_dates('RR.L')  # Pass actual ticker symbol
    earnings_impact = create_earnings_features(data, earnings_dates)
    
    # Add earnings impact to Prophet data
    prophet_data['earnings_impact'] = earnings_impact.values
    
    return prophet_data, indicator_weights, earnings_dates

# =======================
# Model Implementation
# =======================
"""
Prophet Model Configuration:
- Yearly, weekly, and daily seasonality enabled
- Additional regressors: P/E ratio, analyst targets, fear/greed index
- Changepoint prior scale: 0.05 (controls flexibility of the trend)

SARIMAX Model Configuration:
- Order (2,1,2): AR(2), I(1), MA(2)
- Seasonal Order (1,1,1,12): Captures yearly patterns
"""

# Function to fit and predict using Prophet
def fit_predict_prophet(prophet_data):
    """Enhanced Prophet model with adjusted technical indicator impact"""
    print("Fitting Prophet model...")
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0  # Increase seasonality impact
    )
    
    # Add regressors with specified prior scales
    prophet_model.add_regressor('technical_score', prior_scale=15.0)  # Increase technical score impact
    prophet_model.add_regressor('earnings_impact', prior_scale=20.0)  # High impact for earnings
    prophet_model.add_regressor('pe_ratio', prior_scale=5.0)
    prophet_model.add_regressor('analyst_target', prior_scale=8.0)
    prophet_model.add_regressor('fear_greed_index', prior_scale=3.0)
    
    prophet_model.fit(prophet_data)
    
    # Create future dataframe with all features
    future = prophet_model.make_future_dataframe(periods=90)
    future['technical_score'] = prophet_data['technical_score'].iloc[-1]
    future['pe_ratio'] = prophet_data['pe_ratio'].iloc[-1]
    future['analyst_target'] = prophet_data['analyst_target'].iloc[-1]
    future['fear_greed_index'] = prophet_data['fear_greed_index'].iloc[-1]
    future['earnings_impact'] = prophet_data['earnings_impact'].iloc[-1]
    
    forecast = prophet_model.predict(future)
    return prophet_model, forecast

# ========================
# Ensemble Prediction
# ========================
"""
Dynamic Weighting Strategy:
1. Short-term predictions prioritize SARIMAX (70%) due to better handling of recent patterns
2. Medium-term uses equal weights to balance both models
3. Long-term favors Prophet (70%) due to better trend capture and seasonality handling
"""

def calculate_ensemble_predictions(prophet_forecast, sarimax_forecast, future_dates):
    """Calculate ensemble predictions with dynamic weights"""
    print("Creating ensemble predictions...")
    ensemble_predictions = []
    prophet_predictions = prophet_forecast['yhat'].values[-90:]

    for i in range(90):
        weights = get_dynamic_weights(i + 1)
        pred = (weights['prophet'] * prophet_predictions[i] +
                weights['sarimax'] * sarimax_forecast[i])
        ensemble_predictions.append(pred)
    
    return ensemble_predictions

def get_dynamic_weights(horizon):
    """
    Returns weights based on forecast horizon:
    - Short-term (1-30 days): SARIMAX weighted higher
    - Medium-term (31-60 days): slight SARIMAX weighted higher
    - Long-term (61-90 days): Prophet weighted higher
    """
    if horizon <= 30:
        return {'prophet': 0.2, 'sarimax': 0.8}
    elif horizon <= 60:
        return {'prophet': 0.4, 'sarimax': 0.6}
    else:
        return {'prophet': 0.6, 'sarimax': 0.4}

# =========================
# Historical Validation
# =========================
"""
Backtesting and historical prediction analysis:
- Generates in-sample predictions from both models
- Applies dynamic weighting to historical data
- Used for model validation and comparison
"""

def get_historical_predictions(prophet_model, sarimax_results, data):
    """Get historical (in-sample) predictions from both models"""
    print("Calculating historical predictions...")
    
    # Prophet historical predictions
    prophet_hist = prophet_model.predict(prophet_model.history)
    prophet_historical = prophet_hist['yhat'].values
    
    # SARIMAX historical predictions
    sarimax_historical = sarimax_results.get_prediction(start=0).predicted_mean.values
    
    # Ensure arrays are the same length
    min_length = min(len(prophet_historical), len(sarimax_historical), len(data))
    prophet_historical = prophet_historical[:min_length]
    sarimax_historical = sarimax_historical[:min_length]
    
    # Calculate historical ensemble using numpy for efficiency
    weights_array = np.array([get_dynamic_weights(min(i + 1, 90)) for i in range(min_length)])
    prophet_weights = np.array([w['prophet'] for w in weights_array])
    sarimax_weights = np.array([w['sarimax'] for w in weights_array])
    
    historical_ensemble = (prophet_weights * prophet_historical + 
                         sarimax_weights * sarimax_historical)
    
    return prophet_historical, sarimax_historical, historical_ensemble

# =======================
# Visualization Methods
# =======================
"""
Visualization components:
1. Interactive time series plot with:
   - Historical prices
   - Model predictions
   - Confidence intervals
2. Prediction table with 30-day forecast
3. Toggle functionality to switch between views
"""

def create_prediction_table(future_dates, ensemble_predictions):
    """Create a DataFrame with the predictions for display"""
    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in future_dates[:30]],
        'Price (P)': [f'{p:.2f}' for p in ensemble_predictions[:30]]
    })
    return df

def plot_results(data, prophet_forecast, sarimax_forecast, ensemble_predictions, 
                future_dates, ticker, historical_predictions=None):
    print("Generating plots...")
    
    # Create prediction table data
    pred_df = create_prediction_table(future_dates, ensemble_predictions)
    
    if (USE_BOKEH):
        output_file("stock_predictions.html")
        
        # Create the main plot
        p = figure(
            x_axis_type="datetime", 
            title=f"{ticker} Stock Price Predictions\nDynamic Ensemble with Historical Performance", 
            height=600,
            width=1000,
            sizing_mode="stretch_width"
        )
        p.grid.grid_line_alpha = 0.3
        
        # Historical prices
        p.line(data.index, data['Close'], color='blue', legend_label='Historical Prices')
        
        # Historical predictions (if available)
        if historical_predictions:
            prophet_hist, sarimax_hist, ensemble_hist = historical_predictions
            p.line(data.index, prophet_hist, color='green', line_dash='dotted', 
                   legend_label='Prophet Historical', alpha=0.3)
            p.line(data.index, sarimax_hist, color='purple', line_dash='dotted', 
                   legend_label='SARIMAX Historical', alpha=0.3)
            p.line(data.index, ensemble_hist, color='red', line_dash='dotted', 
                   legend_label='Ensemble Historical', alpha=0.3)
        
        # Future predictions
        p.line(future_dates, prophet_forecast['yhat'][-90:], 
               color='green', line_dash='dashed', legend_label='Prophet Forecast')
        p.line(future_dates, sarimax_forecast, 
               color='purple', line_dash='dashed', legend_label='SARIMAX Forecast')
        p.line(future_dates, ensemble_predictions, 
               color='red', line_width=2, legend_label='Dynamic Ensemble Forecast')
        
        # Confidence intervals
        source = ColumnDataSource(data=dict(
            x=future_dates,
            upper=prophet_forecast['yhat_upper'][-90:],
            lower=prophet_forecast['yhat_lower'][-90:]
        ))
        band = Band(base='x', upper='upper', lower='lower', source=source, level='underlay', 
                   fill_alpha=0.1, line_width=1, line_color='green')
        p.add_layout(band)
        
        p.legend.location = "top_left"
        p.xaxis.axis_label = 'Date'
        p.yaxis.axis_label = 'Stock Price (P)'
        
        # Create the table
        source = ColumnDataSource(pred_df)
        columns = [
            TableColumn(field="Date", title="Date"),
            TableColumn(field="Price (P)", title="Price (P)")
        ]
        data_table = DataTable(source=source, columns=columns, 
                             height=400, width=300,
                             index_position=None)
        
        # Create toggle button
        toggle = Toggle(label='Switch View', button_type='success', active=True)
        
        # Create JavaScript callback for toggle
        callback = CustomJS(args=dict(p=p, table=data_table), code="""
            if (cb_obj.active) {
                p.visible = true;
                table.visible = false;
            } else {
                p.visible = false;
                table.visible = true;
            }
        """)
        
        toggle.js_on_change('active', callback)
        
        # Initial visibility
        data_table.visible = False
        
        # Layout with toggle button
        layout_obj = Column(
            toggle,
            Row(p, data_table, sizing_mode="stretch_width"),
            sizing_mode="stretch_width"
        )
        
        show(layout_obj)
    else:
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplot for the main plot
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=4)
        
        # ... existing matplotlib plotting code for ax1 instead of plt ...
        ax1.plot(data.index, data['Close'], label='Historical Prices', color='blue', alpha=0.6)
        
        if historical_predictions:
            prophet_hist, sarimax_hist, ensemble_hist = historical_predictions
            ax1.plot(data.index, prophet_hist, color='green', linestyle=':', 
                    label='Prophet Historical', alpha=0.3)
            ax1.plot(data.index, sarimax_hist, color='purple', linestyle=':', 
                    label='SARIMAX Historical', alpha=0.3)
            ax1.plot(data.index, ensemble_hist, color='red', linestyle=':', 
                    label='Ensemble Historical', alpha=0.3)
        
        ax1.plot(future_dates, prophet_forecast['yhat'][-90:], 
                label='Prophet Forecast', color='green', linestyle='--', alpha=0.5)
        ax1.plot(future_dates, sarimax_forecast, 
                label='SARIMAX Forecast', color='purple', linestyle='--', alpha=0.5)
        ax1.plot(future_dates, ensemble_predictions, 
                label='Dynamic Ensemble Forecast', color='red', linewidth=2)
        ax1.fill_between(future_dates,
                        prophet_forecast['yhat_lower'][-90:],
                        prophet_forecast['yhat_upper'][-90:],
                        color='green', alpha=0.1, label='Prophet Confidence Interval')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Price (P)')
        ax1.set_title(f'{ticker} Stock Price Predictions\nDynamic Ensemble')
        ax1.legend()
        ax1.grid(True)
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # Create subplot for the table
        ax2 = plt.subplot2grid((1, 5), (0, 4))
        ax2.axis('off')
        table = ax2.table(
            cellText=pred_df.values,
            colLabels=pred_df.columns,
            cellLoc='right',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        plt.show()

# ======================
# Main Execution Flow
# ======================
"""
Program execution flow:
1. Data collection and preparation
2. Model training and prediction
3. Ensemble combination
4. Visualization and reporting
"""

# Main function to run the analysis
def main():
    ticker = 'RR.L'
    data = download_stock_data(ticker)
    data = data.asfreq('B')
    
    pe_ratio, analyst_target = download_additional_data(ticker)
    fear_greed_index = fetch_fear_greed_index()
    
    # Enhanced Prophet model with factor analysis
    prophet_data, indicator_weights, earnings_dates = prepare_prophet_data(
        data, pe_ratio, analyst_target, fear_greed_index
    )
    
    # Print upcoming earnings dates
    future_earnings = [d for d in earnings_dates if d > datetime.now()]
    if future_earnings:
        print("\nUpcoming Earnings Dates:")
        for date in future_earnings:
            print(f"  {date.strftime('%Y-%m-%d')}")
    
    prophet_model, prophet_forecast = fit_predict_prophet(prophet_data)
    
    # SARIMAX model
    print("Preparing SARIMAX model...")
    sarimax_model = SARIMAX(data['Close'],
                           order=(2, 1, 2),
                           seasonal_order=(1, 1, 1, 12),
                           freq='B')  # Specify frequency
    sarimax_results = sarimax_model.fit(disp=False)
    sarimax_forecast = sarimax_results.forecast(steps=90).values
    
    # Get historical predictions
    historical_predictions = get_historical_predictions(prophet_model, sarimax_results, data)
    
    # Calculate future dates and ensemble predictions
    future_dates = pd.date_range(start=datetime.today(), periods=90, freq='B')  # Add frequency
    ensemble_predictions = calculate_ensemble_predictions(prophet_forecast, sarimax_forecast, future_dates)
    
    # Plot results with historical predictions
    plot_results(data, prophet_forecast, sarimax_forecast, ensemble_predictions, 
                future_dates, ticker, historical_predictions)
    
    # Print predictions
    print("\n--- Future Price Predictions ---")
    print("\nShort-term Predictions (Days 1-30, SARIMAX weighted 80%):")
    for date, price in zip(future_dates[:30], ensemble_predictions[:30]):
        print(f"{date.strftime('%Y-%m-%d')}: {price:.2f} P")

if __name__ == "__main__":
    main()