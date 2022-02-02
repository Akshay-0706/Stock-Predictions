import logging
import streamlit as st
from datetime import date

import yfinance as yf
import numpy as np
import pandas as pd
from collections import OrderedDict
from fbprophet import Prophet
from models import CmdStanPyBackend, IStanBackend, PyStanBackend, StanBackendEnum
from plotly import graph_objs as go

from enum import Enum

logger = logging.getLogger('fbprophet')
logger.setLevel(logging.INFO)




class StanBackendEnum(Enum):
    PYSTAN = PyStanBackend
    CMDSTANPY = CmdStanPyBackend

    @staticmethod
    def get_backend_class(name: str) -> IStanBackend:
        try:
            return StanBackendEnum[name].value
        except KeyError as e:
            raise ValueError("Unknown stan backend: {}".format(name)) from e



def plot_plotly(m, fcst, uncertainty=True, plot_cap=True, trend=False, changepoints=False,
                changepoints_threshold=0.01, xlabel='ds', ylabel='y', figsize=(900, 600)):

    prediction_color = '#0072B2'
    error_color = 'rgba(0, 114, 178, 0.2)'  # '#0072B2' with 0.2 opacity
    actual_color = 'black'
    cap_color = 'black'
    trend_color = '#B23B00'
    line_width = 2
    marker_size = 4

    data = []
    # Add actual
    data.append(go.Scatter(
        name='Actual',
        x=m.history['ds'],
        y=m.history['y'],
        marker=dict(color=actual_color, size=marker_size),
        mode='markers'
    ))
    # Add lower bound
    if uncertainty and m.uncertainty_samples:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip'
        ))
    # Add prediction
    data.append(go.Scatter(
        name='Predicted',
        x=fcst['ds'],
        y=fcst['yhat'],
        mode='lines',
        line=dict(color=prediction_color, width=line_width),
        fillcolor=error_color,
        fill='tonexty' if uncertainty and m.uncertainty_samples else 'none'
    ))
    # Add upper bound
    if uncertainty and m.uncertainty_samples:
        data.append(go.Scatter(
            x=fcst['ds'],
            y=fcst['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            fillcolor=error_color,
            fill='tonexty',
            hoverinfo='skip'
        ))
    # Add caps
    if 'cap' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Cap',
            x=fcst['ds'],
            y=fcst['cap'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    if m.logistic_floor and 'floor' in fcst and plot_cap:
        data.append(go.Scatter(
            name='Floor',
            x=fcst['ds'],
            y=fcst['floor'],
            mode='lines',
            line=dict(color=cap_color, dash='dash', width=line_width),
        ))
    # Add trend
    if trend:
        data.append(go.Scatter(
            name='Trend',
            x=fcst['ds'],
            y=fcst['trend'],
            mode='lines',
            line=dict(color=trend_color, width=line_width),
        ))
    # Add changepoints
    if changepoints and len(m.changepoints) > 0:
        signif_changepoints = m.changepoints[
            np.abs(np.nanmean(m.params['delta'], axis=0)) >= changepoints_threshold
        ]
        data.append(go.Scatter(
            x=signif_changepoints,
            y=fcst.loc[fcst['ds'].isin(signif_changepoints), 'trend'],
            marker=dict(size=50, symbol='line-ns-open', color=trend_color,
                        line=dict(width=line_width)),
            mode='markers',
            hoverinfo='skip'
        ))

    layout = dict(
        showlegend=False,
        width=figsize[0],
        height=figsize[1],
        yaxis=dict(
            title=ylabel
        ),
        xaxis=dict(
            title=xlabel,
            type='date',
            rangeselector=dict(
                buttons=list([
                    dict(count=7,
                         label='1w',
                         step='day',
                         stepmode='backward'),
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='6m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                         label='1y',
                         step='year',
                         stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    return fig





class Prophet(object):
    def __init__(
            self,
            growth='linear',
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            holidays=None,
            seasonality_mode='additive',
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
            stan_backend=None
    ):
        self.growth = growth

        self.changepoints = changepoints
        if self.changepoints is not None:
            self.changepoints = pd.Series(pd.to_datetime(self.changepoints), name='ds')
            self.n_changepoints = len(self.changepoints)
            self.specified_changepoints = True
        else:
            self.n_changepoints = n_changepoints
            self.specified_changepoints = False

        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays

        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.holidays_prior_scale = float(holidays_prior_scale)

        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples

        # Set during fitting or by other methods
        self.start = None
        self.y_scale = None
        self.logistic_floor = False
        self.t_scale = None
        self.changepoints_t = None
        self.seasonalities = OrderedDict({})
        self.extra_regressors = OrderedDict({})
        self.country_holidays = None
        self.stan_fit = None
        self.params = {}
        self.history = None
        self.history_dates = None
        self.train_component_cols = None
        self.component_modes = None
        self.train_holiday_names = None
        self.fit_kwargs = {}
        # self.validate_inputs()
        self._load_stan_backend(stan_backend)

    def _load_stan_backend(self, stan_backend):
        if stan_backend is None:
            for i in StanBackendEnum:
                try:
                    logger.debug("Trying to load backend: %s", i.name)
                    return self._load_stan_backend(i.name)
                except Exception as e:
                    logger.debug("Unable to load backend %s (%s), trying the next one", i.name, e)
        else:
            self.stan_backend = StanBackendEnum.get_backend_class(stan_backend)()

        logger.debug("Loaded stan backend: %s", self.stan_backend.get_type())




    def fit(self, df, **kwargs):
       
        if self.history is not None:
            raise Exception('Prophet object can only be fit once. '
                            'Instantiate a new object.')
        if ('ds' not in df) or ('y' not in df):
            raise ValueError(
                'Dataframe must have columns "ds" and "y" with the dates and '
                'values respectively.'
            )
        history = df[df['y'].notnull()].copy()
        if history.shape[0] < 2:
            raise ValueError('Dataframe has less than 2 non-NaN rows.')
        self.history_dates = pd.to_datetime(pd.Series(df['ds'].unique(), name='ds')).sort_values()

        history = self.setup_dataframe(history, initialize_scales=True)
        self.history = history
        self.set_auto_seasonalities()
        seasonal_features, prior_scales, component_cols, modes = (
            self.make_all_seasonality_features(history))
        self.train_component_cols = component_cols
        self.component_modes = modes
        self.fit_kwargs = deepcopy(kwargs)

        self.set_changepoints()

        trend_indicator = {'linear': 0, 'logistic': 1, 'flat': 2}

        dat = {
            'T': history.shape[0],
            'K': seasonal_features.shape[1],
            'S': len(self.changepoints_t),
            'y': history['y_scaled'],
            't': history['t'],
            't_change': self.changepoints_t,
            'X': seasonal_features,
            'sigmas': prior_scales,
            'tau': self.changepoint_prior_scale,
            'trend_indicator': trend_indicator[self.growth],
            's_a': component_cols['additive_terms'],
            's_m': component_cols['multiplicative_terms'],
        }

        if self.growth == 'linear':
            dat['cap'] = np.zeros(self.history.shape[0])
            kinit = self.linear_growth_init(history)
        elif self.growth == 'flat':
            dat['cap'] = np.zeros(self.history.shape[0])
            kinit = self.flat_growth_init(history)
        else:
            dat['cap'] = history['cap_scaled']
            kinit = self.logistic_growth_init(history)

        stan_init = {
            'k': kinit[0],
            'm': kinit[1],
            'delta': np.zeros(len(self.changepoints_t)),
            'beta': np.zeros(seasonal_features.shape[1]),
            'sigma_obs': 1,
        }

        if history['y'].min() == history['y'].max() and \
                (self.growth == 'linear' or self.growth == 'flat'):
            self.params = stan_init
            self.params['sigma_obs'] = 1e-9
            for par in self.params:
                self.params[par] = np.array([self.params[par]])
        elif self.mcmc_samples > 0:
            self.params = self.stan_backend.sampling(stan_init, dat, self.mcmc_samples, **kwargs)
        else:
            self.params = self.stan_backend.fit(stan_init, dat, **kwargs)

        # If no changepoints were requested, replace delta with 0s
        if len(self.changepoints) == 0:
            # Fold delta into the base rate k
            self.params['k'] = (self.params['k']
                                + self.params['delta'].reshape(-1))
            self.params['delta'] = (np.zeros(self.params['delta'].shape)
                                      .reshape((-1, 1)))

        return self





START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(layout="wide")

st.title('Stock Forecast App')

selected_stock = st.text_input('Select dataset for prediction', 'GOOG')

n_years = st.slider('Days of prediction:', 100, 500)
period = n_years 


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Range Slider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast

st.subheader('Forecast data')
st.write(forecast.tail())


st.subheader(f'Forecast plot for {n_years} days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


