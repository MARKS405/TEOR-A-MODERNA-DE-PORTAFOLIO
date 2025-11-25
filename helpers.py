import numpy as np
import pandas as pd

def returns_from_prices(prices: pd.DataFrame):
    # simple log returns
    return np.log(prices / prices.shift(1)).dropna()

def load_sample_data(n_assets=6, n_obs=500, seed=42):
    np.random.seed(seed)
    # simulate correlated returns and build price series
    mu = np.random.uniform(0.0002, 0.001, size=n_assets)  # daily drift
    A = np.random.normal(size=(n_assets, n_assets))
    cov = A @ A.T * 1e-5
    rets = np.random.multivariate_normal(mu, cov, size=n_obs)
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"Asset_{i+1}" for i in range(n_assets)]
    df = pd.DataFrame(prices, columns=cols)
    df.index = pd.date_range(end=pd.Timestamp.today(), periods=n_obs, freq='B')
    return df
