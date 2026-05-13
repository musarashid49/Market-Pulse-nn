"""
src/features/technical.py
==========================
Technical indicator computation using the `ta` library.

All indicators are added as new columns to the existing OHLCV DataFrame.
NaN rows introduced by rolling windows are forward-filled where safe,
then any remaining NaNs at the start of the series are dropped.

Indicators computed:
  RSI          : momentum oscillator [0, 100]
  MACD + Signal: trend / momentum
  Bollinger %B : price position within the bands [0, 1] where 0.5 = mid
  Bollinger Width: normalised band width (volatility proxy)
  ATR           : Average True Range (volatility)
  Volume MA     : 20-day moving average of volume
  Volume ratio  : today's volume / 20-day MA (surge indicator)

Usage:
    from src.features import TechnicalFeatures
    eng = TechnicalFeatures(config=CFG)
    df_with_indicators = eng.compute(price_df)
"""

import numpy as np
import pandas as pd
from ..utils.helpers import get_logger

logger = get_logger(__name__)


class TechnicalFeatures:
    """
    Adds technical indicators to a single-ticker OHLCV DataFrame.

    Parameters pulled from CFG:
      RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
      BB_PERIOD, BB_STD, ATR_PERIOD
    """

    def __init__(self, config) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    def compute(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Computes all technical indicators and appends them as columns.

        Parameters
        ----------
        df      : OHLCV DataFrame with columns: date, open, high, low, close, volume
        drop_na : drop rows with NaN indicators (first ~max_period rows)

        Returns
        -------
        DataFrame with indicator columns appended
        """
        df = df.copy().sort_values("date").reset_index(drop=True)

        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger(df)
        df = self._add_atr(df)
        df = self._add_volume_features(df)
        df = self._add_price_lags(df)

        if drop_na:
            before = len(df)
            df = df.dropna().reset_index(drop=True)
            logger.info(f"Dropped {before - len(df)} NaN rows after indicator computation.")

        logger.info(f"Technical features computed | {len(df):,} rows, {df.shape[1]} cols")
        return df

    # ------------------------------------------------------------------
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index -- momentum oscillator [0, 100]."""
        period = self.cfg.RSI_PERIOD
        delta  = df["close"].diff()
        gain   = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss   = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs     = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    # ------------------------------------------------------------------
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD = EMA(fast) - EMA(slow).
        Signal = EMA(MACD, signal_period).
        Histogram = MACD - Signal (crossover signal).
        """
        fast   = self.cfg.MACD_FAST
        slow   = self.cfg.MACD_SLOW
        signal = self.cfg.MACD_SIGNAL

        ema_fast         = df["close"].ewm(span=fast,   adjust=False).mean()
        ema_slow         = df["close"].ewm(span=slow,   adjust=False).mean()
        df["macd"]       = ema_fast - ema_slow
        df["macd_signal"]= df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"]  = df["macd"] - df["macd_signal"]
        return df

    # ------------------------------------------------------------------
    def _add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands.
        %B = (close - lower) / (upper - lower)  -> position within bands
        Width = (upper - lower) / mid            -> normalised band width
        """
        period = self.cfg.BB_PERIOD
        std    = self.cfg.BB_STD

        mid              = df["close"].rolling(period).mean()
        rolling_std      = df["close"].rolling(period).std()
        upper            = mid + std * rolling_std
        lower            = mid - std * rolling_std
        band_width       = upper - lower

        df["bb_pct_b"]   = (df["close"] - lower) / band_width.replace(0, np.nan)
        df["bb_width"]   = band_width / mid.replace(0, np.nan)   # normalised
        return df

    # ------------------------------------------------------------------
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Average True Range -- absolute volatility measure.
        Normalised ATR = ATR / close (relative volatility).
        """
        period  = self.cfg.ATR_PERIOD
        high    = df["high"]
        low     = df["low"]
        close   = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low  - close.shift()).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["atr"]     = tr.ewm(com=period - 1, adjust=False).mean()
        df["atr_pct"] = df["atr"] / close   # normalised ATR
        return df

    # ------------------------------------------------------------------
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume MA and surge ratio."""
        window               = 20
        vol_ma               = df["volume"].rolling(window).mean()
        df["volume_ma20"]    = vol_ma
        df["volume_ratio"]   = df["volume"] / vol_ma.replace(0, np.nan)
        df["log_volume"]     = np.log1p(df["volume"])
        return df

    # ------------------------------------------------------------------
    def _add_price_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged log-return features (provide temporal context to the model)."""
        for lag in self.cfg.PRICE_LAGS:
            df[f"log_return_lag{lag}"] = df["log_return"].shift(lag)
        return df
