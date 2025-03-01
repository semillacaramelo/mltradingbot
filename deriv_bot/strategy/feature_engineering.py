"""
Module for creating advanced technical indicators and features for ML model
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.market_regimes = None

    def calculate_features(self, df):
        """
        Calculate technical indicators and features

        Args:
            df: DataFrame with OHLCV data
        """
        try:
            # Ensure DataFrame has required columns
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Available columns: {df.columns.tolist()}")
                return None

            logger.info("Starting feature calculation")

            # Add momentum indicators
            df = self._add_momentum_indicators(df)
            if df is None:
                return None

            # Add volatility indicators
            df = self._add_volatility_indicators(df)
            if df is None:
                return None

            # Add trend indicators
            df = self._add_trend_indicators(df)
            if df is None:
                return None

            # Add market regime features
            df = self._add_market_regime(df)
            if df is None:
                return None

            # Add price pattern features
            df = self._add_price_patterns(df)
            if df is None:
                return None

            # Drop NaN values from calculations
            df.dropna(inplace=True)

            logger.info(f"Feature calculation completed. Final shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            return None

    def _add_momentum_indicators(self, df):
        """Calculate momentum-based indicators"""
        try:
            # RSI with multiple periods
            for period in [9, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

            # Enhanced MACD
            for (fast, slow, signal) in [(12, 26, 9), (5, 35, 5)]:
                exp1 = df['close'].ewm(span=fast, adjust=False).mean()
                exp2 = df['close'].ewm(span=slow, adjust=False).mean()
                macd = exp1 - exp2
                df[f'MACD_{fast}_{slow}'] = macd
                df[f'MACD_Signal_{fast}_{slow}'] = macd.ewm(span=signal, adjust=False).mean()

            # Stochastic RSI
            for period in [14, 21]:
                if f'RSI_{period}' in df.columns:
                    stoch_k = df[f'RSI_{period}'].rolling(window=14).apply(
                        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) * 100
                        if x.max() != x.min() else 50
                    )
                    df[f'StochRSI_{period}'] = stoch_k.rolling(window=3).mean()

            return df

        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return None

    def _add_volatility_indicators(self, df):
        """Calculate volatility-based indicators"""
        try:
            # Enhanced Bollinger Bands
            for period in [20, 50]:
                df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                df[f'BB_Upper_{period}'] = df[f'SMA_{period}'] + (std * 2)
                df[f'BB_Lower_{period}'] = df[f'SMA_{period}'] - (std * 2)
                df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'SMA_{period}']

            # Average True Range (ATR) with multiple periods
            for period in [14, 21]:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                df[f'ATR_{period}'] = true_range.rolling(window=period).mean()

            # Volatility ratio
            if 'ATR_14' in df.columns and 'ATR_21' in df.columns:
                df['Volatility_Ratio'] = df['ATR_14'] / df['ATR_21']

            return df

        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            return None

    def _add_trend_indicators(self, df):
        """Calculate trend-based indicators"""
        try:
            # Multiple timeframe moving averages
            for period in [10, 20, 50, 100]:
                df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

            # Add moving average crossovers
            for period in [10, 20, 50]:
                if f'SMA_{period}' in df.columns and f'SMA_{period * 2}' in df.columns:
                    df[f'MA_Cross_{period}'] = np.where(
                        df[f'SMA_{period}'] > df[f'SMA_{period * 2}'], 1, -1
                    )

            # Triple moving average crossover
            if all(f'SMA_{p}' in df.columns for p in [10, 20, 50]):
                df['Triple_MA_Cross'] = np.where(
                    (df['SMA_10'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_50']), 1,
                    np.where((df['SMA_10'] < df['SMA_20']) & (df['SMA_20'] < df['SMA_50']), -1, 0)
                )

            # Price Rate of Change for multiple periods
            for period in [5, 10, 20]:
                df[f'ROC_{period}'] = df['close'].pct_change(periods=period) * 100

            return df

        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
            return None

    def _add_market_regime(self, df):
        """Identify market regimes using clustering"""
        try:
            # Calculate features for regime classification
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            trend = df['close'].rolling(window=20).mean().pct_change()

            # Clean and prepare features for clustering
            features = np.column_stack([
                returns.fillna(0),
                volatility.fillna(0),
                trend.fillna(0)
            ])

            # Apply K-means clustering
            n_regimes = 4  # Identify 4 market regimes
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            df['Market_Regime'] = kmeans.fit_predict(features)

            # Store clustering model for future use
            self.market_regimes = kmeans

            return df

        except Exception as e:
            logger.error(f"Error calculating market regime: {str(e)}")
            return df

    def _add_price_patterns(self, df):
        """Identify price patterns and candlestick patterns"""
        try:
            # Calculate candlestick body and shadows
            df['Body'] = df['close'] - df['open']
            df['Upper_Shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['Lower_Shadow'] = df[['open', 'close']].min(axis=1) - df['low']

            # Identify doji patterns
            df['Doji'] = (abs(df['Body']) <= 0.1 * (df['high'] - df['low'])).astype(int)

            # Identify hammer patterns
            df['Hammer'] = (
                (df['Lower_Shadow'] > 2 * abs(df['Body'])) &
                (df['Upper_Shadow'] <= 0.1 * abs(df['Body']))
            ).astype(int)

            # Identify shooting star patterns
            df['Shooting_Star'] = (
                (df['Upper_Shadow'] > 2 * abs(df['Body'])) &
                (df['Lower_Shadow'] <= 0.1 * abs(df['Body']))
            ).astype(int)

            return df

        except Exception as e:
            logger.error(f"Error calculating price patterns: {str(e)}")
            return df