"""Yahoo Finance data provider for price and volume data"""

from datetime import datetime
import polars as pl
import pandas as pd
import yfinance as yf
from loguru import logger

from .base import DataProvider, validate_date_range, validate_symbols


class YFinancePriceProvider(DataProvider):
    """
    Provider for OHLCV (price and volume) data from Yahoo Finance.

    Features:
    - Free and unlimited access
    - Historical data back to IPO for most stocks
    - Splits and dividends available
    - ~15 minute delay for real-time data

    Schema:
        - symbol: str - Ticker symbol
        - date: datetime - Trading date
        - open: float - Opening price (unadjusted)
        - high: float - High price (unadjusted)
        - low: float - Low price (unadjusted)
        - close: float - Closing price (unadjusted)
        - adj_close: float - Adjusted closing price (for splits & dividends)
        - volume: int - Trading volume
        - dividends: float - Dividend amount (if any)
        - stock_splits: float - Split ratio (if any)

    Note:
        Use 'adj_close' for calculating returns to account for splits/dividends.
        'close' is the raw closing price as reported on that day.
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize provider.

        Args:
            cache_enabled: Whether to use yfinance's built-in caching
        """
        self.cache_enabled = cache_enabled
        logger.info("YFinancePriceProvider initialized")

    def fetch(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pl.DataFrame:
        """
        Fetch OHLCV data for given symbols.

        Uses vectorized operations (pandas.stack()) instead of looping
        for better performance with multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            **kwargs: Additional arguments (ignored)

        Returns:
            Polars DataFrame with OHLCV data in long format

        Raises:
            ValueError: If inputs invalid
            ConnectionError: If yfinance unavailable
        """
        # Validate inputs
        validate_symbols(symbols)
        validate_date_range(start_date, end_date)

        logger.info(
            f"Fetching price data for {len(symbols)} symbols "
            f"from {start_date.date()} to {end_date.date()}"
        )

        try:
            # Use yfinance's batch download (handles both single and multiple symbols)
            raw_data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                auto_adjust=False,  # Get both close and adj_close
                actions=True,       # Include dividends and splits
                group_by='ticker',  # Group by ticker for multi-symbol
                progress=False
            )

            if raw_data.empty:
                logger.warning(f"No data returned for {symbols}")
                return pl.DataFrame()

            # Convert to standardized format (vectorized, no loops!)
            df = self._standardize_data(raw_data, symbols)

            # Validate results
            self.validate(df)

            logger.info(
                f"Successfully fetched {len(df)} rows for {len(symbols)} symbols"
            )
            return df

        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            raise

    def _standardize_data(
        self,
        raw_data: pd.DataFrame,
        symbols: list[str]
    ) -> pl.DataFrame:
        """
        Convert yfinance DataFrame to standardized Polars format.

        Handles both single-symbol and multi-symbol DataFrames from yf.download()
        using vectorized pandas operations (no Python loops!).

        Method:
        - For MultiIndex (multiple symbols): Use pandas.stack() to reshape
        - For single column (one symbol): Simple rename
        - This is ~10-100x faster than looping through symbols

        Args:
            raw_data: Raw DataFrame from yf.download()
            symbols: List of symbols that were requested

        Returns:
            Polars DataFrame in standardized long format with all symbols
        """
        # Handle MultiIndex columns (multiple symbols) vs regular columns (single symbol)
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Multiple symbols: Use stack() to convert to long format
            # stack() pivots the ticker level into rows - MUCH faster than looping
            df_long = raw_data.stack(level=0, future_stack=True).reset_index()
            # After stack: columns are ['Date', 'Ticker', 'Open', 'High', ...]
            df_long = df_long.rename(columns={'Ticker': 'symbol', 'Date': 'date'})
        else:
            # Single symbol: Just reset index and add symbol column
            df_long = raw_data.reset_index()
            df_long['symbol'] = symbols[0]
            df_long = df_long.rename(columns={'Date': 'date'})

        # Standardize column names (yfinance uses title case)
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'stock_splits',
        }
        df_long = df_long.rename(columns=column_mapping)

        # Ensure all required columns exist (fill missing with defaults)
        if 'dividends' not in df_long.columns:
            df_long['dividends'] = 0.0
        if 'stock_splits' not in df_long.columns:
            df_long['stock_splits'] = 0.0

        # Select and order columns
        df_long = df_long[[
            'symbol',
            'date',
            'open',
            'high',
            'low',
            'close',
            'adj_close',
            'volume',
            'dividends',
            'stock_splits',
        ]]

        # Convert to Polars (much faster for subsequent operations)
        df_polars = pl.from_pandas(df_long)

        # Cast date to pl.Date (daily data doesn't need time component)
        df_polars = df_polars.with_columns([
            pl.col('date').cast(pl.Date)
        ])

        return df_polars

    def validate(self, df: pl.DataFrame) -> bool:
        """
        Validate OHLCV data quality.

        Checks:
        - Required columns present
        - No negative prices
        - High >= Low
        - Volume >= 0
        - No duplicate (symbol, date) pairs

        Args:
            df: DataFrame to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if df.is_empty():
            logger.warning("Empty DataFrame - skipping validation")
            return True

        # Check required columns
        required_cols = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for negative prices
        price_cols = ["open", "high", "low", "close", "adj_close"]
        for col in price_cols:
            if (df[col] < 0).any():
                raise ValueError(f"Found negative values in {col}")

        # Check high >= low
        if (df["high"] < df["low"]).any():
            raise ValueError("Found rows where high < low")

        # Check non-negative volume
        if (df["volume"] < 0).any():
            raise ValueError("Found negative volume")

        # Check for duplicates
        duplicates = df.group_by(["symbol", "date"]).agg(pl.len()).filter(pl.col("len") > 1)
        if len(duplicates) > 0:
            raise ValueError(f"Found {len(duplicates)} duplicate (symbol, date) pairs")

        logger.debug("Data validation passed")
        return True

    def get_available_symbols(self) -> list[str]:
        """
        Get available symbols.

        Note: Yahoo Finance doesn't provide an API for listing all symbols.
        Return empty list.
        """
        return []


# Example usage and testing
if __name__ == "__main__":
    from datetime import timedelta

    # Configure logging
    logger.add("yfinance_provider.log", rotation="10 MB")

    # Create provider
    provider = YFinancePriceProvider()

    # Test 1: Single stock
    print("=" * 60)
    print("Test 1: Single symbol (AAPL)")
    print("=" * 60)
    symbols = ["AAPL"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    df = provider.fetch(symbols, start_date, end_date)
    print(f"\nFetched {len(df)} rows")
    print(f"Columns: {df.columns}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Test 2: Multiple stocks
    print("\n" + "=" * 60)
    print("Test 2: Multiple symbols (AAPL, MSFT, GOOGL)")
    print("=" * 60)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    df = provider.fetch(symbols, start_date, end_date)
    print(f"\nFetched {len(df)} rows for {symbols}")
    print(f"\nRows per symbol:")
    print(df.group_by("symbol").agg(pl.len()))
    print(f"\nSample data:")
    print(df.head(10))
