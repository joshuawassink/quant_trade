"""
YFinance Market & Sector Data Provider

Fetches market-wide indicators and sector performance data.
Provides context about overall market conditions and sector trends.

Data Categories:
1. Sector ETFs - Performance of 11 major sectors (XLK, XLF, etc.)
2. Market Benchmarks - Major indices (S&P 500, NASDAQ, Dow)
3. Volatility Indicators - VIX, VXN
4. Interest Rates - Treasury yields (10Y, 30Y, etc.)

This data helps identify:
- Sector rotation trends
- Market regime changes (bull/bear/volatile)
- Interest rate environment
- Relative sector strength
"""

from datetime import datetime
from loguru import logger
import pandas as pd
import polars as pl
import yfinance as yf

from .base import DataProvider


class YFinanceMarketProvider(DataProvider):
    """Provider for market-wide and sector data from Yahoo Finance.

    Fetches time-series data for:
    - 11 SPDR Sector ETFs (XLK, XLF, XLV, etc.)
    - 5 Market Benchmarks (SPY, S&P 500, NASDAQ, Dow, QQQ)
    - 2 Volatility Indicators (VIX, VXN)
    - 5 Interest Rate Instruments (10Y, 30Y yields, Treasury ETFs)

    Schema (daily time-series):
    - symbol: Ticker symbol
    - date: Trading date
    - category: 'sector_etf', 'market_benchmark', 'volatility', 'interest_rate'
    - name: Human-readable name
    - open: Opening price
    - high: Highest price
    - low: Lowest price
    - close: Closing price
    - volume: Trading volume (0 for indices)

    Total of 23 symbols tracked daily.
    """

    # Define instrument universes
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communication Services',
        'XLU': 'Utilities'
    }

    MARKET_BENCHMARKS = {
        'SPY': 'S&P 500 ETF',
        '^GSPC': 'S&P 500 Index',
        '^DJI': 'Dow Jones Industrial',
        '^IXIC': 'NASDAQ Composite',
        'QQQ': 'NASDAQ 100 ETF'
    }

    VOLATILITY_INDICATORS = {
        '^VIX': 'CBOE Volatility Index',
        '^VXN': 'NASDAQ Volatility Index'
    }

    INTEREST_RATES = {
        '^TNX': '10-Year Treasury Yield',
        '^IRX': '13-Week Treasury Bill',
        '^TYX': '30-Year Treasury Yield',
        'TLT': '20+ Year Treasury Bond ETF',
        'SHY': '1-3 Year Treasury Bond ETF'
    }

    def __init__(self):
        # Combine all instruments
        self.instruments = {
            **{k: ('sector_etf', v) for k, v in self.SECTOR_ETFS.items()},
            **{k: ('market_benchmark', v) for k, v in self.MARKET_BENCHMARKS.items()},
            **{k: ('volatility', v) for k, v in self.VOLATILITY_INDICATORS.items()},
            **{k: ('interest_rate', v) for k, v in self.INTEREST_RATES.items()}
        }
        logger.info(f"YFinanceMarketProvider initialized with {len(self.instruments)} instruments")

    def fetch(
        self,
        symbols: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Fetch market and sector data.

        Args:
            symbols: List of symbols to fetch (if None, fetches all instruments)
            start_date: Start date for historical data
            end_date: End date for historical data
            **kwargs: Additional arguments (unused)

        Returns:
            Polars DataFrame with market/sector time-series data
        """
        # If no symbols specified, fetch all
        if symbols is None:
            symbols = list(self.instruments.keys())
        else:
            # Filter to only valid instruments
            symbols = [s for s in symbols if s in self.instruments]

        logger.info(f"Fetching market data for {len(symbols)} instruments")

        # Convert dates to strings for yfinance
        start_str = start_date.strftime('%Y-%m-%d') if start_date else None
        end_str = end_date.strftime('%Y-%m-%d') if end_date else None

        all_data = []

        for symbol in symbols:
            try:
                category, name = self.instruments[symbol]

                # Fetch data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_str, end=end_str, actions=False)

                if hist.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Convert to long format
                hist_reset = hist.reset_index()
                hist_reset['symbol'] = symbol
                hist_reset['category'] = category
                hist_reset['name'] = name

                # Rename columns to match our schema
                hist_reset = hist_reset.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })

                # Select only needed columns
                hist_reset = hist_reset[['symbol', 'date', 'category', 'name',
                                        'open', 'high', 'low', 'close', 'volume']]

                all_data.append(hist_reset)
                logger.debug(f"Fetched {len(hist_reset)} records for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

        if not all_data:
            logger.warning("No market data fetched")
            return pl.DataFrame()

        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)

        # Convert to Polars
        df = pl.from_pandas(combined)

        # Validate
        if not self.validate(df):
            logger.error("Market data validation failed")
            return pl.DataFrame()

        logger.info(f"Successfully fetched market data: {len(df)} records, {df['symbol'].n_unique()} instruments")
        return df

    def validate(self, df: pl.DataFrame) -> bool:
        """
        Validate market data.

        Checks:
        - Required columns present
        - No empty dataframe
        - Valid date ranges
        - Reasonable price values
        """
        if df.is_empty():
            logger.error("Market DataFrame is empty")
            return False

        required_cols = ['symbol', 'date', 'category', 'name', 'close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False

        # Check for valid closes
        if df['close'].is_null().all():
            logger.error("All close prices are null")
            return False

        logger.debug("Market data validation passed")
        return True


# Test function
if __name__ == "__main__":
    from datetime import timedelta

    provider = YFinanceMarketProvider()

    # Test with recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Last 30 days

    print(f"Testing market provider...")
    print(f"Date range: {start_date.date()} to {end_date.date()}\n")

    df = provider.fetch(start_date=start_date, end_date=end_date)

    if not df.is_empty():
        print(f"✓ Success! Fetched {len(df)} records")
        print(f"  Instruments: {df['symbol'].n_unique()}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Categories: {df['category'].unique().to_list()}")

        # Show breakdown by category
        print(f"\n  Breakdown by category:")
        category_counts = df.group_by('category').agg([
            pl.len().alias('records'),
            pl.col('symbol').n_unique().alias('instruments')
        ]).sort('category')
        print(category_counts)

        # Show sample data
        print(f"\n  Sample (recent prices):")
        sample = df.filter(pl.col('date') == df['date'].max()).select([
            'symbol', 'name', 'category', 'close'
        ]).sort('category', 'symbol').head(10)
        print(sample)
    else:
        print("✗ Failed to fetch data")
