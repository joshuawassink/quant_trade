"""
Script to fetch sample data for testing and initial model development.

Fetches 3 years of historical price data for a small universe of stocks.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.providers.yfinance_provider import YFinancePriceProvider
from loguru import logger

# Configure logging
logger.add("logs/fetch_sample_data.log", rotation="10 MB")


def main():
    """Fetch sample data for testing"""

    # Small but diverse universe for testing
    # Include different sectors, market caps, and characteristics
    universe = [
        # Large cap tech
        "AAPL", "MSFT", "GOOGL", "META", "NVDA",
        # Large cap other sectors
        "JPM", "JNJ", "PG", "XOM", "WMT",
        # Mid cap
        "SNAP", "DDOG", "MDB", "CRWD", "NET",
        # Different sectors
        "BA", "DIS", "NKE", "TSLA", "NFLX",
    ]

    logger.info(f"Fetching data for {len(universe)} stocks")

    # Date range: 3 years of history
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    # Create provider
    provider = YFinancePriceProvider()

    # Fetch data
    logger.info("Starting data fetch...")
    df = provider.fetch(universe, start_date, end_date)

    # Summary statistics
    logger.info(f"Fetched {len(df)} total rows")
    logger.info(f"Date range in data: {df['date'].min()} to {df['date'].max()}")

    # Rows per symbol
    rows_per_symbol = df.group_by("symbol").agg(pl.len()).sort("len", descending=True)
    logger.info(f"Rows per symbol:\n{rows_per_symbol}")

    # Save to parquet
    output_dir = Path("data/price/daily")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"sample_universe_{start_date.date()}_to_{end_date.date()}.parquet"
    df.write_parquet(output_file)

    logger.info(f"Data saved to {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Also save individual files per symbol (for easy access)
    for symbol in universe:
        symbol_data = df.filter(pl.col("symbol") == symbol)
        symbol_file = output_dir / f"{symbol}.parquet"
        symbol_data.write_parquet(symbol_file)

    logger.info(f"Individual symbol files saved to {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATA FETCH COMPLETE")
    print("=" * 60)
    print(f"Universe: {len(universe)} stocks")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to: {output_file}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    import polars as pl
    df = main()

    # Show sample
    print("\nSample of data:")
    print(df.head(10))

