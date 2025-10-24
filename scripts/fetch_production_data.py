"""
Script to fetch production data for the full 377-stock universe.

Fetches 3 years of historical price data for production model training.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.data.providers.yfinance_provider import YFinancePriceProvider
from src.shared.config.universe import get_universe
from loguru import logger

# Configure logging
logger.add("logs/fetch_production_data.log", rotation="10 MB")


def main():
    """Fetch production data for 377-stock universe"""

    # Use production universe (377 stocks)
    universe = get_universe('production')

    logger.info(f"Fetching data for {len(universe)} stocks")
    logger.info(f"Universe: {universe[:10]}... (showing first 10)")

    # Date range: 3 years of history (matches what we validated in universe generation)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    # Create provider
    provider = YFinancePriceProvider()

    # Fetch data
    logger.info("Starting data fetch... (this will take several minutes)")
    df = provider.fetch(universe, start_date, end_date)

    # Summary statistics
    logger.info(f"Fetched {len(df)} total rows")
    logger.info(f"Date range in data: {df['date'].min()} to {df['date'].max()}")

    # Rows per symbol statistics
    rows_per_symbol = df.group_by("symbol").agg(pl.len().alias("rows")).sort("rows", descending=True)
    logger.info(f"Symbols with data: {len(rows_per_symbol)}")
    logger.info(f"Average rows per symbol: {rows_per_symbol['rows'].mean():.0f}")
    logger.info(f"Min rows: {rows_per_symbol['rows'].min()}")
    logger.info(f"Max rows: {rows_per_symbol['rows'].max()}")

    # Check for missing symbols
    fetched_symbols = set(df['symbol'].unique().to_list())
    requested_symbols = set(universe)
    missing_symbols = requested_symbols - fetched_symbols
    if missing_symbols:
        logger.warning(f"Missing data for {len(missing_symbols)} symbols: {sorted(missing_symbols)}")

    # Save to parquet
    output_dir = Path("data/price/daily")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual files per symbol (for easy access and consistency with existing setup)
    logger.info("Saving individual symbol files...")
    for symbol in fetched_symbols:
        symbol_data = df.filter(pl.col("symbol") == symbol)
        symbol_file = output_dir / f"{symbol}.parquet"
        symbol_data.write_parquet(symbol_file)

    logger.info(f"Individual symbol files saved to {output_dir}")

    # Also save combined file for convenience
    combined_file = output_dir / f"production_universe_{start_date.date()}_to_{end_date.date()}.parquet"
    df.write_parquet(combined_file)
    logger.info(f"Combined data saved to {combined_file}")
    logger.info(f"File size: {combined_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Print summary
    print("\n" + "=" * 70)
    print("PRODUCTION DATA FETCH COMPLETE")
    print("=" * 70)
    print(f"Universe: {len(universe)} stocks requested")
    print(f"Fetched: {len(fetched_symbols)} stocks with data")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to: {output_dir}")
    if missing_symbols:
        print(f"\n⚠  Missing {len(missing_symbols)} symbols: {sorted(list(missing_symbols))[:10]}...")
    print("=" * 70)

    return df


if __name__ == "__main__":
    df = main()

    # Show sample
    print("\nSample of data:")
    print(df.head(10))
