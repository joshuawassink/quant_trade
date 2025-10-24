"""
Fetch quarterly financial statements for the stock universe.

Retrieves:
- Income statement metrics (revenue, profits, margins)
- Balance sheet metrics (assets, liabilities, equity)
- Cashflow metrics (OCF, FCF, capex)
- Computed financial ratios (ROE, ROA, margins, etc.)

Data saved to: data/financials/quarterly_financials_YYYYMMDD.parquet
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
from src.shared.data.providers.yfinance_financials_provider import YFinanceFinancialsProvider
from src.shared.config.universe import get_universe

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def main():
    """Fetch quarterly financials for the stock universe."""

    # Use standardized universe configuration
    # Change to 'production' to fetch for full 377-stock universe
    universe_name = 'production'  # or 'sample' for testing
    universe = get_universe(universe_name)
    logger.info(f"Using '{universe_name}' universe")

    logger.info(f"Fetching quarterly financials for {len(universe)} stocks")

    # Initialize provider
    provider = YFinanceFinancialsProvider()

    # Fetch data
    logger.info("Starting financials fetch...")
    df = provider.fetch(universe)

    if df.is_empty():
        logger.error("No data fetched")
        return

    logger.info(f"Fetched {len(df)} quarterly records for {df['symbol'].n_unique()} symbols")

    # Show statistics
    logger.info(f"Date range: {df['quarter_end_date'].min()} to {df['quarter_end_date'].max()}")
    logger.info(f"Average quarters per symbol: {len(df) / df['symbol'].n_unique():.1f}")

    # Show data quality
    logger.info("\nData completeness:")
    key_metrics = ['total_revenue', 'net_income', 'roe', 'roa', 'gross_margin', 'net_margin']
    for metric in key_metrics:
        non_null = df[metric].is_not_null().sum()
        pct = non_null / len(df) * 100
        logger.info(f"  {metric}: {non_null}/{len(df)} ({pct:.1f}%)")

    # Save data
    output_dir = project_root / 'data' / 'financials'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save with date stamp
    date_str = datetime.now().strftime('%Y%m%d')
    output_file = output_dir / f'quarterly_financials_{date_str}.parquet'
    df.write_parquet(output_file)
    logger.info(f"\nFinancials saved to {output_file}")

    # Also save as "latest"
    latest_file = output_dir / 'quarterly_financials_latest.parquet'
    df.write_parquet(latest_file)
    logger.info(f"Also saved as {latest_file}")

    # Show sample for a few companies
    print("\n" + "="*60)
    print("QUARTERLY FINANCIALS FETCH COMPLETE")
    print("="*60)
    print(f"Symbols: {df['symbol'].n_unique()}")
    print(f"Quarters: {len(df)}")
    print(f"Saved to: {output_file}")
    print("="*60)

    print("\nSample (AAPL recent quarters):")
    sample_cols = ['quarter_end_date', 'total_revenue', 'net_income', 'roe', 'roa',
                   'gross_margin', 'net_margin', 'debt_to_equity']
    sample = df.filter(pl.col('symbol') == 'AAPL').sort('quarter_end_date', descending=True).head(3)
    print(sample.select(sample_cols))

    print("\nSample (TSLA recent quarters):")
    sample = df.filter(pl.col('symbol') == 'TSLA').sort('quarter_end_date', descending=True).head(3)
    print(sample.select(sample_cols))


if __name__ == "__main__":
    main()
