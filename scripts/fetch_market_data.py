"""
Fetch sector ETF and market indicator data.

Retrieves:
- 11 SPDR Sector ETFs (XLK, XLF, XLV, etc.)
- 5 Market Benchmarks (SPY, S&P 500, NASDAQ, Dow, QQQ)
- 2 Volatility Indicators (VIX, VXN)
- 5 Interest Rate Instruments (Treasury yields, bond ETFs)

Data saved to: data/market/daily/market_data_YYYYMMDD.parquet
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
from src.shared.data.providers.yfinance_market_provider import YFinanceMarketProvider

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def main():
    """Fetch market and sector data for the same period as stock prices."""

    # Match the date range from sample price data (3 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # 3 years

    logger.info(f"Fetching market data from {start_date.date()} to {end_date.date()}")

    # Initialize provider (fetches all 23 instruments)
    provider = YFinanceMarketProvider()

    # Fetch data
    logger.info("Starting market data fetch...")
    df = provider.fetch(start_date=start_date, end_date=end_date)

    if df.is_empty():
        logger.error("No data fetched")
        return

    logger.info(f"Fetched {len(df)} records for {df['symbol'].n_unique()} instruments")

    # Show statistics
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Breakdown by category
    logger.info("\nBreakdown by category:")
    category_summary = df.group_by('category').agg([
        pl.len().alias('records'),
        pl.col('symbol').n_unique().alias('instruments')
    ]).sort('category')

    for row in category_summary.iter_rows(named=True):
        logger.info(f"  {row['category']:20s}: {row['instruments']:2d} instruments, {row['records']:5d} records")

    # Check data quality
    trading_days = df.filter(pl.col('category') == 'sector_etf')['date'].n_unique()
    logger.info(f"\nTrading days: {trading_days}")
    logger.info(f"Average records per instrument: {len(df) / df['symbol'].n_unique():.1f}")

    # Save data
    output_dir = project_root / 'data' / 'market' / 'daily'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save with date stamp
    date_str = datetime.now().strftime('%Y%m%d')
    output_file = output_dir / f'market_data_{date_str}.parquet'
    df.write_parquet(output_file)
    logger.info(f"\nMarket data saved to {output_file}")

    # Also save as "latest"
    latest_file = output_dir / 'market_data_latest.parquet'
    df.write_parquet(latest_file)
    logger.info(f"Also saved as {latest_file}")

    # Show sample for each category
    print("\n" + "="*70)
    print("MARKET DATA FETCH COMPLETE")
    print("="*70)
    print(f"Instruments: {df['symbol'].n_unique()}")
    print(f"Records: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to: {output_file}")
    print("="*70)

    print("\nSample (most recent day):")
    latest_date = df['date'].max()
    sample = df.filter(pl.col('date') == latest_date).select([
        'category', 'symbol', 'name', 'close'
    ]).sort('category', 'symbol')
    print(sample)

    # Show VIX history (useful for market regime identification)
    print("\nVIX levels over time (market volatility):")
    vix_summary = df.filter(pl.col('symbol') == '^VIX').select([
        pl.col('close').min().alias('min'),
        pl.col('close').quantile(0.25).alias('25th_pct'),
        pl.col('close').median().alias('median'),
        pl.col('close').quantile(0.75).alias('75th_pct'),
        pl.col('close').max().alias('max'),
        pl.col('close').mean().alias('mean')
    ])
    print(vix_summary)


if __name__ == "__main__":
    main()
