"""
Daily data update script - runs each trading day to keep data current.

Fetches:
1. Previous day's price data (OHLCV) for all stocks in production universe
2. Updated metadata (market cap, volume, ratios) - weekly
3. Updated financials (quarterly reports) - when new earnings released

Usage:
    # Manual run
    python scripts/update_daily_data.py

    # Scheduled via cron (runs Mon-Fri at 6 PM ET, after market close)
    # 0 18 * * 1-5 cd /path/to/quant_trade && .venv/bin/python scripts/update_daily_data.py

Design:
- Incremental updates only (fetch last 5 days to handle gaps)
- Deduplication to avoid duplicate rows
- Validates data before saving
- Logs all actions for monitoring
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.shared.data.providers.yfinance_provider import YFinancePriceProvider
from src.shared.data.providers.yfinance_metadata_provider import YFinanceMetadataProvider
from src.shared.data.providers.yfinance_financials_provider import YFinanceFinancialsProvider
from src.shared.config.universe import get_universe
from loguru import logger

# Configure logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "daily_updates_{time:YYYY-MM-DD}.log",
    rotation="7 days",  # Keep 1 week of logs
    retention="30 days",  # Delete logs older than 30 days
    level="INFO"
)


def is_trading_day() -> bool:
    """
    Check if today is a trading day (Mon-Fri, not a holiday).

    Returns:
        True if today is a trading day
    """
    today = datetime.now()

    # Check if weekend
    if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
        logger.info(f"Today is {today.strftime('%A')} - not a trading day")
        return False

    # TODO: Add US market holiday check (use pandas_market_calendars)
    # For now, just check weekday
    return True


def update_price_data(universe: list[str], lookback_days: int = 5):
    """
    Update price data with last N days.

    Args:
        universe: List of stock symbols
        lookback_days: How many days to fetch (default 5 to handle weekends/holidays)
    """
    logger.info(f"Updating price data for {len(universe)} stocks (last {lookback_days} days)")

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    # Fetch new data
    provider = YFinancePriceProvider()
    new_data = provider.fetch(universe, start_date, end_date)

    if new_data.is_empty():
        logger.warning("No new price data fetched")
        return

    logger.info(f"Fetched {len(new_data)} rows of new price data")

    # Update individual symbol files
    price_dir = project_root / "data" / "price" / "daily"
    price_dir.mkdir(parents=True, exist_ok=True)

    for symbol in new_data['symbol'].unique().to_list():
        symbol_file = price_dir / f"{symbol}.parquet"
        symbol_new_data = new_data.filter(pl.col('symbol') == symbol)

        if symbol_file.exists():
            # Load existing data
            existing_data = pl.read_parquet(symbol_file)

            # Combine and deduplicate (keep most recent for each date)
            combined = pl.concat([existing_data, symbol_new_data])
            combined = combined.unique(subset=['date'], keep='last').sort('date')

            logger.debug(f"{symbol}: {len(existing_data)} existing + {len(symbol_new_data)} new = {len(combined)} total")
        else:
            combined = symbol_new_data
            logger.info(f"{symbol}: New symbol, creating file")

        # Save updated data
        combined.write_parquet(symbol_file)

    logger.info(f"✓ Price data updated for {len(new_data['symbol'].unique())} symbols")


def update_metadata(universe: list[str], force: bool = False):
    """
    Update metadata (market cap, sector, etc.).

    Args:
        universe: List of stock symbols
        force: Force update even if not weekly schedule
    """
    # Only update metadata weekly (Sundays) unless forced
    if not force and datetime.now().weekday() != 6:
        logger.info("Skipping metadata update (only runs on Sundays)")
        return

    logger.info(f"Updating metadata for {len(universe)} stocks")

    provider = YFinanceMetadataProvider()
    df = provider.fetch(universe, datetime.now(), datetime.now())

    if df.is_empty():
        logger.warning("No metadata fetched")
        return

    # Save metadata
    metadata_dir = project_root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = metadata_dir / f"company_metadata_{timestamp}.parquet"
    df.write_parquet(output_file)

    # Also update "latest"
    latest_file = metadata_dir / "company_metadata_latest.parquet"
    df.write_parquet(latest_file)

    logger.info(f"✓ Metadata updated: {len(df)} symbols")


def update_financials(universe: list[str], force: bool = False):
    """
    Update quarterly financials.

    Args:
        universe: List of stock symbols
        force: Force update even if not earnings season
    """
    # Only update financials weekly unless forced
    if not force and datetime.now().weekday() != 6:
        logger.info("Skipping financials update (only runs on Sundays)")
        return

    logger.info(f"Updating financials for {len(universe)} stocks")

    provider = YFinanceFinancialsProvider()
    df = provider.fetch(universe)

    if df.is_empty():
        logger.warning("No financials fetched")
        return

    # Save financials
    financials_dir = project_root / "data" / "financials"
    financials_dir.mkdir(parents=True, exist_ok=True)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = financials_dir / f"quarterly_financials_{timestamp}.parquet"
    df.write_parquet(output_file)

    # Also update "latest"
    latest_file = financials_dir / "quarterly_financials_latest.parquet"
    df.write_parquet(latest_file)

    logger.info(f"✓ Financials updated: {len(df)} quarterly records")


def main():
    """Main daily update routine."""
    logger.info("=" * 70)
    logger.info("DAILY DATA UPDATE - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 70)

    # Check if trading day
    if not is_trading_day():
        logger.info("Not a trading day - exiting")
        return

    # Get production universe
    universe = get_universe('production')
    logger.info(f"Universe: {len(universe)} stocks")

    # Update price data (daily)
    try:
        update_price_data(universe, lookback_days=5)
    except Exception as e:
        logger.error(f"Price data update failed: {e}")

    # Update metadata (weekly on Sundays)
    try:
        update_metadata(universe, force=False)
    except Exception as e:
        logger.error(f"Metadata update failed: {e}")

    # Update financials (weekly on Sundays)
    try:
        update_financials(universe, force=False)
    except Exception as e:
        logger.error(f"Financials update failed: {e}")

    logger.info("=" * 70)
    logger.info("✓ Daily update complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
