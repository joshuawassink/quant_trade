"""
Script to fetch metadata for stock universe.

Fetches current metadata (sector, industry, market cap, employees, etc.)
for all stocks in our universe and saves to parquet.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from src.shared.data.providers.yfinance_metadata_provider import YFinanceMetadataProvider
from src.shared.config.universe import get_universe
from loguru import logger

# Configure logging
logger.add("logs/fetch_metadata.log", rotation="10 MB")


def main():
    """Fetch metadata for universe"""

    # Use standardized universe configuration
    # Change to 'production' to fetch for full 377-stock universe
    universe_name = 'production'  # or 'sample' for testing
    universe = get_universe(universe_name)
    logger.info(f"Using '{universe_name}' universe")

    logger.info(f"Fetching metadata for {len(universe)} stocks")

    # Create provider
    provider = YFinanceMetadataProvider()

    # Fetch metadata
    logger.info("Starting metadata fetch...")
    df = provider.fetch(universe, datetime.now(), datetime.now())

    # Summary
    logger.info(f"Fetched metadata for {len(df)} symbols")

    # Show summary by sector
    if 'sector' in df.columns:
        sector_counts = df.group_by('sector').agg(pl.len().alias('count')).sort('count', descending=True)
        logger.info(f"Sector distribution:\n{sector_counts}")

    # Save to parquet
    output_dir = Path("data/metadata")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"company_metadata_{timestamp}.parquet"
    df.write_parquet(output_file)

    logger.info(f"Metadata saved to {output_file}")

    # Also save as "latest" for easy access
    latest_file = output_dir / "company_metadata_latest.parquet"
    df.write_parquet(latest_file)
    logger.info(f"Also saved as {latest_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("METADATA FETCH COMPLETE")
    print("=" * 60)
    print(f"Symbols: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {output_file}")
    print("=" * 60)

    # Show key stats
    print("\nKey Statistics:")
    print(f"  Companies with sector info: {df['sector'].drop_nulls().len()}/{len(df)}")
    print(f"  Companies with employee count: {df['employees'].drop_nulls().len()}/{len(df)}")
    print(f"  Average market cap: ${df['market_cap'].mean() / 1e9:.1f}B")
    print(f"  Average P/E ratio: {df['pe_ratio'].mean():.1f}")

    # Show sample
    print("\nSample (key fields):")
    key_cols = ['symbol', 'company_name', 'sector', 'industry', 'employees', 'market_cap']
    print(df.select(key_cols).head(10))

    return df


if __name__ == "__main__":
    df = main()
