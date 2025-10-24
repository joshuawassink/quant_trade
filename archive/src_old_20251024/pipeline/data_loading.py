"""
Data Loading Step

Loads raw data from storage (parquet files) for training pipeline.
"""

from pathlib import Path
from typing import Optional
import polars as pl
from loguru import logger


class DataLoader:
    """Load raw data for ML pipeline."""

    def __init__(self, data_root: Optional[Path] = None):
        """
        Initialize data loader.

        Args:
            data_root: Root directory for data files. Defaults to project_root/data
        """
        if data_root is None:
            # Default to project root/data
            self.data_root = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_root = Path(data_root)

    def load_price_data(self, symbols: list[str]) -> pl.DataFrame:
        """
        Load price data for specified symbols.

        Args:
            symbols: List of stock symbols to load

        Returns:
            Combined DataFrame with price data for all symbols
        """
        logger.info(f"Loading price data for {len(symbols)} symbols...")

        price_dir = self.data_root / "price" / "daily"

        dfs = []
        missing = []

        for symbol in symbols:
            file_path = price_dir / f"{symbol}.parquet"
            if file_path.exists():
                df = pl.read_parquet(file_path)
                dfs.append(df)
            else:
                missing.append(symbol)

        if missing:
            logger.warning(f"  Missing price data for {len(missing)} symbols: {missing[:5]}...")

        if not dfs:
            raise ValueError("No price data found for any symbols")

        combined = pl.concat(dfs)
        logger.info(f"  ✓ Loaded {len(combined):,} rows for {combined['symbol'].n_unique()} symbols")

        return combined

    def load_market_data(self) -> pl.DataFrame:
        """
        Load market data (SPY, VIX, sector ETFs).

        Returns:
            DataFrame with market data
        """
        logger.info("Loading market data...")

        market_file = self.data_root / "market" / "daily" / "market_data_latest.parquet"

        if not market_file.exists():
            raise FileNotFoundError(f"Market data not found: {market_file}")

        df = pl.read_parquet(market_file)
        logger.info(f"  ✓ Loaded {len(df):,} rows for {df['symbol'].n_unique()} market symbols")

        return df

    def load_financials(self) -> pl.DataFrame:
        """
        Load quarterly financial data.

        Returns:
            DataFrame with financial data
        """
        logger.info("Loading financial data...")

        fin_file = self.data_root / "financials" / "quarterly_financials_latest.parquet"

        if not fin_file.exists():
            raise FileNotFoundError(f"Financials not found: {fin_file}")

        df = pl.read_parquet(fin_file)
        logger.info(f"  ✓ Loaded {len(df):,} quarterly records for {df['symbol'].n_unique()} symbols")

        return df

    def load_metadata(self) -> pl.DataFrame:
        """
        Load company metadata (sectors, etc.).

        Returns:
            DataFrame with company metadata
        """
        logger.info("Loading metadata...")

        meta_file = self.data_root / "metadata" / "company_metadata_latest.parquet"

        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_file}")

        df = pl.read_parquet(meta_file)
        logger.info(f"  ✓ Loaded metadata for {len(df)} companies")

        return df

    def load_all(self, symbols: list[str]) -> dict[str, pl.DataFrame]:
        """
        Load all data sources at once.

        Args:
            symbols: List of stock symbols

        Returns:
            Dictionary with keys: 'price', 'market', 'financials', 'metadata'
        """
        logger.info("=" * 70)
        logger.info("LOADING ALL DATA SOURCES")
        logger.info("=" * 70)

        data = {
            'price': self.load_price_data(symbols),
            'market': self.load_market_data(),
            'financials': self.load_financials(),
            'metadata': self.load_metadata(),
        }

        logger.info(f"\n✓ All data sources loaded successfully")

        return data
