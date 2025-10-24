"""Yahoo Finance metadata provider for company information"""

from datetime import datetime
from typing import Optional
import polars as pl
import yfinance as yf
from loguru import logger

from .base import DataProvider, validate_symbols


class YFinanceMetadataProvider(DataProvider):
    """
    Provider for company metadata from Yahoo Finance.

    Fetches static and semi-static company information including:
    - Company profile (sector, industry, employees)
    - Market data (market cap, shares outstanding)
    - Valuation ratios (P/E, P/B, etc.)
    - Financial health metrics
    - Trading characteristics (beta, volume)

    Schema:
        - symbol: str - Ticker symbol
        - fetch_date: datetime - When metadata was fetched
        - company_name: str - Full company name
        - sector: str - GICS sector
        - industry: str - GICS industry
        - country: str - Primary country
        - employees: int - Full-time employees (optional)
        - market_cap: float - Market capitalization
        - enterprise_value: float - Enterprise value
        - shares_outstanding: float - Total shares
        - float_shares: float - Tradeable shares
        - pe_ratio: float - Price to earnings (trailing)
        - pb_ratio: float - Price to book
        - beta: float - Stock beta
        - website: str - Company website (optional)
        - description: str - Business description (optional)

    Note:
        This is point-in-time data. For historical analysis, capture
        metadata periodically (e.g., weekly) to build time-series.
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize metadata provider.

        Args:
            cache_enabled: Whether to use yfinance's built-in caching
        """
        self.cache_enabled = cache_enabled
        logger.info("YFinanceMetadataProvider initialized")

    def fetch(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pl.DataFrame:
        """
        Fetch metadata for given symbols.

        Note: start_date and end_date are ignored (metadata is point-in-time)
        but kept for interface compatibility.

        Args:
            symbols: List of ticker symbols
            start_date: Ignored (kept for interface)
            end_date: Ignored (kept for interface)
            **kwargs: Additional arguments (ignored)

        Returns:
            Polars DataFrame with metadata for each symbol

        Raises:
            ValueError: If symbols invalid
        """
        validate_symbols(symbols)

        logger.info(f"Fetching metadata for {len(symbols)} symbols")

        fetch_time = datetime.now()
        metadata_list = []

        for symbol in symbols:
            try:
                metadata = self._fetch_single(symbol, fetch_time)
                if metadata:
                    metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to fetch metadata for {symbol}: {e}")
                continue

        if not metadata_list:
            logger.warning("No metadata fetched for any symbols")
            return pl.DataFrame()

        # Convert to DataFrame
        df = pl.DataFrame(metadata_list)

        # Validate
        self.validate(df)

        logger.info(f"Successfully fetched metadata for {len(df)} symbols")
        return df

    def _fetch_single(self, symbol: str, fetch_time: datetime) -> Optional[dict]:
        """
        Fetch metadata for a single symbol.

        Args:
            symbol: Ticker symbol
            fetch_time: Timestamp of fetch

        Returns:
            Dictionary with metadata, or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Check if we got valid data
            if not info or 'symbol' not in info:
                logger.warning(f"No valid info for {symbol}")
                return None

            # Extract and standardize metadata
            metadata = {
                'symbol': symbol,
                'fetch_date': fetch_time,

                # Company Profile
                'company_name': info.get('longName') or info.get('shortName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'country': info.get('country'),
                'employees': info.get('fullTimeEmployees'),
                'website': info.get('website'),
                'description': info.get('longBusinessSummary'),

                # Market Data
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),

                # Valuation Ratios
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),
                'enterprise_to_revenue': info.get('enterpriseToRevenue'),
                'enterprise_to_ebitda': info.get('enterpriseToEbitda'),

                # Financial Health
                'total_cash': info.get('totalCash'),
                'total_debt': info.get('totalDebt'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),

                # Profitability
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
                'gross_margin': info.get('grossMargins'),
                'operating_margin': info.get('operatingMargins'),
                'ebitda_margin': info.get('ebitdaMargins'),

                # Growth
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),

                # Dividends
                'dividend_rate': info.get('dividendRate'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),

                # Trading Characteristics
                'beta': info.get('beta'),
                'avg_volume': info.get('averageVolume'),
                'avg_volume_10d': info.get('averageVolume10days'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            }

            return metadata

        except Exception as e:
            logger.error(f"Error fetching metadata for {symbol}: {e}")
            return None

    def validate(self, df: pl.DataFrame) -> bool:
        """
        Validate metadata quality.

        Checks:
        - Required columns present
        - Symbol column not empty
        - At least some companies have sector/industry

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
        required_cols = ['symbol', 'fetch_date', 'company_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check that we have symbols
        if df['symbol'].null_count() > 0:
            raise ValueError("Found null symbols")

        # Warn if critical fields are mostly missing
        critical_fields = ['sector', 'industry', 'market_cap']
        for field in critical_fields:
            if field in df.columns:
                null_pct = df[field].null_count() / len(df)
                if null_pct > 0.5:
                    logger.warning(
                        f"{field} is null for {null_pct*100:.1f}% of symbols"
                    )

        logger.debug("Metadata validation passed")
        return True

    def get_available_symbols(self) -> list[str]:
        """
        Get available symbols.

        Note: Yahoo Finance doesn't provide a symbol list API.
        Return empty list.
        """
        return []


# Example usage
if __name__ == "__main__":
    from datetime import datetime

    # Configure logging
    logger.add("yfinance_metadata.log", rotation="10 MB")

    # Test with a few symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM"]

    provider = YFinanceMetadataProvider()

    # Fetch metadata (dates are ignored)
    df = provider.fetch(symbols, datetime.now(), datetime.now())

    print("=" * 60)
    print("METADATA FETCHED")
    print("=" * 60)
    print(f"Symbols: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    # Show key fields
    key_fields = [
        'symbol', 'company_name', 'sector', 'industry',
        'employees', 'market_cap', 'pe_ratio', 'beta'
    ]
    print("Key fields:")
    print(df.select(key_fields))

    print("\n" + "=" * 60)
    print("VALUATION METRICS")
    print("=" * 60)
    valuation_fields = [
        'symbol', 'pe_ratio', 'pb_ratio', 'ps_ratio',
        'enterprise_to_revenue', 'peg_ratio'
    ]
    print(df.select(valuation_fields))

    print("\n" + "=" * 60)
    print("PROFITABILITY METRICS")
    print("=" * 60)
    profit_fields = [
        'symbol', 'roe', 'roa', 'profit_margin',
        'gross_margin', 'operating_margin'
    ]
    print(df.select(profit_fields))
