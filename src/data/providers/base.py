"""Base classes and protocols for data providers"""

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
import polars as pl


class DataProvider(ABC):
    """
    Base class for all data providers.

    Data providers are responsible for fetching data from external sources
    and returning it in a standardized Polars DataFrame format.
    """

    @abstractmethod
    def fetch(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pl.DataFrame:
        """
        Fetch data for given symbols and date range.

        Args:
            symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            **kwargs: Additional provider-specific parameters

        Returns:
            Polars DataFrame with standardized schema

        Raises:
            ValueError: If inputs are invalid
            ConnectionError: If data source unavailable
        """
        pass

    @abstractmethod
    def validate(self, df: pl.DataFrame) -> bool:
        """
        Validate fetched data for quality and completeness.

        Args:
            df: DataFrame to validate

        Returns:
            True if data passes validation checks

        Raises:
            ValueError: If data fails validation with details
        """
        pass

    def get_available_symbols(self) -> list[str]:
        """
        Get list of symbols available from this provider.

        Returns:
            List of valid ticker symbols

        Note:
            Some providers may not support this (return empty list).
            Default implementation returns empty list.
        """
        return []


class CacheableProvider(DataProvider):
    """
    Extended base class for providers that support caching.

    Useful for reducing API calls and speeding up repeated queries.
    """

    def __init__(self, cache_enabled: bool = True):
        """Initialize with cache settings"""
        self.cache_enabled = cache_enabled

    @abstractmethod
    def clear_cache(self, symbols: Optional[list[str]] = None) -> None:
        """
        Clear cached data.

        Args:
            symbols: Specific symbols to clear, or None for all
        """
        pass


def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    """
    Validate that date range is sensible.

    Args:
        start_date: Start date
        end_date: End date

    Raises:
        ValueError: If date range is invalid
    """
    if start_date >= end_date:
        raise ValueError(
            f"start_date ({start_date}) must be before end_date ({end_date})"
        )

    # Check for reasonable date range (not too far in future)
    now = datetime.now()
    if start_date > now:
        raise ValueError(
            f"start_date ({start_date}) cannot be in the future"
        )


def validate_symbols(symbols: list[str]) -> None:
    """
    Validate symbol list.

    Args:
        symbols: List of ticker symbols

    Raises:
        ValueError: If symbols list is invalid
    """
    if not symbols:
        raise ValueError("symbols list cannot be empty")

    if not all(isinstance(s, str) for s in symbols):
        raise ValueError("All symbols must be strings")

    # Check for invalid characters (basic validation)
    for symbol in symbols:
        if not symbol.replace(".", "").replace("-", "").replace("^", "").isalnum():
            raise ValueError(
                f"Invalid symbol: {symbol}. Symbols should be alphanumeric "
                f"with optional '.', '-', or '^' characters"
            )
