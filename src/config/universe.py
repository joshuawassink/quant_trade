"""
Stock Universe Definitions

Central configuration for stock symbols used across data collection.
This ensures consistency across all datasets (price, metadata, financials, etc.).

Usage:
    from src.config.universe import SAMPLE_UNIVERSE, get_universe

    symbols = get_universe('sample')  # Returns list of symbols
"""

# Sample universe for development and testing
# Balanced mix across sectors for diverse model training
SAMPLE_UNIVERSE = [
    # Technology (7 stocks)
    'AAPL',   # Apple - Consumer Electronics
    'MSFT',   # Microsoft - Software
    'GOOGL',  # Alphabet - Internet
    'META',   # Meta - Social Media
    'NVDA',   # NVIDIA - Semiconductors
    'NFLX',   # Netflix - Streaming
    'TSLA',   # Tesla - EVs / Tech

    # Financials (3 stocks)
    'JPM',    # JPMorgan Chase - Banking
    'BAC',    # Bank of America - Banking
    'WFC',    # Wells Fargo - Banking

    # Healthcare (3 stocks)
    'JNJ',    # Johnson & Johnson - Pharma
    'UNH',    # UnitedHealth - Health Insurance
    'PFE',    # Pfizer - Pharma

    # Consumer (4 stocks)
    'PG',     # Procter & Gamble - Consumer Staples
    'KO',     # Coca-Cola - Beverages
    'PEP',    # PepsiCo - Beverages/Snacks
    'WMT',    # Walmart - Retail

    # Energy (2 stocks)
    'XOM',    # Exxon Mobil - Oil & Gas
    'CVX',    # Chevron - Oil & Gas

    # Other (1 stock)
    'DIS',    # Disney - Entertainment
]

# Production universe (larger, more diverse)
# TODO: Expand to 100+ stocks when ready for production
PRODUCTION_UNIVERSE = SAMPLE_UNIVERSE.copy()

# Universe definitions
UNIVERSES = {
    'sample': SAMPLE_UNIVERSE,
    'production': PRODUCTION_UNIVERSE,
}


def get_universe(name: str = 'sample') -> list[str]:
    """
    Get stock universe by name.

    Args:
        name: Universe name ('sample' or 'production')

    Returns:
        List of stock ticker symbols

    Raises:
        ValueError: If universe name is invalid

    Examples:
        >>> symbols = get_universe('sample')
        >>> len(symbols)
        20
    """
    if name not in UNIVERSES:
        raise ValueError(
            f"Unknown universe: {name}. Available: {list(UNIVERSES.keys())}"
        )
    return UNIVERSES[name].copy()


def get_sector_breakdown(universe: list[str] | None = None) -> dict[str, list[str]]:
    """
    Get sector breakdown of universe.

    Args:
        universe: List of symbols (defaults to SAMPLE_UNIVERSE)

    Returns:
        Dictionary mapping sectors to symbol lists

    Note:
        This is a simplified categorization for the sample universe.
        For production, fetch actual sector data from metadata provider.
    """
    if universe is None:
        universe = SAMPLE_UNIVERSE

    # Manual sector mapping for sample universe
    sectors = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'NFLX', 'TSLA'],
        'Financials': ['JPM', 'BAC', 'WFC'],
        'Healthcare': ['JNJ', 'UNH', 'PFE'],
        'Consumer': ['PG', 'KO', 'PEP', 'WMT'],
        'Energy': ['XOM', 'CVX'],
        'Entertainment': ['DIS'],
    }

    # Filter to only symbols in the universe
    universe_set = set(universe)
    return {
        sector: [s for s in symbols if s in universe_set]
        for sector, symbols in sectors.items()
        if any(s in universe_set for s in symbols)
    }


if __name__ == "__main__":
    # Test the universe configuration
    print("="*70)
    print("STOCK UNIVERSE CONFIGURATION")
    print("="*70)

    sample = get_universe('sample')
    print(f"\n📊 Sample Universe: {len(sample)} stocks")
    print(f"   {', '.join(sample)}")

    sectors = get_sector_breakdown()
    print(f"\n🏢 Sector Breakdown:")
    for sector, symbols in sectors.items():
        print(f"   {sector:15s}: {len(symbols):2d} stocks - {', '.join(symbols)}")

    print("\n" + "="*70)
    print("✓ Universe configuration validated")
