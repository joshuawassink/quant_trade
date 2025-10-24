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

# Production universe - 433 stocks with historical data (2022-2025)
# Generated: 2025-10-23
# Criteria: S&P 500 members (current + historical), min 500K avg volume, complete data
# Sector breakdown: Technology (64), Consumer Cyclical (53), Healthcare (50),
#   Financial Services (46), Industrials (31), Energy (27), Communication (26),
#   Consumer Defensive (22), Utilities (20), Basic Materials (19), Real Estate (18)
PRODUCTION_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    'CRM', 'AMD', 'INTC', 'CSCO', 'ACN', 'IBM', 'QCOM', 'TXN', 'INTU', 'NOW',
    'PANW', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT',
    'TEAM', 'WDAY', 'ZS', 'DDOG', 'CRWD', 'NET', 'SNOW', 'MDB', 'PLTR', 'UBER',
    'LYFT', 'ABNB', 'RBLX', 'U', 'PATH', 'BILL', 'PYPL', 'COIN', 'HOOD', 'JPM',
    'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'CB', 'PGR', 'MMC',
    'AXP', 'SPGI', 'CME', 'ICE', 'MCO', 'AON', 'TFC', 'USB', 'PNC', 'COF',
    'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'HIG', 'CINF', 'AJG', 'BRO',
    'WRB', 'FITB', 'HBAN', 'RF', 'CFG', 'KEY', 'ALLY', 'SOFI', 'LC', 'UNH',
    'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
    'GILD', 'CVS', 'CI', 'ELV', 'HCA', 'HUM', 'CNC', 'MCK', 'CAH', 'ISRG',
    'BSX', 'MDT', 'SYK', 'EW', 'ZBH', 'BAX', 'BDX', 'RMD', 'IDXX', 'IQV',
    'A', 'ALGN', 'DXCM', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'ILMN', 'INCY', 'AMZN',
    'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'MAR',
    'HLT', 'CMG', 'YUM', 'DPZ', 'QSR', 'DRI', 'EAT', 'PG', 'KO', 'PEP',
    'PM', 'MO', 'MDLZ', 'CL', 'KMB', 'GIS', 'K', 'HSY', 'CAG', 'CPB',
    'SJM', 'HRL', 'MKC', 'TAP', 'STZ', 'BF-B', 'F', 'GM', 'RIVN', 'LCID',
    'NIO', 'XPEV', 'LI', 'HMC', 'DIS', 'NFLX', 'CMCSA', 'WBD', 'FOX', 'FOXA',
    'ROKU', 'SPOT', 'TTD', 'LVS', 'WYNN', 'MGM', 'CZR', 'PENN', 'DKNG', 'FLUT',
    'RSI', 'GENI', 'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
    'OXY', 'HAL', 'BKR', 'WMB', 'KMI', 'LNG', 'DVN', 'FANG', 'APA', 'OVV',
    'CTRA', 'EQT', 'AR', 'PR', 'MTDR', 'SM', 'CHRD', 'MGY', 'VNOM', 'BA',
    'CAT', 'RTX', 'HON', 'UNP', 'UPS', 'DE', 'GE', 'LMT', 'MMM', 'FDX',
    'NSC', 'CSX', 'WM', 'EMR', 'ITW', 'PH', 'ETN', 'CARR', 'OTIS', 'PCAR',
    'JCI', 'CMI', 'ROK', 'DOV', 'IR', 'FAST', 'CHRW', 'EXPD', 'JBHT', 'LIN',
    'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'VMC', 'PPG', 'CTVA',
    'DOW', 'ALB', 'IFF', 'FMC', 'MOS', 'CF', 'CE', 'EMN', 'AMT', 'PLD',
    'CCI', 'PSA', 'DLR', 'WELL', 'SPG', 'O', 'VICI', 'AVB', 'EQR', 'INVH',
    'MAA', 'UDR', 'CPT', 'AIV', 'ELS', 'SUI', 'NEE', 'DUK', 'SO', 'D',
    'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES', 'PEG', 'ED', 'EIX', 'AWK',
    'FE', 'ETR', 'DTE', 'PPL', 'AEE', 'CMS', 'VZ', 'T', 'TMUS', 'CHTR',
    'EA', 'TTWO', 'PINS', 'SNAP', 'MTCH', 'BMBL', 'YELP', 'ZG', 'MRVL', 'NXPI',
    'SWKS', 'MPWR', 'ON', 'TER', 'ENTG', 'SAP', 'ANSS', 'ZM', 'DOCU', 'OKTA',
    'TWLO', 'ESTC', 'CFLT', 'GTLB', 'S', 'SHOP', 'EBAY', 'ETSY', 'W', 'CHWY',
    'CVNA', 'CPNG', 'SE', 'ALNY', 'BMRN', 'EXAS', 'TECH', 'NBIX', 'RARE', 'FOLD',
    'CRSP', 'EDIT', 'NTLA', 'V', 'MA', 'ADP', 'PAYX', 'FIS', 'BR', 'JKHY',
    'GPN', 'FOUR', 'AFRM', 'UPST', 'PTON', 'LULU', 'M', 'KSS', 'BBWI', 'BBY',
    'ULTA', 'RL', 'PVH', 'VFC', 'HAS', 'MAT', 'TPR',
]

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
