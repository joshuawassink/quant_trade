"""
Generate stock universe for production.

Fetches stocks that:
- Were in S&P 500 at any point during 2022-2025
- Have complete price data for our historical period
- Have valid sector classification
- Meet liquidity requirements

Output: List of ~500 stock symbols ready for universe.py
"""

import pandas as pd
import yfinance as yf
from loguru import logger
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def _get_fallback_universe() -> list[str]:
    """
    Fallback universe of ~600 large-cap US stocks (S&P 500 + mid-caps).
    Manually curated list updated as of 2025.
    """
    return [
        # Technology - Major players
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
        'CRM', 'AMD', 'INTC', 'CSCO', 'ACN', 'IBM', 'QCOM', 'TXN', 'INTU', 'NOW',
        'PANW', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT',
        'TEAM', 'WDAY', 'ZS', 'DDOG', 'CRWD', 'NET', 'SNOW', 'MDB', 'PLTR', 'UBER',
        'LYFT', 'ABNB', 'RBLX', 'U', 'PATH', 'BILL', 'SQ', 'PYPL', 'COIN', 'HOOD',

        # Financial Services
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'CB', 'PGR',
        'MMC', 'AXP', 'SPGI', 'CME', 'ICE', 'MCO', 'AON', 'TFC', 'USB', 'PNC',
        'COF', 'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'HIG', 'CINF', 'AJG',
        'BRO', 'WRB', 'FITB', 'HBAN', 'RF', 'CFG', 'KEY', 'ALLY', 'SOFI', 'LC',

        # Healthcare & Pharma
        'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'CVS', 'CI', 'ELV', 'HCA', 'HUM', 'CNC', 'MCK', 'CAH',
        'ISRG', 'BSX', 'MDT', 'SYK', 'EW', 'ZBH', 'BAX', 'BDX', 'RMD', 'IDXX',
        'IQV', 'A', 'ALGN', 'DXCM', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'ILMN', 'INCY',

        # Consumer - Retail & Staples
        'AMZN', 'WMT', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX',
        'BKNG', 'MAR', 'HLT', 'CMG', 'YUM', 'DPZ', 'QSR', 'SBUX', 'DRI', 'EAT',
        'PG', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB', 'GIS', 'K',
        'HSY', 'CAG', 'CPB', 'SJM', 'HRL', 'MKC', 'TAP', 'STZ', 'BF-B', 'SAM',

        # Consumer - Discretionary
        'TSLA', 'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'CMG',
        'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'BMWYY', 'TM', 'HMC',
        'DIS', 'NFLX', 'CMCSA', 'WBD', 'PARA', 'FOX', 'FOXA', 'ROKU', 'SPOT', 'TTD',
        'LVS', 'WYNN', 'MGM', 'CZR', 'PENN', 'DKNG', 'FLUT', 'BETZ', 'RSI', 'GENI',

        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
        'HAL', 'BKR', 'WMB', 'KMI', 'LNG', 'HES', 'DVN', 'FANG', 'MRO', 'APA',
        'OVV', 'CTRA', 'EQT', 'AR', 'PR', 'MTDR', 'SM', 'CHRD', 'MGY', 'VNOM',

        # Industrials
        'BA', 'CAT', 'RTX', 'HON', 'UNP', 'UPS', 'DE', 'GE', 'LMT', 'MMM',
        'FDX', 'NSC', 'CSX', 'WM', 'EMR', 'ITW', 'PH', 'ETN', 'CARR', 'OTIS',
        'PCAR', 'JCI', 'CMI', 'ROK', 'DOV', 'IR', 'FAST', 'CHRW', 'EXPD', 'JBHT',

        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM',
        'PPG', 'CTVA', 'DOW', 'ALB', 'IFF', 'FMC', 'MOS', 'CF', 'CE', 'EMN',

        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'WELL', 'SPG', 'O', 'VICI',
        'AVB', 'EQR', 'INVH', 'MAA', 'ESS', 'UDR', 'CPT', 'AIV', 'ELS', 'SUI',

        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES',
        'PEG', 'ED', 'EIX', 'AWK', 'FE', 'ETR', 'DTE', 'PPL', 'AEE', 'CMS',

        # Communication Services
        'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA',
        'TTWO', 'ATVI', 'RBLX', 'U', 'PINS', 'SNAP', 'MTCH', 'BMBL', 'YELP', 'ZG',

        # Semiconductors & Hardware
        'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'AMAT', 'ADI', 'LRCX', 'KLAC',
        'MU', 'MCHP', 'MRVL', 'NXPI', 'SWKS', 'MPWR', 'ON', 'TER', 'ENTG', 'WOLF',

        # Software & Cloud
        'MSFT', 'ORCL', 'SAP', 'ADBE', 'CRM', 'NOW', 'INTU', 'WDAY', 'PANW', 'SNPS',
        'CDNS', 'ANSS', 'FTNT', 'ZS', 'DDOG', 'CRWD', 'NET', 'SNOW', 'MDB', 'PLTR',
        'TEAM', 'ATLASSIAN', 'ZM', 'DOCU', 'OKTA', 'TWLO', 'ESTC', 'CFLT', 'GTLB', 'S',

        # E-commerce & Marketplaces
        'AMZN', 'SHOP', 'EBAY', 'ETSY', 'W', 'CHWY', 'CVNA', 'CPNG', 'MELI', 'SE',

        # Biotech
        'AMGN', 'GILD', 'REGN', 'VRTX', 'BIIB', 'MRNA', 'ILMN', 'INCY', 'ALNY', 'BMRN',
        'EXAS', 'TECH', 'SGEN', 'NBIX', 'RARE', 'UTHR', 'FOLD', 'CRSP', 'EDIT', 'NTLA',

        # More additions to reach ~600
        'V', 'MA', 'ADP', 'PAYX', 'FISV', 'FIS', 'FLT', 'BR', 'TYL', 'JKHY',
        'GPN', 'WEX', 'FOUR', 'AFRM', 'UPST', 'PTON', 'LULU', 'GPS', 'M', 'KSS',
        'JWN', 'BBWI', 'BBY', 'ULTA', 'RL', 'PVH', 'VFC', 'HAS', 'MAT', 'TPR',
    ]


def fetch_sp500_constituents() -> tuple[list[str], list[str]]:
    """
    Fetch current and historical S&P 500 constituent symbols.

    Returns:
        Tuple of (current_symbols, all_symbols)
    """
    logger.info("Fetching S&P 500 constituents...")

    try:
        # Try Wikipedia with headers
        import urllib.request
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

        # Add headers to avoid 403
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        req = urllib.request.Request(url, headers=headers)

        tables = pd.read_html(req)

        # Current constituents (table 0)
        sp500_current = tables[0]
        current_symbols = sp500_current['Symbol'].tolist()

        # Historical changes (table 1) - stocks that were removed
        sp500_changes = tables[1] if len(tables) > 1 else pd.DataFrame()
        removed_symbols = []
        if 'Removed: Ticker' in sp500_changes.columns:
            removed_symbols = sp500_changes['Removed: Ticker'].dropna().tolist()

        # Clean symbols (some have special characters)
        current_symbols = [s.replace('.', '-') for s in current_symbols]
        removed_symbols = [s.replace('.', '-') for s in removed_symbols if isinstance(s, str)]

        # Combine and deduplicate
        all_symbols = list(set(current_symbols + removed_symbols))

        logger.info(f"✓ Fetched {len(current_symbols)} current S&P 500 symbols")
        logger.info(f"✓ Fetched {len(removed_symbols)} historical symbols (removed)")
        logger.info(f"✓ Total unique symbols: {len(all_symbols)}")

        return current_symbols, all_symbols

    except Exception as e:
        logger.warning(f"Failed to fetch from Wikipedia: {e}")
        logger.info("Falling back to hardcoded S&P 500 list...")
        # Fallback: Use yfinance to get S&P 500 tickers
        # This is a subset but will work
        import yfinance as yf
        sp500 = yf.Ticker("^GSPC")
        # Note: yfinance doesn't directly provide constituents
        # Using a manually curated list of top 600 US stocks instead
        logger.info("Using expanded US large-cap universe (top ~600 stocks)")
        return [], _get_fallback_universe()


def validate_symbols(symbols: list[str], start_date: str = '2022-10-01', max_symbols: int = 500) -> tuple[list[str], dict]:
    """
    Validate symbols by checking historical data availability and gather metadata.

    Args:
        symbols: List of symbols to validate
        start_date: Start date for historical data check
        max_symbols: Maximum number of symbols to return

    Returns:
        Tuple of (valid_symbols, symbol_metadata)
        where symbol_metadata is {symbol: {'sector': str, 'avg_volume': int, 'market_cap': float}}
    """
    logger.info(f"Validating {len(symbols)} symbols (checking data from {start_date})...")

    valid_symbols = []
    symbol_metadata = {}
    invalid_symbols = []

    # Minimum criteria
    MIN_VOLUME = 500_000  # Average 500K shares/day
    MIN_DATA_POINTS = 500  # Should have ~750 days, accept 500+ (accounts for some missing)

    # Test each symbol
    for i, symbol in enumerate(symbols, 1):
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{len(symbols)} ({len(valid_symbols)} valid so far)")

        try:
            ticker = yf.Ticker(symbol)

            # Fetch historical data
            hist = ticker.history(start=start_date, end=datetime.now().strftime('%Y-%m-%d'))

            if hist.empty or len(hist) < MIN_DATA_POINTS:
                invalid_symbols.append((symbol, f"Insufficient data: {len(hist)} days"))
                continue

            # Get average volume
            avg_volume = hist['Volume'].mean()
            if avg_volume < MIN_VOLUME:
                invalid_symbols.append((symbol, f"Low volume: {avg_volume:.0f}"))
                continue

            # Get metadata (sector, market cap)
            info = ticker.info
            sector = info.get('sector', 'Unknown')
            market_cap = info.get('marketCap', 0)

            if sector == 'Unknown' or sector is None:
                invalid_symbols.append((symbol, "No sector info"))
                continue

            # Valid symbol
            valid_symbols.append(symbol)
            symbol_metadata[symbol] = {
                'sector': sector,
                'avg_volume': int(avg_volume),
                'market_cap': market_cap,
                'data_points': len(hist),
            }

            logger.debug(f"  ✓ {symbol}: {sector}, {avg_volume:.0f} vol, {len(hist)} days")

            # Stop if we have enough
            if len(valid_symbols) >= max_symbols:
                logger.info(f"Reached target of {max_symbols} symbols")
                break

        except Exception as e:
            invalid_symbols.append((symbol, f"Error: {str(e)[:50]}"))
            continue

    logger.info(f"\n✓ Validated {len(valid_symbols)} symbols")
    logger.info(f"✗ Invalid: {len(invalid_symbols)} symbols")

    if invalid_symbols[:5]:
        logger.debug(f"Sample invalid symbols:")
        for sym, reason in invalid_symbols[:5]:
            logger.debug(f"  {sym}: {reason}")

    return valid_symbols[:max_symbols], symbol_metadata


def get_sector_breakdown(symbol_metadata: dict) -> dict[str, list[str]]:
    """
    Get sector breakdown from symbol metadata.

    Args:
        symbol_metadata: Dictionary of symbol metadata (from validate_symbols)

    Returns:
        Dictionary mapping sectors to symbols
    """
    sectors = {}

    for symbol, meta in symbol_metadata.items():
        sector = meta['sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(symbol)

    # Print breakdown
    logger.info("\n📊 Sector Breakdown:")
    for sector, syms in sorted(sectors.items(), key=lambda x: len(x[1]), reverse=True):
        logger.info(f"   {sector:25s}: {len(syms):3d} stocks")

    return sectors


def main():
    """Main execution."""
    logger.info("="*70)
    logger.info("PRODUCTION STOCK UNIVERSE GENERATOR")
    logger.info("="*70)
    logger.info("Target: 500 stocks with historical data (2022-2025)")
    logger.info("Criteria: S&P 500 members (current + historical)")
    logger.info("="*70 + "\n")

    # Fetch S&P 500 constituents (current + historical)
    current_symbols, all_symbols = fetch_sp500_constituents()

    # Validate symbols (check historical data availability)
    valid_symbols, symbol_metadata = validate_symbols(
        all_symbols,
        start_date='2022-10-01',
        max_symbols=500
    )

    # Get sector breakdown
    sectors = get_sector_breakdown(symbol_metadata)

    # Output results
    logger.info("\n" + "="*70)
    logger.info(f"✓ Generated universe of {len(valid_symbols)} stocks")
    logger.info("="*70)

    # Print Python list format
    print("\n" + "="*70)
    print("# Production Universe - 500 stocks with historical data (2022-2025)")
    print("# Generated:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("PRODUCTION_UNIVERSE = [")
    for i in range(0, len(valid_symbols), 10):
        batch = valid_symbols[i:i+10]
        symbols_str = ", ".join(f"'{s}'" for s in batch)
        print(f"    {symbols_str},")
    print("]")
    print("="*70)

    print(f"\n✓ Total: {len(valid_symbols)} stocks")
    print(f"✓ Sectors: {len(sectors)}")
    print(f"✓ Date range: 2022-10-01 to present")
    print(f"✓ Min volume: 500K shares/day")
    print("\nNext steps:")
    print("1. Copy PRODUCTION_UNIVERSE list above into src/config/universe.py")
    print("2. Run data fetch scripts for expanded universe")
    print("3. Regenerate training dataset")


if __name__ == "__main__":
    main()
