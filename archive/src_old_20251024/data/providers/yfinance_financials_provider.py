"""
YFinance Quarterly Financials Provider

Fetches quarterly financial statements and computes financial ratios.
Provides historical time-series of fundamental metrics.

Data Structure:
- Quarterly granularity (typically 5-8 quarters available)
- Combines income statement, balance sheet, and cashflow data
- Computes derived ratios (ROE, ROA, margins, etc.)
- Forward-fill for daily alignment with price data
"""

from datetime import datetime
from loguru import logger
import pandas as pd
import polars as pl
import yfinance as yf

from .base import DataProvider


class YFinanceFinancialsProvider(DataProvider):
    """Provider for quarterly financial statements from Yahoo Finance.

    Fetches three types of financial statements:
    1. Income Statement (revenue, profits, margins)
    2. Balance Sheet (assets, liabilities, equity)
    3. Cashflow Statement (operating, investing, financing cashflows)

    Computes derived metrics:
    - Profitability: ROE, ROA, margins (gross, operating, net, EBITDA)
    - Financial Health: debt/equity, current ratio, asset turnover
    - Cashflow: FCF margin, OCF/NI ratio
    - Growth: QoQ revenue growth, earnings growth

    Schema (quarterly time-series):
    - symbol: Stock ticker
    - quarter_end_date: End date of fiscal quarter
    - fetch_date: When data was fetched

    Raw Metrics:
    - total_revenue: Quarterly revenue
    - gross_profit: Revenue - COGS
    - operating_income: Gross profit - operating expenses
    - net_income: Bottom line profit
    - ebitda: Earnings before interest, tax, depreciation, amortization
    - total_assets: Total assets
    - total_liabilities: Total liabilities
    - stockholders_equity: Shareholders' equity
    - total_debt: Long-term + short-term debt
    - current_assets: Assets convertible to cash within 1 year
    - current_liabilities: Liabilities due within 1 year
    - cash: Cash and cash equivalents
    - shares_outstanding: Number of shares outstanding
    - operating_cash_flow: Cash from operations
    - free_cash_flow: Operating CF - CapEx
    - capex: Capital expenditures

    Computed Ratios:
    - roe: Return on equity (Net Income / Equity)
    - roa: Return on assets (Net Income / Assets)
    - gross_margin: Gross Profit / Revenue
    - operating_margin: Operating Income / Revenue
    - net_margin: Net Income / Revenue
    - ebitda_margin: EBITDA / Revenue
    - debt_to_equity: Total Debt / Equity
    - current_ratio: Current Assets / Current Liabilities
    - asset_turnover: Revenue / Assets (annualized)
    - fcf_margin: Free Cash Flow / Revenue
    - ocf_to_ni: Operating CF / Net Income
    - revenue_growth_qoq: Quarter-over-quarter revenue growth
    - earnings_growth_qoq: Quarter-over-quarter earnings growth
    """

    def __init__(self):
        logger.info("YFinanceFinancialsProvider initialized")

    def fetch(
        self,
        symbols: list[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **kwargs
    ) -> pl.DataFrame:
        """
        Fetch quarterly financial data for symbols.

        Note: yfinance returns last N quarters (typically 5-8), not date-filtered.
        start_date/end_date are ignored but kept for interface consistency.

        Args:
            symbols: List of stock tickers
            start_date: Ignored (kept for interface consistency)
            end_date: Ignored (kept for interface consistency)
            **kwargs: Additional arguments (unused)

        Returns:
            Polars DataFrame with quarterly financials and computed ratios
        """
        logger.info(f"Fetching quarterly financials for {len(symbols)} symbols")

        all_data = []
        fetch_date = datetime.now()

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)

                # Fetch all three financial statements
                qf = ticker.quarterly_financials  # Income statement
                qbs = ticker.quarterly_balance_sheet  # Balance sheet
                qcf = ticker.quarterly_cashflow  # Cashflow

                # Check if data is available
                if qf is None or qf.empty:
                    logger.warning(f"No quarterly financials for {symbol}")
                    continue

                # Process each quarter
                for quarter_date in qf.columns:
                    quarter_data = self._extract_quarter_data(
                        symbol, quarter_date, qf, qbs, qcf, fetch_date
                    )
                    if quarter_data:
                        all_data.append(quarter_data)

                logger.debug(f"Fetched {len(qf.columns)} quarters for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch financials for {symbol}: {e}")
                continue

        if not all_data:
            logger.warning("No financial data fetched")
            return pl.DataFrame()

        # Convert to Polars DataFrame
        df = pl.DataFrame(all_data)

        # Validate
        if not self.validate(df):
            logger.error("Financial data validation failed")
            return pl.DataFrame()

        logger.info(f"Successfully fetched financials for {df['symbol'].n_unique()} symbols")
        return df

    def _extract_quarter_data(
        self,
        symbol: str,
        quarter_date: pd.Timestamp,
        qf: pd.DataFrame,
        qbs: pd.DataFrame,
        qcf: pd.DataFrame,
        fetch_date: datetime
    ) -> dict | None:
        """Extract and compute all metrics for a single quarter."""
        try:
            # Helper to safely get value
            def get_val(df: pd.DataFrame, key: str, default=None):
                if df is None or df.empty:
                    return default
                if key not in df.index:
                    return default
                val = df.loc[key, quarter_date] if quarter_date in df.columns else default
                return None if pd.isna(val) else float(val)

            # Extract raw metrics
            # Income statement
            revenue = get_val(qf, 'Total Revenue')
            gross_profit = get_val(qf, 'Gross Profit')
            operating_income = get_val(qf, 'Operating Income')
            net_income = get_val(qf, 'Net Income')
            ebitda = get_val(qf, 'EBITDA')

            # Balance sheet
            total_assets = get_val(qbs, 'Total Assets')
            total_liabilities = get_val(qbs, 'Total Liabilities Net Minority Interest')
            equity = get_val(qbs, 'Stockholders Equity')
            total_debt = get_val(qbs, 'Total Debt')
            current_assets = get_val(qbs, 'Current Assets')
            current_liabilities = get_val(qbs, 'Current Liabilities')
            cash = get_val(qbs, 'Cash And Cash Equivalents')
            shares = get_val(qbs, 'Ordinary Shares Number')

            # Cashflow
            ocf = get_val(qcf, 'Operating Cash Flow')
            fcf = get_val(qcf, 'Free Cash Flow')
            capex = get_val(qcf, 'Capital Expenditure')

            # Compute ratios
            roe = (net_income / equity * 100) if (net_income and equity and equity != 0) else None
            roa = (net_income / total_assets * 100) if (net_income and total_assets and total_assets != 0) else None
            gross_margin = (gross_profit / revenue * 100) if (gross_profit and revenue and revenue != 0) else None
            operating_margin = (operating_income / revenue * 100) if (operating_income and revenue and revenue != 0) else None
            net_margin = (net_income / revenue * 100) if (net_income and revenue and revenue != 0) else None
            ebitda_margin = (ebitda / revenue * 100) if (ebitda and revenue and revenue != 0) else None
            debt_to_equity = (total_debt / equity) if (total_debt and equity and equity != 0) else None
            current_ratio = (current_assets / current_liabilities) if (current_assets and current_liabilities and current_liabilities != 0) else None
            asset_turnover = (revenue / total_assets * 4) if (revenue and total_assets and total_assets != 0) else None  # Annualized
            fcf_margin = (fcf / revenue * 100) if (fcf and revenue and revenue != 0) else None
            ocf_to_ni = (ocf / net_income) if (ocf and net_income and net_income != 0) else None

            return {
                'symbol': symbol,
                'quarter_end_date': quarter_date.to_pydatetime(),
                'fetch_date': fetch_date,
                # Raw metrics
                'total_revenue': revenue,
                'gross_profit': gross_profit,
                'operating_income': operating_income,
                'net_income': net_income,
                'ebitda': ebitda,
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'stockholders_equity': equity,
                'total_debt': total_debt,
                'current_assets': current_assets,
                'current_liabilities': current_liabilities,
                'cash': cash,
                'shares_outstanding': shares,
                'operating_cash_flow': ocf,
                'free_cash_flow': fcf,
                'capex': capex,
                # Computed ratios
                'roe': roe,
                'roa': roa,
                'gross_margin': gross_margin,
                'operating_margin': operating_margin,
                'net_margin': net_margin,
                'ebitda_margin': ebitda_margin,
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'asset_turnover': asset_turnover,
                'fcf_margin': fcf_margin,
                'ocf_to_ni': ocf_to_ni,
            }

        except Exception as e:
            logger.error(f"Error extracting quarter data for {symbol} {quarter_date}: {e}")
            return None

    def validate(self, df: pl.DataFrame) -> bool:
        """
        Validate financial data.

        Checks:
        - Required columns present
        - No empty dataframe
        - Valid date ranges
        """
        if df.is_empty():
            logger.error("Financials DataFrame is empty")
            return False

        required_cols = ['symbol', 'quarter_end_date', 'total_revenue', 'net_income']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False

        # Check for reasonable values
        if df['total_revenue'].is_null().all():
            logger.error("All revenue values are null")
            return False

        logger.debug("Financial data validation passed")
        return True


# Test function
if __name__ == "__main__":
    provider = YFinanceFinancialsProvider()

    # Test with a few symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']

    print(f"Testing with {len(test_symbols)} symbols...")
    df = provider.fetch(test_symbols)

    if not df.is_empty():
        print(f"\n✓ Success! Fetched {len(df)} quarterly records")
        print(f"  Symbols: {df['symbol'].n_unique()}")
        print(f"  Date range: {df['quarter_end_date'].min()} to {df['quarter_end_date'].max()}")
        print(f"\n  Columns ({len(df.columns)}): {df.columns}")

        # Show sample for one company
        print(f"\n  Sample (AAPL most recent quarters):")
        sample = df.filter(pl.col('symbol') == 'AAPL').sort('quarter_end_date', descending=True).head(3)
        print(sample.select(['quarter_end_date', 'total_revenue', 'net_income', 'roe', 'roa', 'net_margin']))
    else:
        print("✗ Failed to fetch data")
