"""
Fundamental Features Module

Computes fundamental metrics from quarterly financial statements.

Features computed:
- Profitability trends: ROE/ROA changes quarter-over-quarter
- Margin trends: Gross/operating/net margin changes
- Quality metrics: Debt trends, current ratio changes
- Growth metrics: Revenue/earnings growth rates
- Efficiency: Asset turnover trends

Note: Quarterly data is forward-filled to daily frequency for alignment.
"""

import polars as pl
from typing import Optional


class FundamentalFeatures:
    """
    Compute fundamental features from quarterly financial data.

    Input schema (quarterly):
    - symbol: str
    - quarter_end_date: datetime
    - ROE, ROA, margins, debt/equity, etc. (from YFinanceFinancialsProvider)

    Output: Same data with change/trend features added.
    """

    def __init__(self):
        """Initialize fundamental features computer."""
        pass

    def compute_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute all fundamental features.

        Args:
            df: Quarterly financials data

        Returns:
            DataFrame with fundamental features added
        """
        # Ensure sorted
        df = df.sort(['symbol', 'quarter_end_date'])

        # Add quarter-over-quarter changes
        df = self.add_qoq_changes(df)

        # NOTE: YoY features removed - they have 90% nulls due to requiring 4 quarters of history
        # QoQ features are more useful and have much better coverage

        # Add trend indicators
        df = self.add_trends(df)

        return df

    def add_qoq_changes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add quarter-over-quarter change features.

        Features:
        - roe_qoq_change: Change in ROE from previous quarter
        - margin_qoq_change: Changes in various margins
        - revenue_qoq_growth: Sequential revenue growth
        """
        metrics = [
            'roe', 'roa', 'gross_margin', 'operating_margin',
            'net_margin', 'debt_to_equity', 'current_ratio',
            'total_revenue', 'net_income'
        ]

        for metric in metrics:
            if metric in df.columns:
                # Calculate change from previous quarter
                df = df.with_columns([
                    (
                        pl.col(metric) - pl.col(metric).shift(1)
                    ).over('symbol').alias(f'{metric}_qoq_change')
                ])

                # Calculate percent change (for ratios/margins)
                if metric in ['total_revenue', 'net_income']:
                    df = df.with_columns([
                        (
                            (pl.col(metric) / pl.col(metric).shift(1) - 1) * 100
                        ).over('symbol').alias(f'{metric}_qoq_pct')
                    ])

        return df

    def add_yoy_changes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add year-over-year change features.

        Features:
        - roe_yoy_change: Change in ROE from 4 quarters ago
        - revenue_yoy_growth: Year-over-year revenue growth
        """
        metrics = [
            'roe', 'roa', 'gross_margin', 'net_margin',
            'total_revenue', 'net_income'
        ]

        for metric in metrics:
            if metric in df.columns:
                # YoY change (4 quarters back)
                df = df.with_columns([
                    (
                        pl.col(metric) - pl.col(metric).shift(4)
                    ).over('symbol').alias(f'{metric}_yoy_change')
                ])

                # YoY percent change
                if metric in ['total_revenue', 'net_income']:
                    df = df.with_columns([
                        (
                            (pl.col(metric) / pl.col(metric).shift(4) - 1) * 100
                        ).over('symbol').alias(f'{metric}_yoy_pct')
                    ])

        return df

    def add_trends(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add trend indicators (improving/deteriorating).

        Features:
        - roe_trend: Is ROE improving over last 3 quarters?
        - margin_trend: Are margins expanding or contracting?
        """
        # ROE trend (positive if improving)
        if 'roe' in df.columns:
            df = df.with_columns([
                (
                    (pl.col('roe') > pl.col('roe').shift(1)) &
                    (pl.col('roe').shift(1) > pl.col('roe').shift(2))
                ).over('symbol').cast(pl.Int8).alias('roe_improving')
            ])

        # Margin expansion
        if 'gross_margin' in df.columns:
            df = df.with_columns([
                (
                    pl.col('gross_margin') > pl.col('gross_margin').shift(1)
                ).over('symbol').cast(pl.Int8).alias('margin_expanding')
            ])

        # Revenue acceleration (growth rate increasing)
        if 'total_revenue' in df.columns:
            df = df.with_columns([
                (
                    pl.col('total_revenue_qoq_pct') > pl.col('total_revenue_qoq_pct').shift(1)
                ).over('symbol').cast(pl.Int8).alias('revenue_accelerating')
            ])

        return df

    def forward_fill_to_daily(
        self,
        quarterly_df: pl.DataFrame,
        daily_dates: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Forward-fill quarterly data to daily frequency.

        Args:
            quarterly_df: Quarterly financials (symbol, quarter_end_date, metrics)
            daily_dates: DataFrame with (symbol, date) for all trading days

        Returns:
            Daily dataframe with quarterly values forward-filled
        """
        # Rename quarter_end_date to date for joining
        quarterly_df = quarterly_df.rename({'quarter_end_date': 'date'})

        # Join with daily dates using asof join (forward-fill)
        result = daily_dates.join_asof(
            quarterly_df,
            on='date',
            by='symbol',
            strategy='backward'  # Use most recent quarterly data
        )

        return result


# Test function
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Load sample financials data
    financials_df = pl.read_parquet('data/financials/quarterly_financials_latest.parquet')

    print("="*70)
    print("FUNDAMENTAL FEATURES TEST")
    print("="*70)

    print(f"\nInput data: {len(financials_df)} quarterly records")
    print(f"Symbols: {financials_df['symbol'].n_unique()}")
    print(f"Date range: {financials_df['quarter_end_date'].min()} to {financials_df['quarter_end_date'].max()}")
    print(f"Columns: {len(financials_df.columns)}")

    # Compute features
    fundamental = FundamentalFeatures()
    features_df = fundamental.compute_all(financials_df)

    print(f"\nOutput data: {len(features_df)} quarterly records")
    print(f"Columns: {len(features_df.columns)}")
    print(f"\nFeatures added: {len(features_df.columns) - len(financials_df.columns)}")

    # Show new columns
    new_cols = [col for col in features_df.columns if col not in financials_df.columns]
    print(f"\nNew feature columns ({len(new_cols)}):")
    for col in sorted(new_cols):
        print(f"  - {col}")

    # Show sample for one stock
    print(f"\nSample (AAPL recent quarters):")
    sample_cols = [
        'symbol', 'quarter_end_date', 'roe', 'roe_qoq_change', 'roe_yoy_change',
        'roe_improving', 'total_revenue_qoq_pct', 'revenue_accelerating'
    ]
    sample = features_df.filter(pl.col('symbol') == 'AAPL').select(sample_cols).tail(4)
    print(sample)

    print("\n" + "="*70)
    print("✓ Fundamental features computed successfully")
