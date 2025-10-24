"""
Model Returns Analysis Step

Translates ML predictions into actual trading strategy returns.
This is use-case specific and focuses on financial performance.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelReturnsAnalyzer:
    """
    Analyze financial returns from model predictions.

    This is strategy-specific and customizable for each use case.
    Examples:
    - Long-only top decile
    - Long-short quintile spread
    - Market-neutral portfolio
    - Sector-neutral long-short
    """

    def __init__(self, predictions_df: pl.DataFrame):
        """
        Initialize returns analyzer.

        Args:
            predictions_df: DataFrame with predictions from ModelEvaluator
                Must contain: symbol, date, predicted_return, actual_return
        """
        self.predictions_df = predictions_df
        self.strategy_returns = None

    @classmethod
    def from_parquet(cls, predictions_path: Path):
        """
        Load predictions from parquet file.

        Args:
            predictions_path: Path to predictions.parquet from ModelEvaluator

        Returns:
            ModelReturnsAnalyzer instance
        """
        predictions_df = pl.read_parquet(predictions_path)
        return cls(predictions_df)

    def calculate_quintile_strategy(
        self,
        long_quintile: int = 5,
        short_quintile: int = 1,
        long_only: bool = False,
    ) -> pl.DataFrame:
        """
        Calculate returns for quintile-based strategy.

        Strategy:
        1. Rank stocks by predicted return each date
        2. Long top quintile, short bottom quintile (or long-only)
        3. Equal-weight portfolio
        4. Calculate daily portfolio returns

        Args:
            long_quintile: Which quintile to long (5 = top 20%)
            short_quintile: Which quintile to short (1 = bottom 20%)
            long_only: If True, only go long (no short)

        Returns:
            DataFrame with daily strategy returns
        """
        logger.info("=" * 70)
        logger.info(f"QUINTILE STRATEGY: Long Q{long_quintile}, Short Q{short_quintile}")
        logger.info("=" * 70)

        # Rank stocks by predicted return within each date
        df = self.predictions_df.with_columns([
            pl.col('predicted_return').qcut(5, labels=['1', '2', '3', '4', '5'])
            .over('date')
            .alias('quintile')
        ])

        # Long position
        long_df = df.filter(pl.col('quintile') == str(long_quintile))

        # Short position (if not long-only)
        if not long_only:
            short_df = df.filter(pl.col('quintile') == str(short_quintile))

        # Calculate daily returns
        daily_long = long_df.group_by('date').agg([
            pl.col('target_return_30d_vs_market').mean().alias('long_return'),
            pl.len().alias('n_long'),
        ])

        if long_only:
            daily_returns = daily_long.with_columns([
                pl.col('long_return').alias('strategy_return'),
                pl.lit(0).alias('n_short'),
                pl.lit(0.0).alias('short_return'),
            ])
        else:
            daily_short = short_df.group_by('date').agg([
                pl.col('target_return_30d_vs_market').mean().alias('short_return'),
                pl.len().alias('n_short'),
            ])

            daily_returns = daily_long.join(daily_short, on='date', how='inner')
            daily_returns = daily_returns.with_columns([
                (pl.col('long_return') - pl.col('short_return')).alias('strategy_return')
            ])

        daily_returns = daily_returns.sort('date')

        logger.info(f"  ✓ Strategy returns calculated for {len(daily_returns)} days")
        logger.info(f"  Avg long positions: {daily_returns['n_long'].mean():.0f}")
        if not long_only:
            logger.info(f"  Avg short positions: {daily_returns['n_short'].mean():.0f}")

        self.strategy_returns = daily_returns
        return daily_returns

    def calculate_top_n_strategy(
        self,
        top_n: int = 10,
        equal_weight: bool = True,
    ) -> pl.DataFrame:
        """
        Calculate returns for top-N stock strategy.

        Strategy:
        1. Select top N stocks by predicted return each date
        2. Equal or value-weight portfolio
        3. Long-only

        Args:
            top_n: Number of top stocks to hold
            equal_weight: If True, equal weight. Else weight by predicted return.

        Returns:
            DataFrame with daily strategy returns
        """
        logger.info("=" * 70)
        logger.info(f"TOP-{top_n} STRATEGY (Equal Weight: {equal_weight})")
        logger.info("=" * 70)

        # Rank stocks and select top N each day
        df = self.predictions_df.with_columns([
            pl.col('predicted_return').rank(descending=True).over('date').alias('rank')
        ])

        top_df = df.filter(pl.col('rank') <= top_n)

        if equal_weight:
            # Equal weight
            daily_returns = top_df.group_by('date').agg([
                pl.col('target_return_30d_vs_market').mean().alias('strategy_return'),
                pl.len().alias('n_stocks'),
            ])
        else:
            # Weight by predicted return (normalized to sum to 1)
            top_df = top_df.with_columns([
                (pl.col('predicted_return') / pl.col('predicted_return').sum().over('date')).alias('weight')
            ])
            daily_returns = top_df.group_by('date').agg([
                (pl.col('target_return_30d_vs_market') * pl.col('weight')).sum().alias('strategy_return'),
                pl.len().alias('n_stocks'),
            ])

        daily_returns = daily_returns.sort('date')

        logger.info(f"  ✓ Strategy returns calculated for {len(daily_returns)} days")
        logger.info(f"  Avg positions: {daily_returns['n_stocks'].mean():.0f}")

        self.strategy_returns = daily_returns
        return daily_returns

    def calculate_performance_metrics(self) -> dict:
        """
        Calculate financial performance metrics.

        Returns:
            Dictionary with performance metrics:
            - Total return
            - Annualized return
            - Sharpe ratio
            - Max drawdown
            - Win rate
            - Average win/loss
        """
        if self.strategy_returns is None:
            raise ValueError("Must calculate strategy returns first")

        logger.info("Calculating financial performance metrics...")

        returns = self.strategy_returns['strategy_return'].to_numpy()

        # Cumulative return
        cum_return = (1 + returns).prod() - 1

        # Annualized return (assuming ~252 trading days per year)
        n_days = len(returns)
        years = n_days / 252
        annual_return = (1 + cum_return) ** (1 / years) - 1 if years > 0 else 0

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Win rate
        wins = returns > 0
        win_rate = np.mean(wins) * 100

        # Average win/loss
        avg_win = np.mean(returns[returns > 0]) if np.any(wins) else 0
        avg_loss = np.mean(returns[returns <= 0]) if np.any(~wins) else 0

        metrics = {
            'total_return': cum_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'n_periods': n_days,
        }

        logger.info(f"  ✓ Calculated {len(metrics)} performance metrics")

        return metrics

    def compare_to_market(self, market_returns: Optional[pl.DataFrame] = None) -> dict:
        """
        Compare strategy to market benchmark.

        Args:
            market_returns: DataFrame with date and market returns.
                           If None, assumes strategy returns are already market-relative.

        Returns:
            Dictionary with comparison metrics
        """
        logger.info("Comparing to market benchmark...")

        if market_returns is not None:
            # Join strategy with market returns
            comparison = self.strategy_returns.join(market_returns, on='date', how='inner')
            excess_returns = (comparison['strategy_return'] - comparison['market_return']).to_numpy()
        else:
            # Already market-relative
            excess_returns = self.strategy_returns['strategy_return'].to_numpy()

        # Information ratio (like Sharpe but for excess returns)
        info_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

        metrics = {
            'information_ratio': info_ratio,
            'avg_excess_return': np.mean(excess_returns),
            'excess_return_volatility': np.std(excess_returns),
        }

        logger.info(f"  Information Ratio: {info_ratio:.2f}")
        logger.info(f"  Avg Excess Return: {np.mean(excess_returns)*100:.2f}% per period")

        return metrics

    def plot_cumulative_returns(self, output_path: Optional[Path] = None):
        """
        Plot cumulative returns over time.

        Args:
            output_path: Optional path to save plot
        """
        if self.strategy_returns is None:
            raise ValueError("Must calculate strategy returns first")

        logger.info("Plotting cumulative returns...")

        fig, ax = plt.subplots(figsize=(15, 8))

        # Calculate cumulative returns
        returns = self.strategy_returns['strategy_return'].to_numpy()
        dates = self.strategy_returns['date'].to_list()
        cum_returns = (1 + returns).cumprod() - 1

        # Plot
        ax.plot(dates, cum_returns * 100, linewidth=2, label='Strategy')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Strategy Cumulative Returns', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"  ✓ Saved to {output_path}")

        plt.close()

    def generate_report(self, output_dir: Path):
        """
        Generate comprehensive returns report.

        Args:
            output_dir: Directory to save report
        """
        logger.info("=" * 70)
        logger.info("GENERATING FINANCIAL RETURNS REPORT")
        logger.info("=" * 70)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate metrics
        perf_metrics = self.calculate_performance_metrics()
        market_metrics = self.compare_to_market()

        # Generate plots
        self.plot_cumulative_returns(output_dir / 'cumulative_returns.png')

        # Text report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FINANCIAL RETURNS ANALYSIS")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("STRATEGY PERFORMANCE")
        report_lines.append("=" * 80)
        report_lines.append(f"Total Return:        {perf_metrics['total_return']*100:+.2f}%")
        report_lines.append(f"Annual Return:       {perf_metrics['annual_return']*100:+.2f}%")
        report_lines.append(f"Sharpe Ratio:        {perf_metrics['sharpe_ratio']:.2f}")
        report_lines.append(f"Max Drawdown:        {perf_metrics['max_drawdown']*100:.2f}%")
        report_lines.append(f"Win Rate:            {perf_metrics['win_rate']:.1f}%")
        report_lines.append(f"Avg Win:             {perf_metrics['avg_win']*100:+.2f}%")
        report_lines.append(f"Avg Loss:            {perf_metrics['avg_loss']*100:+.2f}%")
        report_lines.append(f"Periods:             {perf_metrics['n_periods']}")
        report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("MARKET-RELATIVE PERFORMANCE")
        report_lines.append("=" * 80)
        report_lines.append(f"Information Ratio:   {market_metrics['information_ratio']:.2f}")
        report_lines.append(f"Avg Excess Return:   {market_metrics['avg_excess_return']*100:+.2f}%")
        report_lines.append(f"Excess Vol:          {market_metrics['excess_return_volatility']*100:.2f}%")
        report_lines.append("")

        report_lines.append("=" * 80)

        # Save report
        output_file = output_dir / "returns_report.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"✓ Report saved to {output_file}")

        # Print to console
        print("\n" + '\n'.join(report_lines))

        logger.info("✓ Financial returns analysis complete")
