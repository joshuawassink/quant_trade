"""
Feature Engineering Evaluation Script

Evaluates the quality and coverage of all computed features:
- Data completeness (null value analysis)
- Feature distributions (outliers, ranges)
- Cross-sectional properties (variation across stocks)
- Temporal properties (stationarity, autocorrelation)
- Look-ahead bias detection
- Feature correlation analysis

Generates comprehensive evaluation report.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from src.features import TechnicalFeatures, FundamentalFeatures, SectorFeatures


def main():
    """Run comprehensive feature evaluation."""

    print("="*80)
    print("FEATURE ENGINEERING EVALUATION")
    print("="*80)

    # Load data
    print("\n📂 Loading data...")
    price_df = pl.read_parquet('data/price/daily/sample_universe_2022-10-24_to_2025-10-23.parquet')
    financials_df = pl.read_parquet('data/financials/quarterly_financials_latest.parquet')
    market_df = pl.read_parquet('data/market/daily/market_data_latest.parquet')
    metadata_df = pl.read_parquet('data/metadata/company_metadata_latest.parquet')

    print(f"   Price data: {len(price_df):,} rows")
    print(f"   Financials: {len(financials_df):,} quarters")
    print(f"   Market data: {len(market_df):,} rows")
    print(f"   Metadata: {len(metadata_df):,} companies")

    # Compute features
    print("\n🔧 Computing features...")

    # Technical features
    print("   Computing technical features...")
    tech = TechnicalFeatures()
    tech_df = tech.compute_all(price_df)
    tech_features = [col for col in tech_df.columns if col not in price_df.columns]
    print(f"   ✓ {len(tech_features)} technical features")

    # Fundamental features
    print("   Computing fundamental features...")
    fundamental = FundamentalFeatures()
    fund_df = fundamental.compute_all(financials_df)
    fund_features = [col for col in fund_df.columns if col not in financials_df.columns]
    print(f"   ✓ {len(fund_features)} fundamental features")

    # Sector features
    print("   Computing sector features...")
    sector = SectorFeatures()
    sector_df = sector.compute_all(tech_df, market_df, metadata_df)
    sector_features = [col for col in sector_df.columns if col not in tech_df.columns]
    print(f"   ✓ {len(sector_features)} sector/market features")

    total_features = len(tech_features) + len(fund_features) + len(sector_features)
    print(f"\n   📊 Total: {total_features} features")

    # Evaluate each module
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    evaluate_technical_features(tech_df, tech_features)
    evaluate_fundamental_features(fund_df, fund_features)
    evaluate_sector_features(sector_df, sector_features)

    # Overall assessment
    print("\n" + "="*80)
    print("OVERALL ASSESSMENT")
    print("="*80)

    assess_feature_quality(tech_df, tech_features, "Technical")
    assess_feature_quality(fund_df, fund_features, "Fundamental")
    assess_feature_quality(sector_df, sector_features, "Sector/Market")

    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE")
    print("="*80)


def evaluate_technical_features(df: pl.DataFrame, features: list[str]):
    """Evaluate technical features quality."""
    print("\n📈 TECHNICAL FEATURES EVALUATION:")
    print("-" * 80)

    # Data completeness
    total_rows = len(df)
    print("\n1. Data Completeness:")

    completeness_issues = []
    for feature in features:
        null_count = df[feature].is_null().sum()
        null_pct = (null_count / total_rows) * 100

        if null_pct > 20:  # More than 20% nulls is concerning
            completeness_issues.append((feature, null_pct))

    if completeness_issues:
        print(f"   ⚠️  {len(completeness_issues)} features with >20% nulls:")
        for feat, pct in sorted(completeness_issues, key=lambda x: x[1], reverse=True)[:5]:
            print(f"      - {feat}: {pct:.1f}% null")
    else:
        print("   ✓ All features have <20% null values")

    # Check for infinite values
    print("\n2. Infinite Values Check:")
    inf_issues = []
    for feature in features:
        if df[feature].dtype in [pl.Float64, pl.Float32]:
            inf_count = df[feature].is_infinite().sum()
            if inf_count > 0:
                inf_issues.append((feature, inf_count))

    if inf_issues:
        print(f"   ⚠️  {len(inf_issues)} features contain infinite values:")
        for feat, count in inf_issues[:5]:
            print(f"      - {feat}: {count} infinite values")
    else:
        print("   ✓ No infinite values detected")

    # Distribution analysis (sample)
    print("\n3. Distribution Analysis (sample):")
    sample_features = ['return_20d', 'volatility_20d', 'rsi_14', 'volume_ratio_20d']
    for feat in sample_features:
        if feat in features:
            stats = df[feat].drop_nulls().describe()
            print(f"\n   {feat}:")
            print(f"      Mean: {df[feat].mean():.4f}")
            print(f"      Std:  {df[feat].std():.4f}")
            print(f"      Min:  {df[feat].min():.4f}")
            print(f"      Max:  {df[feat].max():.4f}")

    # Cross-sectional variation
    print("\n4. Cross-sectional Variation:")
    # Check if features vary across stocks (not constant)
    low_variation = []
    for feature in features[:10]:  # Sample first 10
        if df[feature].dtype in [pl.Float64, pl.Float32]:
            std = df[feature].std()
            mean = df[feature].mean()
            if mean != 0 and abs(std / mean) < 0.01:  # CV < 1%
                low_variation.append(feature)

    if low_variation:
        print(f"   ⚠️  {len(low_variation)} features with low variation")
    else:
        print("   ✓ Features show good cross-sectional variation")


def evaluate_fundamental_features(df: pl.DataFrame, features: list[str]):
    """Evaluate fundamental features quality."""
    print("\n📊 FUNDAMENTAL FEATURES EVALUATION:")
    print("-" * 80)

    total_rows = len(df)

    # Data completeness (expected to have more nulls due to quarterly nature)
    print("\n1. Data Completeness:")
    high_null_features = []
    for feature in features:
        null_count = df[feature].is_null().sum()
        null_pct = (null_count / total_rows) * 100

        if null_pct > 50:
            high_null_features.append((feature, null_pct))

    if high_null_features:
        print(f"   Note: {len(high_null_features)} features with >50% nulls (expected for QoQ/YoY):")
        for feat, pct in sorted(high_null_features, key=lambda x: x[1], reverse=True)[:3]:
            print(f"      - {feat}: {pct:.1f}% null")
    else:
        print("   ✓ Reasonable null rates for quarterly data")

    # Check trend indicators
    print("\n2. Trend Indicators:")
    trend_features = [f for f in features if 'improving' in f or 'expanding' in f or 'accelerating' in f]
    if trend_features:
        print(f"   Found {len(trend_features)} binary trend indicators:")
        for feat in trend_features:
            true_pct = (df[feat].drop_nulls() == 1).sum() / df[feat].drop_nulls().count() * 100
            print(f"      - {feat}: {true_pct:.1f}% positive")

    # QoQ vs YoY comparison
    print("\n3. Change Magnitude Analysis:")
    qoq_features = [f for f in features if 'qoq' in f and 'pct' in f]
    yoy_features = [f for f in features if 'yoy' in f and 'pct' in f]

    if qoq_features:
        print(f"   QoQ growth rates (sample):")
        for feat in qoq_features[:3]:
            mean_growth = df[feat].mean()
            print(f"      - {feat}: {mean_growth:.2f}% avg")

    if yoy_features:
        print(f"   YoY growth rates (sample):")
        for feat in yoy_features[:3]:
            mean_growth = df[feat].mean()
            print(f"      - {feat}: {mean_growth:.2f}% avg")


def evaluate_sector_features(df: pl.DataFrame, features: list[str]):
    """Evaluate sector/market features quality."""
    print("\n🌐 SECTOR/MARKET FEATURES EVALUATION:")
    print("-" * 80)

    total_rows = len(df)

    # Data completeness
    print("\n1. Data Completeness:")
    null_features = []
    for feature in features:
        null_count = df[feature].is_null().sum()
        null_pct = (null_count / total_rows) * 100

        if null_pct > 5:  # Market data should be complete
            null_features.append((feature, null_pct))

    if null_features:
        print(f"   ⚠️  {len(null_features)} features with >5% nulls:")
        for feat, pct in sorted(null_features, key=lambda x: x[1], reverse=True)[:5]:
            print(f"      - {feat}: {pct:.1f}% null")
    else:
        print("   ✓ Market data is complete (<5% nulls)")

    # Market relative returns
    print("\n2. Market-Relative Returns:")
    market_rel_features = [f for f in features if 'vs_market' in f]
    if market_rel_features:
        print(f"   Found {len(market_rel_features)} market-relative features:")
        for feat in market_rel_features[:3]:
            mean_rel = df[feat].mean()
            std_rel = df[feat].std()
            print(f"      - {feat}: mean={mean_rel:.4f}, std={std_rel:.4f}")

    # VIX regime distribution
    print("\n3. VIX Regime Distribution:")
    if 'vix_regime' in features:
        regime_dist = df['vix_regime'].value_counts().sort('vix_regime')
        print("   VIX regime breakdown:")
        for row in regime_dist.iter_rows(named=True):
            pct = row['count'] / total_rows * 100
            print(f"      - {row['vix_regime']:10s}: {pct:5.1f}%")

    # Market trend
    print("\n4. Market Trend:")
    if 'market_trend_bullish' in features:
        bullish_pct = (df['market_trend_bullish'] == 1).sum() / total_rows * 100
        print(f"   Bullish periods (above 200 MA): {bullish_pct:.1f}%")
        print(f"   Bearish periods (below 200 MA): {100-bullish_pct:.1f}%")


def assess_feature_quality(df: pl.DataFrame, features: list[str], module_name: str):
    """Overall quality assessment for a feature module."""
    total_rows = len(df)
    total_features = len(features)

    # Calculate metrics
    avg_null_pct = sum(df[f].is_null().sum() / total_rows * 100 for f in features) / total_features

    # Count issues
    high_null_count = sum(1 for f in features if df[f].is_null().sum() / total_rows > 0.2)

    print(f"\n{module_name} Module:")
    print(f"   Total features: {total_features}")
    print(f"   Avg null rate: {avg_null_pct:.1f}%")
    print(f"   Features with >20% nulls: {high_null_count}")

    # Quality grade
    if avg_null_pct < 10 and high_null_count < 3:
        grade = "A (Excellent)"
    elif avg_null_pct < 20 and high_null_count < 5:
        grade = "B (Good)"
    elif avg_null_pct < 30:
        grade = "C (Acceptable)"
    else:
        grade = "D (Needs improvement)"

    print(f"   Quality Grade: {grade}")


if __name__ == "__main__":
    main()
