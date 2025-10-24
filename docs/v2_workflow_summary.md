# V2 Workflow Summary - Proper Train/Test Split

## What Changed

### Problem with V1
The original workflow had prediction and evaluation combined, which:
- Made it harder to reuse predictions
- Mixed training and evaluation concerns
- Didn't follow standard ML train/test split practice

### V2 Improvements

**New Components:**
1. **`data_splitting.py`** - Proper time-series train/test split
2. **`model_prediction.py`** - Separate prediction step
3. **`model_evaluation_v2.py`** - Evaluation of pre-generated predictions

**New Pipeline Steps:**
```
1. Load Data
2. Feature Engineering
3. Target Generation
4. Data Filtering
5. Data Splitting (NEW!) ← 80/20 train/test split
6. Model Training      ← Train on TRAIN set only
7. Model Prediction (NEW!) ← Predict on TEST set
8. Model Evaluation    ← Evaluate predictions
9. Returns Analysis    ← Financial performance
```

## Key Design Principles

### 1. Separation of Training and Testing

```python
# V1 (Old) - No separation
train_data = load_all_data()
model = train(train_data)
evaluate(model, train_data)  # Evaluating on training data!

# V2 (New) - Proper separation
train_data, test_data = split(all_data)
model = train(train_data)
predictions = predict(model, test_data)
evaluate(predictions)  # Evaluating on held-out test set
```

### 2. Time-Series Aware Splitting

```python
# NOT random split (would cause lookahead bias)
# Time-series split: last 20% becomes test set
df = df.sort('date')
train_df = df.head(80%)  # Earlier dates
test_df = df.tail(20%)   # Later dates
```

### 3. Prediction as Separate Step

**Benefits:**
- Predictions saved to parquet file
- Can re-evaluate without re-predicting
- Can try different strategies on same predictions
- Clear separation of concerns

```python
# Step 7: Predict
predictions.parquet ← predict(model, test_set)

# Step 8: Evaluate (uses predictions.parquet)
metrics ← evaluate(predictions.parquet)

# Step 9: Returns (uses same predictions.parquet)
returns ← analyze_returns(predictions.parquet)
```

## File Structure

### Data Files
```
data/training/
├── train_30d_latest.parquet  ← Training set (80%)
└── test_30d_latest.parquet   ← Test set (20%)
```

### Model Files
```
models/ridge_30d/
├── ridge_model.joblib
├── preprocessor.joblib
├── feature_names.txt
├── model_info.txt
└── predictions_test.parquet  ← Predictions on test set
```

### Reports
```
reports/20251024_120000/
├── evaluation_report.txt     ← ML metrics on TEST set
├── error_distribution.png
├── temporal_performance.png
├── sector_performance.png
└── returns/
    ├── returns_report.txt
    └── cumulative_returns.png
```

## Usage

### Full Pipeline
```bash
python workflows/wf_30d_returns_v2.py --full
```

**What it does:**
1. Load data
2. Engineer features
3. Generate target
4. Filter data
5. **Split into train/test (80/20)**
6. **Train on TRAIN set**
7. **Predict on TEST set → save predictions**
8. Evaluate predictions (ML metrics)
9. Analyze returns (financial metrics)

### Predict + Evaluate Only
```bash
python workflows/wf_30d_returns_v2.py --predict-eval
```

**What it does:**
- Assumes model and test set already exist
- Generates new predictions on test set
- Evaluates predictions
- Useful for: Re-evaluating after model changes

**Use when:**
- Model has been retrained
- Want fresh predictions
- Testing different strategies

## Component Details

### DataSplitter
```python
from src.pipeline.data_splitting import DataSplitter

splitter = DataSplitter()

# Time-series split (no shuffle!)
train_df, test_df = splitter.time_series_split(
    df=filtered_df,
    test_size=0.2,  # Last 20%
)

# Save splits
train_path, test_path = splitter.save_splits(
    train_df, test_df, horizon_days=30
)
```

**Output:**
```
Total samples:     252,132
Train samples:     201,705 (80.0%)
Test samples:      50,427 (20.0%)
Train date range:  2022-01-03 to 2024-08-15
Test date range:   2024-08-16 to 2024-12-31
```

### ModelPredictor
```python
from src.pipeline.model_prediction import ModelPredictor

predictor = ModelPredictor(model_dir='models/ridge_30d')
predictor.load_model()

# Predict on test set
predictions_df = predictor.predict(
    data_path='data/training/test_30d_latest.parquet',
    output_path='models/ridge_30d/predictions_test.parquet',
)
```

**predictions.parquet schema:**
```
symbol: str
date: date
sector: str (if available)
predicted_return: float
actual_return: float  (if available in input)
```

### ModelEvaluatorV2
```python
from src.pipeline.model_evaluation_v2 import ModelEvaluatorV2

evaluator = ModelEvaluatorV2(
    predictions_path='models/ridge_30d/predictions_test.parquet'
)

metrics = evaluator.run_evaluation(
    output_dir='reports/20251024_120000'
)
```

**Differences from V1:**
- Takes predictions parquet (not raw data + model)
- Focuses purely on evaluation
- No prediction logic
- Cleaner separation of concerns

## Why This Matters

### 1. Prevents Data Leakage
```python
# V1 Risk: Might accidentally use test data in training
# V2: Clear separation, impossible to mix train/test

# Training uses ONLY train set
model = train(train_data)

# Evaluation uses ONLY test set
predictions = predict(model, test_data)
```

### 2. Realistic Performance Estimates
```python
# V1: Metrics on training data (overly optimistic)
# V2: Metrics on held-out test set (realistic)

# Test set has NEVER been seen by model
# True generalization performance
```

### 3. Reusable Predictions
```python
# Generate predictions once
predict(model, test_set) → predictions.parquet

# Use many times
evaluate(predictions.parquet)
analyze_returns(predictions.parquet)
compare_strategies(predictions.parquet)

# No need to re-predict!
```

### 4. Production Ready
```python
# Same pattern works for production

# Training
train_model(historical_data)

# Production
new_data → predict() → predictions.parquet
         → store in database
         → generate trading signals
```

## Comparison: V1 vs V2

| Aspect | V1 | V2 |
|--------|----|----|
| Train/Test Split | No | Yes (80/20) |
| Prediction Step | Combined with eval | Separate |
| Evaluation Input | Raw data + model | Predictions parquet |
| Data Leakage Risk | Higher | Lower |
| Performance Metrics | Training set | Test set |
| Prediction Reuse | No | Yes |
| Production Ready | Partial | Yes |

## Migration Path

### If you have V1 models:

1. **Re-run with V2 to get proper test set evaluation:**
   ```bash
   python workflows/wf_30d_returns_v2.py --full
   ```

2. **Compare metrics:**
   - V1 metrics (training set) will be optimistic
   - V2 metrics (test set) will be realistic
   - Expect V2 R² to be lower (normal!)

### If you want both workflows:

- Keep V1 for quick experiments
- Use V2 for final evaluation
- V2 is recommended for production

## Best Practices

### 1. Always use time-series split for time-series data
```python
# GOOD
splitter.time_series_split(test_size=0.2)

# BAD (would cause lookahead bias)
sklearn.train_test_split(shuffle=True)  # Don't do this!
```

### 2. Never touch test set during development
```python
# During development: Use train set + CV
train_with_cv(train_set, n_splits=5)

# Only at the end: Evaluate on test set
final_metrics = evaluate(test_set)
```

### 3. Save predictions for reuse
```python
# Predict once
predictions = predict(model, test_set)
predictions.write_parquet('predictions.parquet')

# Reuse many times
evaluate(predictions)
backtest(predictions)
compare_strategies(predictions)
```

### 4. Use test_size appropriate for your data
```python
# More data → can afford larger test set
test_size = 0.3  # 30% if you have lots of data

# Less data → smaller test set
test_size = 0.1  # 10% if data is scarce

# Standard: 20%
test_size = 0.2  # Good default
```

## Next Steps

1. **Run V2 pipeline** to get proper test set metrics
2. **Compare with V1** to see difference
3. **Use predictions.parquet** for strategy development
4. **Extend to production** using same pattern
