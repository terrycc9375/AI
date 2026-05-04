# Class Imbalance Handling Guide

## Overview
This guide explains the three different approaches implemented to handle class imbalance in the hallucination classification task.

### Class Distribution (Original)
- Attribution Failure (0): 1910 samples (33.6%)
- Entity (1): 1790 samples (31.4%)
- Overgeneralization (3): 1616 samples (28.4%)
- Number (2): 249 samples (4.4%) ⚠️ Minority
- Temporal (4): 130 samples (2.3%) ⚠️ Severe Minority

**Imbalance Ratio**: 14.7x (1910 / 130)

---

## Approach 1: Standard (Baseline)
**Command:**
```bash
python train.py standard
```

**Description:**
- No resampling applied
- Uses original data distribution
- Baseline for comparison

**When to Use:**
- Establish baseline performance
- Check if class imbalance is actually problematic

**Output Model:** `model_standard.pt`

---

## Approach 2: Oversample
**Command:**
```bash
python train.py oversample
```

**Description:**
- Oversamples minority classes to match the majority class count
- Minority (Temporal): 130 → 1910 samples
- Number: 249 → 1910 samples
- **Final dataset size:** ~7,000 → ~8,500 samples

**Pros:**
- No data loss
- Minority classes get more training exposure
- Simple implementation

**Cons:**
- Risk of overfitting on minority classes (duplicate data)
- Larger training dataset → longer training time

**When to Use:**
- When you have sufficient computational resources
- When you want to maximally prioritize minority class performance
- When overfitting is less of a concern

**Output Model:** `model_oversample.pt`

---

## Approach 3: Undersample
**Command:**
```bash
python train.py undersample
```

**Description:**
- Undersamples majority classes to 4x minority count
- Target count per class: 130 × 4 = 520 samples
- **Final dataset size:** ~5,695 → ~2,600 samples

**Pros:**
- Reduces training time significantly
- Prevents overemphasis on majority class
- More balanced learning

**Cons:**
- Loses information from majority classes
- May reduce overall accuracy if majority class info is important

**When to Use:**
- Limited computational resources
- Want faster training cycles
- Concerned about majority class dominance

**Output Model:** `model_undersample.pt`

---

## Approach 4: Combined
**Command:**
```bash
python train.py combined
```

**Description:**
- Moderate oversampling: Minority classes → 70% of majority count
- Temporal: 130 → 1,337 samples
- Number: 249 → 1,337 samples
- **Adjusted hyperparameters:**
  - Learning rate: 2e-4 → 1e-4 (lower)
  - Warmup steps: 5 → 500 (more gradual)
  - Max steps: 60 → 100 (longer training)
- **Final dataset size:** ~7,000 samples

**Pros:**
- Balanced between data retention and class balance
- Reduced learning rate helps with stability on imbalanced data
- More robust convergence

**Cons:**
- Moderate dataset inflation
- More training time due to longer schedule
- Requires tuning to find optimal hyperparameters

**When to Use:**
- Default recommended approach
- When you want balance between all considerations
- Want more stable training with class imbalance

**Output Model:** `model_combined.pt`

---

## Evaluation
Evaluate any trained model on dev.csv:

**Baseline:**
```bash
python test.py standard
```

**Oversampled:**
```bash
python test.py oversample
```

**Undersampled:**
```bash
python test.py undersample
```

**Combined:**
```bash
python test.py combined
```

### Output Files
- `evaluation_results_{approach}.json` - Comprehensive metrics
- `predictions_{approach}.json` - Per-sample predictions

### Metrics Computed
- **Accuracy**: Overall correctness
- **F1 Score (Weighted)**: Accounts for class imbalance
- **F1 Score (Macro)**: Equal weight to all classes
- **Precision & Recall (Weighted)**: Per-class performance
- **Per-Class Breakdown**: Accuracy for each hallucination type
- **Confusion Matrix**: Detailed error analysis

---

## Comparison Strategy

1. **Train all approaches:**
   ```bash
   python train.py standard
   python train.py oversample
   python train.py undersample
   python train.py combined
   ```

2. **Evaluate all on dev.csv:**
   ```bash
   python test.py standard
   python test.py oversample
   python test.py undersample
   python test.py combined
   ```

3. **Compare `evaluation_results_*.json` files:**
   - Look at overall accuracy
   - Check F1-weighted and F1-macro for class imbalance handling
   - Review per-class metrics for minority classes
   - Examine confusion matrices for error patterns

---

## Recommendations

### If Goal is Overall Accuracy
→ Use **Standard** or **Oversample**

### If Goal is Balanced Performance Across Classes
→ Use **Combined** (recommended default)

### If Computational Resources are Limited
→ Use **Undersample**

### If Minority Class Precision is Critical
→ Use **Oversample**

---

## Technical Details

### Data Resampling Logic
```python
# Oversample: duplicate minority samples
duplicates = np.random.choice(minority_samples, size=shortage, replace=True)

# Undersample: randomly select majority samples
selected = np.random.choice(majority_samples, size=target, replace=False)

# Combined: moderate oversampling
target = max_count * 0.7
```

### Model Architecture (Same for All)
- Base Model: Qwen2.5-3B-Instruct (4-bit quantized)
- LoRA Rank: 32
- LoRA Alpha: 32
- LoRA Dropout: 0.05
- Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Evidence Retrieval (Same for All)
- Embedding Model: BAAI/bge-large-en-v1.5
- Reranker: BAAI/bge-reranker-base
- Retrieval: Hybrid (FAISS + BM25) with RRF
- Top-k Retrieved: 40
- Top-k Reranked: 2

---

## Expected Training Times
- **Standard**: ~20-30 minutes (60 steps, 5,695 samples)
- **Oversample**: ~25-35 minutes (60 steps, 8,500 samples)
- **Undersample**: ~15-20 minutes (60 steps, 2,600 samples)
- **Combined**: ~30-45 minutes (100 steps, 7,000 samples)

*Times vary based on GPU and system load*

---

## Troubleshooting

### Q: "No cache found. Indexing all papers in train/..."
→ First run always indexes papers (takes ~5-7 hours). Subsequent runs reuse `indices_cache.pkl`.

### Q: CUDA out of memory
→ Reduce batch size in TrainingArguments (default: 2)

### Q: Model not predicting correctly
→ Check if model path exists (e.g., `model_standard.pt`)
→ Ensure dev.csv contains valid paper IDs in train/

### Q: Low accuracy
→ Try **combined** approach for more stable training
→ Verify evidence retrieval is working (check retrieved chunks quality)
→ Consider increasing max_steps in training config

---

## Integration with Evaluation

To compare all approaches programmatically:

```python
import json

approaches = ["standard", "oversample", "undersample", "combined"]
results = {}

for approach in approaches:
    with open(f"evaluation_results_{approach}.json") as f:
        results[approach] = json.load(f)

# Compare metrics
for approach in approaches:
    acc = results[approach]["accuracy"]
    f1 = results[approach]["f1_weighted"]
    print(f"{approach:12} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
```

