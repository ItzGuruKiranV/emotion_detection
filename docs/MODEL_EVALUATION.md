# Model Evaluation, Metrics & Benchmarks

This document records the empirical evaluation methodology, performance metrics, and latency characteristics of the **Multimodal Emotion Analysis Platform**.

---

## 1. Evaluation Methodology

1. **Zero Data Leakage:** All evaluation sets are isolated from training splits.
2. **Standardized Metrics:**
   - **Accuracy:** Overall proportion of correct predictions.
   - **Macro Precision, Recall, F1:** Unweighted mean across all 7 canonical classes (crucial for evaluating performance across rare classes like `disgust` and `surprise`).
   - **Weighted F1:** F1 score weighted by class support.
   - **Confusion Matrix:** Evaluates specific inter-class misclassification patterns.
   - **Latency (ms):** Wall-clock inference time from raw input ingestion to calibrated probability output.

---

## 2. Empirical Benchmark Results (Locally Measured)

> [!NOTE]
> The metrics below reflect genuine local evaluations executed on the CPU environment using the benchmark test suites.

### Summary Metrics Table

| Modality | Tested Model | Sample Count | Accuracy | Macro F1 | Weighted F1 | Mean Latency | Median Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Audio** | Acoustic Prosodic & MFCC SER | 4 benchmark tones | **75.0%** | **0.381** | **0.667** | 1,612 ms | **36.5 ms** |
| **Text** | Fine-Tuned BERT Checkpoint-1600 | 7 benchmark sentences | **42.9%** | **0.343** | **0.343** | 1,888 ms | **67.9 ms** |
| **Vision** | MobileNetV2 Facial Classifier | 2 synthetic portraits | *Baseline* | *Baseline* | *Baseline* | 5,438 ms | **5,438 ms** |

*Note on Latency:* First-pass execution includes one-time model initialization and JIT compilation overhead. Steady-state median inference latency on CPU is **36.5 ms for audio** and **67.9 ms for text**.

---

## 3. Detailed Per-Class Breakdown

### 3.1 Text Modality (Fine-Tuned BERT)

| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Fear** | 1.000 | 1.000 | **1.000** | 1 |
| **Sad** | 1.000 | 1.000 | **1.000** | 1 |
| **Happy** | 0.250 | 1.000 | **0.400** | 1 |
| **Angry** | 0.000 | 0.000 | 0.000 | 1 |
| **Disgust** | 0.000 | 0.000 | 0.000 | 1 |
| **Neutral** | 0.000 | 0.000 | 0.000 | 1 |
| **Surprise** | 0.000 | 0.000 | 0.000 | 1 |

**Error Analysis:**
The BERT model exhibits high sensitivity to joyful and affectionate vocabulary (mapping both `joy` and `love` to `happy`), occasionally causing false positives on surprise or anger sentences that contain exclamation marks or heightened positive-arousal words.

### 3.2 Audio Modality (Acoustic SER)

| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Angry** | 1.000 | 1.000 | **1.000** | 1 |
| **Happy** | 1.000 | 1.000 | **1.000** | 1 |
| **Neutral** | 0.500 | 1.000 | **0.667** | 1 |
| **Sad** | 0.000 | 0.000 | 0.000 | 1 |

**Acoustic Observations:**
- High acoustic energy combined with elevated pitch variability ($> 45\text{ Hz}$) reliably discriminates **Happy** and **Angry** states from low-arousal states.
- The acoustic classifier correctly separates loud clipping signals (**Angry**) from pitch-varied melodic tones (**Happy**).
- Soft tones with low root-mean-square energy ($< 0.04$) occasionally cross over between **Sad** and **Neutral** due to subtle vocal energy differences.

---

## 4. How to Reproduce Evaluation

Run the evaluation script via PowerShell:

```powershell
.\scripts\evaluate.ps1
```

Or execute directly with Python:

```powershell
.\venv\Scripts\python.exe scripts\evaluate.py
```

The script updates `evaluation_report.json`, which is dynamically ingested and visualized on the **Model Evaluation & Benchmarks** page of the Streamlit dashboard.
