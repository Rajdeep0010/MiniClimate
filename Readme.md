# MiniClimate: Domain-Adaptive and Task-Adaptive Pretrained Model for Climate-Related NLP Tasks

---

**Author:** Rajdeep Bose (University Roll No: 23478124026)

**Guide:** Dr. Moumita Basu

**Institution:** NSHM Knowledge Campus, Kolkata — M.Sc. Data Science & Analytics (2024–2026)

**University:** Maulana Abul Kalam Azad University of Technology (MAKAUT), West Bengal

---
## Overview

**MiniClimate** investigates whether a lightweight **MiniLM** model can be adapted to the climate domain through **Domain-Adaptive Pretraining (DAPT)**, **Task-Adaptive Pretraining (TAPT)**, and their combination (**DAPT+TAPT**) — using a pretraining corpus built entirely from informal, user-generated climate-related YouTube comments.

The adaptively pretrained models are fine-tuned and evaluated on four diverse climate NLP tasks, and benchmarked against MiniLM-base, ClimateBERT, SciBERT, RoBERTa, and DistilBERT using a rigorous multi-seed statistical framework (mean, standard deviation, paired t-tests, Bonferroni correction).

The final, best-performing variant (DAPT+TAPT-MiniLM) is referred to throughout as **MiniClimate**.


## Problem Statement

Despite growing interest in climate-related NLP, existing domain-specific models typically adapt full-scale architectures using formal sources like news, scientific papers, or corporate disclosures. Little attention has been given to whether a lightweight, computationally efficient model can be effectively adapted using domain- and task-adaptive pretraining strategies (DAPT, TAPT, and DAPT+TAPT), especially when relying on an informal, user-generated corpus.

Furthermore, prior evaluations often rely on a narrow set of tasks and lack the multi-seed statistical rigor needed to confirm that performance gains are genuine rather than accidental. This study bridges these gaps by evaluating whether adaptively pretraining a lightweight MiniLM model on informal, user-generated climate text improves performance across four diverse downstream tasks, using a rigorous, multi-seed statistical framework to benchmark it against larger general and domain-specific models.

---

## Key Findings

- **DAPT+TAPT-MiniLM (MiniClimate)** achieved the strongest mean improvement across all four downstream tasks over the unadapted MiniLM baseline.
- It remained competitive with — and at times outperformed — much larger models (RoBERTa, ClimateBERT, SciBERT) while using **~33M parameters**, roughly 2–3.8× fewer than the benchmark models.
- Statistical significance after Bonferroni correction held for only a subset of comparisons; many improvements were significant at p<0.05 but not the stricter threshold.
- All MiniLM variants underperformed on the informal, tweet-based stance detection task (Task 3), indicating adaptive pretraining benefits are not uniform across all task types.

---
**Model on Hugging Face Hub:** [https://huggingface.co/Raj7722/MiniClimate](https://huggingface.co/Raj7722/MiniClimate)

---
## How to Use

The pretrained MiniClimate checkpoint is hosted on Hugging Face Hub and can be loaded directly:

```notebook-python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("Raj7722/MiniClimate")
model = AutoModelForMaskedLM.from_pretrained("Raj7722/MiniClimate")
```

> This is a **pretrained (MLM) checkpoint**, adapted to climate-domain text via DAPT+TAPT. To use it for a downstream classification task (e.g., stance detection, claim detection, sentiment classification), attach a classification head and fine-tune it on your labeled dataset, following the same protocol described in Section 3.6 of the project report.

### Quick fill-mask test

```notebook-python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
fill_mask("Global warming is caused by [MASK] gas emissions.")
```

---
## Pretraining Dataset Preprocessing Pipeline

```
- - - - - - - -    - - - - - - - -    - - - - - - - -    - - - - - - - -    - - - - - - - -
: 1 - Dataset  :    : 2 - Unicode  :    : 3 - Exact    :    : 4 - Near-    :    : 5 - Fan /    :
: Loading &    : -> : Repair       : -> : Duplicate    : -> : Duplicate    : -> : Spam         :
: Schema       :    :              :    : Removal      :    : Removal      :    : Comment      :
: Normalization:    :              :    : ------------ :    : ------------ :    : Removal      :
:              :    :              :    : 5,761        :    : 3,606        :    : ------------ :
:              :    :              :    : removed      :    : removed      :    : 1,483        :
:              :    :              :    :              :    :              :    : removed      :
- - - - - - - -    - - - - - - - -    - - - - - - - -    - - - - - - - -    - - - - - - - -
                                                                                        |
       .------------------------------------------------------------------------------
       |
       v
- - - - - - - -    - - - - - - - -    - - - - - - - -    - - - - - - - -    - - - - - - - -
: 6 - Noise    :    : 7 - Text     :    : 8 - Length & :    : 9 - Language :    : 10 - Climate :
: Filtering    : -> : Cleaning &   : -> : Quality      : -> : Detection    : -> : Relevance    :
: ------------ :    : Normalization:    : Filtering    :    : ------------ :    : Scoring      :
: 16 noisy     :    :              :    : ------------ :    : Enables      :    : ------------ :
: comments     :    :              :    : 10,457 short :    : multilingual :    : 37,754       :
: removed      :    :              :    : 2 low-quality:    : keyword      :    : off-topic    :
:              :    :              :    : 30 empty     :    : matching     :    : comments     :
:              :    :              :    :              :    :              :    : removed      :
- - - - - - - -    - - - - - - - -    - - - - - - - -    - - - - - - - -    - - - - - - - -
                                                                                        |
                                                                                        v
                                                                        - - - - - - - - - - - - -
                                                                        : Final Corpus            :
                                                                        : ----------------------- :
                                                                        : 17,346 climate-relevant :
                                                                        : rows retained           :
                                                                        - - - - - - - - - - - - -
```

---

## Methodology Pipeline

```
┌─────────────────────────┐        ┌──────────────────────────┐
│   Pretraining Pipeline   │        │  Downstream Task Pipeline │
└─────────────────────────┘        └──────────────────────────┘

- - - - - - - - - - - - -           - - - - - - - - - - - - - - - -
: Cleaned Corpus          :         : 4 Downstream Tasks            :
: (17,346 climate-        :         : Stance Detection,             :
: relevant rows)          :         : Env. Claim Detection,         :
- - - - - - - - - - - - -           : Tweet Stance Detection,       :
            |                        : Sentiment Classification      :
            v                        - - - - - - - - - - - - - - - -
- - - - - - - - - - - - -                        |
: Data Split               :                       v
: Train / Val / Test       :          - - - - - - - - - - - - - - - -
- - - - - - - - - - - - -            : 70 / 15 / 15 Split             :
       /        \                    : (Train / Val / Test)          :
      v          v                   - - - - - - - - - - - - - - - -
- - - - - -  - - - - - -                          |
: Domain   :  : Task     :                        v
: Corpus   :  : Corpus   :            - - - - - - - - - - - - - - - -
- - - - - -  - - - - - -              : 8 Models Evaluated             :
   |    \       /     |               : 4 MiniLM variants +           :
   v     v     v      v               : 4 Benchmark models            :
- - - - -  - - - -  - - - - - -       : (ClimateBERT, SciBERT,        :
: DAPT   :: TAPT   :: DAPT+TAPT :     : DistilBERT, RoBERTa)          :
- - - - -  - - - -  - - - - - -       - - - - - - - - - - - - - - - -
   |          |            |                       |
   v          v            v                        v
- - - - -  - - - -  - - - - - - -    - - - - - - - - - - - - - - - -
: DAPT-  :: TAPT-  :: DAPT+TAPT-  :   : Fine-Tuning                     :
: MiniLM :: MiniLM :: MiniLM       :   : 10 epochs, LR = 5e-5,         :
:        ::        :: (MiniClimate):  : Early stopping (patience 2)  :
- - - - -  - - - -  - - - - - - -    - - - - - - - - - - - - - - - -
                                                    |
                                                    v
                                      - - - - - - - - - - - - - - - -
                                      : 5 Random Seeds                  :
                                      : (0, 42, 2012, 33, 131)         :
                                      - - - - - - - - - - - - - - - -
                                                    |
                                                    v
                                      - - - - - - - - - - - - - - - -
                                      : Per-Seed Metrics                :
                                      : Macro-F1, Precision,           :
                                      : Recall, Accuracy               :
                                      - - - - - - - - - - - - - - - -
                                                    |
                                                    v
                                      - - - - - - - - - - - - - - - -
                                      : Mean & Standard Deviation       :
                                      : (across 5 seeds per model/task) :
                                      - - - - - - - - - - - - - - - -
                                                    |
                                                    v
                                      - - - - - - - - - - - - - - - -
                                      : 28 Pairwise Paired t-Tests      :
                                      : (per downstream task)          :
                                      - - - - - - - - - - - - - - - -
                                                    |
                                                    v
                                      - - - - - - - - - - - - - - - -
                                      : Bonferroni Correction           :
                                      : (α = 0.05 / 28 ≈ 0.00179)      :
                                      - - - - - - - - - - - - - - - -
                                                    |
                                                    v
                                      - - - - - - - - - - - - - - - -
                                      : Final Evaluation Outcome        :
                                      : Per-task rankings +            :
                                      : significance results           :
                                      - - - - - - - - - - - - - - - -
```

---

## Adaptive Pretraining Strategies

| Strategy | Starting Checkpoint | Corpus | Best LR |
|---|---|---|---|
| DAPT | MiniLM-base | Full climate corpus | 5e-5 → 2.5e-5 |
| TAPT | MiniLM-base | Task-specific corpus | 5e-5 |
| DAPT+TAPT (MiniClimate) | DAPT checkpoint | Task-specific corpus | 3e-5 |

---

## Results Snapshot

| Model | Avg F1 (%) across 4 tasks | Parameters | F1/Million Params |
|---|---|---|---|
| **DAPT+TAPT-MiniLM (MiniClimate)** | 75.22 | ~33M | **2.28** |
| DistilBERT | 75.51 | ~66M | 1.14 |
| RoBERTa | 74.80 | ~125M | 0.60 |
| ClimateBERT | 74.32 | ~82M | 0.91 |
| SciBERT | 71.99 | ~110M | 0.65 |
| DAPT-MiniLM | 72.81 | ~33M | 2.21 |
| TAPT-MiniLM | 71.56 | ~33M | 2.17 |
| MiniLM-base | 68.60 | ~33M | 2.08 |

MiniClimate delivers the best **F1-per-parameter efficiency** of all models tested.

---

## Tools & Frameworks

- **Python**, **PyTorch**
- **Hugging Face Transformers** (v4.44.0), **Datasets** (v2.10.0), **Accelerate** (v0.34.0)
- **Scikit-learn**, **SciPy**, **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- Trained on **Google Colab (T4 GPU)**, with checkpoints stored on Google Drive.

---

## Limitations

- Two of the four downstream datasets are small (265 and 320 rows) with class imbalance.
- Pretraining corpus (~657K words, ~17K rows) is small relative to web-scale corpora typically used for DAPT/TAPT.
- Pretraining/evaluation limited to English (no multilingual downstream evaluation).
- Limited to a single T4 GPU — constrained batch sizes and sequence lengths.
- Most statistically significant results held only at p<0.05, not after Bonferroni correction.
- MiniLM variants (including MiniClimate) underperformed all benchmark models on Task 3 (tweet-based stance detection).

---

## Future Work

- Expand the pretraining corpus (more sources, larger scale).
- Multilingual downstream evaluation (Hindi/Hinglish climate text).
- Investigate why informal, tweet-based formats remain challenging for lightweight models.
- Explore additional lightweight architectures beyond MiniLM.

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Rajdeep Bose

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```