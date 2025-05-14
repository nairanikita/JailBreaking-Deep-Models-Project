# Jailbreaking Deep Models – NYU Deep Learning Project (Spring 2025)

###  Team Members
- Nikita Gupta (ng3230)
- Saavy Singh
  
## Overview
This project implements adversarial attacks on a pre-trained ResNet-34 classifier trained on ImageNet-1K. We craft pixel-wise (L∞) and patch-based attacks to significantly degrade model accuracy, then evaluate how well these adversarial examples transfer to other architectures like DenseNet-121.

---


##  Tasks Implemented

| Task | Description |
|------|-------------|
| Task 1 | Evaluate ResNet-34 on clean dataset (Top-1, Top-5) |
| Task 2 | FGSM (ε = 0.02), visualize 3–5 examples |
| Task 3 | Improved attack (MI-FGSM + PGD), ≥ 70% accuracy drop | 
| Task 4 | Patch-based attack (32×32), ε = 0.3–0.5 | 
| Task 5 | Transferability tests on DenseNet-121 |

---

## Methodology


**Task 1 – Baseline:**  
We forwarded the clean 500-image test split through pretrained **ResNet-34** in `eval()` mode and record top-1 / top-5 accuracy.

**Task 2 – FGSM Attack:**  
Created adversarial images with a single **FGSM** step, `x′ = x + 0.02 × sign(∇ₓ L)`, then clamped to valid pixel range (L∞ ≤ 0.02).

**Task 3 – Strong Pixel Attack:**  
A 12-step Momentum Iterative FGSM attack (ε = 0.02, μ = 1.0) was executed, and any image that still retained its original label was subsequently processed with an additional 15-step PGD routine to ensure misclassification

**Task 4 – Patch Attack:**  
Applied MI-FGSM for 20 steps while zeroing gradients outside a random **32 × 32** patch, constraining ε = 0.5 inside the patch to drive the image toward the target class “espresso.”

**Task 5 – Transfer Study:**  
Feeded all three adversarial sets into **DenseNet-121**—without retraining—to measure how well each attack transfers across architectures.


---

## Results

### ResNet-34 Accuracy

| Dataset              | Top-1  | Top-5  |
|----------------------|--------|--------|
| Clean                | 65.52% | 83.28% |
| FGSM (ε=0.02)        |  6.20% | 35.40% |
| Improved Attack      |  0.00% | 14.60% |
| Patch Attack (32x32) | 51.40% | 83.00% |

### DenseNet-121 Transfer

| Dataset              | Top-1  | Top-5  |
|----------------------|--------|--------|
| Clean                | 74.80% | 93.60% |
| FGSM                 | 63.40% | 89.40% |
| Improved             | 60.80% | 88.60% |
| Patch                | 72.20% | 91.80% |

---


### Key Observations
* **FGSM** roughly halves top-1 accuracy while remaining partially transferable.  
* **PGD** causes the steepest drop (≈ Δ₂ %) and transfers best, confirming boundary overlap.  
* **Patch** perturbations are highly effective on ResNet-34 but transfer poorly to DenseNet-121.
* We first try MI-FGSM for its improved convergence due to momentum, but fall back to PGD if MI-FGSM fails to fool the model. This hybrid approach ensures high attack success while staying within the ε-bound.*
---
