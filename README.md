# Jailbreaking Deep Models – NYU Deep Learning Project (Spring 2025)

###  Team Members
- Nikita Gupta (ng3230)
- Saavy Singh  (ss19170)
  
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
## Instructions to Run the Code
### Requirements

To run the notebook, you'll need:

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
- PIL
- tqdm
- scikit-learn
- seaborn (for visualizations)

Install them using:
```bash
pip install torch torchvision matplotlib numpy pillow tqdm scikit-learn seaborn
```

Run the notebook:
```bash
jupyter notebook Jailbreak_Deep-Models-Project3.ipynb
```


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
| Clean                | 76.00% | 94.20% |
| FGSM (ε=0.02)        |  3.40% | 21.20% |
| Improved Attack      |  0.00% | 1.00% |
| Patch Attack (32x32) | 4.40% | 35.60% |

### DenseNet-121 Transfer

| Dataset              | Top-1  | Top-5  |
|----------------------|--------|--------|
| Clean                | 74.80% | 93.60% |
| FGSM                 | 45.80% | 76.20% |
| Improved             | 40.00% | 76.40% |
| Patch                | 67.60% | 90.20% |

---

### Key Observations

- Even simple attacks like FGSM cause sharp accuracy drops under $L_\infty$ constraints.
- Iterative attacks (PGD) are significantly more effective and transfer more reliably.
- Patch attacks, despite being localized, are highly effective when $\epsilon$ is large.
- Top-5 accuracy is more resilient than Top-1.
- Transferability to DenseNet-121 confirms vulnerability beyond white-box settings.

---
