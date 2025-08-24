# 🎯 Disentangled Representation Learning with β-VAE  

[![Report](https://img.shields.io/badge/Report-PDF-blue)](β-VAE_Based_Representation_Learning_for_Interpretable_Generative_Modeling.pdf)  
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)]() [![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)]() [![License](https://img.shields.io/badge/License-MIT-black)]()  

---

## 📌 Introduction  
This project explores **Variational Autoencoders (VAEs)** and their extension, the **β-VAE**, to study the trade-off between reconstruction fidelity and disentangled latent representations.  
We implemented models in **PyTorch**, evaluated them on **MNIST, Fashion-MNIST, and dSprites**, and analyzed the impact of the hyperparameter β.  

---

## 📊 Datasets  
- **MNIST** → Handwritten digit images (28×28 grayscale)  
- **Fashion-MNIST** → Clothing item images (28×28 grayscale)  
- **dSprites** → Synthetic dataset with controlled generative factors (shape, scale, rotation, position)  

---

## 🏗️ Architecture  
- **Encoder–Decoder framework** with reparameterization trick  
- **β-VAE loss function**:  
  \[
  L(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \, KL(q_\phi(z|x) || p(z))
  \]  
- Configurable for **Bernoulli** or **Gaussian decoders**  
- Training pipelines with logging, checkpoints, and visualization utilities  

---

## ⚙️ Implementation Details  
- Framework: **PyTorch** (v2.1)  
- Languages: **Python, SQL** (for utilities & data validation)  
- AWS Integration: **S3 (data storage), Glue (ETL jobs), Lambda (automation), CloudWatch (monitoring)**  
- Optimizer: **Adagrad** with learning rate 1e-2  
- Batch size: **128**  
- Epochs: **30 (MNIST), 50 (dSprites)**  

---

## 📈 Results & Insights  
- **β = 1** → High reconstruction quality, but entangled latent factors  
- **β = 4** → Best trade-off; disentangled factors (digit slant, stroke width, object scale, position) while preserving sharp reconstructions  
- **β ≥ 8** → Over-regularization; blurry outputs, under-utilized latent capacity  

### Sample Reconstructions & Latent Traversals  
*(Add images/plots here if available — e.g., training curves, traversal grids)*  

---

## 📄 Project Report  
For detailed methodology, experiments, and results, check the full report:  

👉 [**Download Project Report (PDF)**](β-VAE_Based_Representation_Learning_for_Interpretable_Generative_Modeling)  

---

## 🚀 How to Run  

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/beta-vae.git
cd beta-vae

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python main.py --dataset mnist --beta 4 --epochs 30

# 4. Evaluate results
python plot_metrics.py
