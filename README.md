# 🍊 Peeling the Truth: Visual Question Answering for Fine-Grained Citrus Disease Detection

This repository presents a **multimodal Visual Question Answering (VQA) framework** for **fine-grained citrus disease detection**, integrating **computer vision** and **natural language understanding** to enable interactive, explainable plant disease diagnosis.

---

## 📌 Key Contributions

- Fine-grained citrus disease detection using image–question pairs  
- Multimodal Transformer architecture (ViT + BERT)  
- Feature-level fusion for cross-modal reasoning  
- VQA formulated as a multi-class classification problem  
- Evaluation using Accuracy, F1-score, and WUPS  
- Supports explainable and interactive diagnosis  

---

## 🧩 Problem Formulation

Given a citrus leaf image and a natural language question, the model predicts the most relevant answer from a predefined answer space.

---

## 🏗️ System Architecture

1. Vision Transformer (ViT) for image encoding  
2. BERT-based encoder for question understanding  
3. Multimodal feature fusion  
4. Classification head for answer prediction  

---

## 📂 Repository Structure

```
├── data/
├── models/
├── training/
├── inference/
├── results/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Peeling-the-Truth-VQA-Citrus.git
cd Peeling-the-Truth-VQA-Citrus
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python training/train_vqa.py
```

---

## 🔍 Inference Example

**Question:** Is citrus greening present?  
**Answer:** Yes

---

## 📊 Evaluation Metrics

- Accuracy  
- Macro F1-score  
- Wu–Palmer Similarity (WUPS)  

---

## 🌱 Applications

- Smart agriculture systems  
- Citrus disease monitoring  
- Explainable AI for plant pathology  

---

## 📖 Citation

```bibtex
@article{peelingtruth2025,
  title={Peeling the Truth: Visual Question Answering for Fine-Grained Citrus Disease Detection},
  author={Your Name et al.},
  year={2025}
}
```

---

## 📬 Contact

Dr. Jamal Hussain Shah  
GitHub: https://github.com/your-username
