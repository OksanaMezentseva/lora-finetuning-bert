# LoRA-BERT Classification

This repository contains two Jupyter notebooks demonstrating the use of **Low-Rank Adaptation (LoRA)** to fine-tune pre-trained BERT-based models for binary text classification tasks.

## 📂 Notebooks

### 1. `bert-lora-for-binary-classification.ipynb`
- Uses the standard `bert-base-uncased` model from Hugging Face Transformers.
- Applies LoRA adapters during fine-tuning.
- Trains a binary classifier on a custom text dataset.
- Includes preprocessing, model configuration, training, and evaluation.

### 2. `allenai-scibert-scivocab-lora-arxiv-classification.ipynb`
- Uses the `allenai/scibert_scivocab_uncased` model (SciBERT), pre-trained on scientific texts.
- Focuses on binary classification of arXiv paper abstracts.
- Leverages PEFT (Parameter-Efficient Fine-Tuning) with LoRA.
- Fine-tuned for academic domain adaptation.

## 🛠️ Technologies Used
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- PyTorch
- Datasets (jackhhao/jailbreak-classification and ccdv/arxiv-classification)
- Tokenizers

## 📌 Requirements

Install dependencies:
```bash
pip install torch transformers peft datasets scikit-learn
