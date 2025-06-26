# 📄 Submission Report

## 👩‍💻 Intern Details
* **Name**: Shalini R  
* **Reg. No.**: 24MCS1020  
* **Internship Role**: ML-NLP Engineer  
* **Organization**: Chakaralya Analytics  

---

## ✅ Task Overview
* The objective of this task was to build a sentiment classification model using NLP techniques.  
* We used the IMDb movie review dataset to classify reviews as **positive** or **negative** using a transformer-based model (DistilBERT).  
* The final model is able to predict sentiments from raw text input with high accuracy.

---

## 🧠 Approach
* Loaded and preprocessed the IMDb dataset using Hugging Face datasets.
* Tokenized the text data using `DistilBertTokenizer` with truncation and padding.
* Fine-tuned the `distilbert-base-uncased` model using the Hugging Face `Trainer` API.
* Used a binary classification head to classify the reviews.
* Evaluated the model on test data using common metrics like Accuracy, Precision, Recall, and F1 Score.

---

## 🧰 Model Decisions
* **Model Chosen**: `distilbert-base-uncased`  
  * Chosen for its efficiency and compactness, making it ideal for fast prototyping.
* **Loss Function**: Cross Entropy Loss (via `Trainer` API by default)
* **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score
* **Training Strategy**:
  * Used 2-3 epochs with early stopping.
  * Batch size and learning rate adjusted for stability.
  * Used GPU when available, else CPU fallback ensured compatibility.

---

## 📊 Results
| Metric     | Value  |
|------------|--------|
| Accuracy   | ✅ 88%  |
| Precision  | ✅ 0.87 |
| Recall     | ✅ 1.00 |
| F1 Score   | ✅ 0.90 |

* The model was especially strong in recall, identifying all positive reviews.
* A few false positives impacted precision slightly.
* Overall, excellent performance using a distilled transformer.

---

## 📚 Key Learnings
* Hands-on understanding of tokenization and transformer fine-tuning.
* Learned to use Hugging Face `Trainer` API for seamless training and evaluation.
* Understood the importance of early stopping and proper evaluation metrics.
* Realized how pretrained models can significantly boost performance even with minimal custom architecture.

---


