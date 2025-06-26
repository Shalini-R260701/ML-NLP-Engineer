# 🎬 **IMDb Sentiment Classification using DistilBERT**

**Internship Role**: ML-NLP Engineer  
**Submitted by**: Shalini R, 24MCS1020  
**Organization**: Chakaralya Analytics

---

## 📌 **Project Overview**
This project demonstrates sentiment classification on movie reviews using a transformer-based NLP model. It leverages the DistilBERT model from Hugging Face to classify IMDb reviews into **positive (1)** or **negative (0)** sentiments. The goal is to fine-tune a lightweight transformer for binary text classification using real-world data.

---

## 📁 **Project Structure**
- **notebooks/**: Contains the main Jupyter notebook used for training and evaluation  
- **src/**: (Optional) Scripts for data processing or model training  
- **models/**: Trained model weights or checkpoints  
- **reports/**: Evaluation metrics, charts, and analysis logs  
- **requirements.txt**: All necessary Python packages  
- **README.md**: Project explanation and setup  
- **submission.md**: Your technical summary and learnings

---

## 🗃️ **Dataset Used**
- **Name**: IMDb Movie Review Dataset  
- **Total Samples**: 50,000 reviews  
  - 25,000 for training  
  - 25,000 for testing  
- **Class Distribution**: Balanced with equal positive and negative reviews  
- **Data Format**: Text reviews with labeled sentiments (0 = Negative, 1 = Positive)

---

## 🤖 **Model Summary**
- **Base Model**: distilbert-base-uncased  
- **Frameworks Used**: Hugging Face Transformers, PyTorch  
- **Architecture**: 6-layer Transformer (DistilBERT)  
- **Tokenizer**: DistilBertTokenizer with truncation and padding  
- **Classification Head**: One hidden layer followed by output logits  
- **Task Type**: Binary sequence classification (positive vs. negative)

---

## ⚙️ **Setup Instructions**

### **Steps: Prepare the Environment**

Use a virtual environment for clean dependency management.

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

### **Install dependencies**

To install the required packages for running the project, use the following command:

```bash
pip install -r requirements.txt

### **Run the Notebook**

* Open the Jupyter notebook and execute all the cells to reproduce training and evaluation.

```bash
jupyter notebook notebooks/IMDB_Sentiment_Classification_using_DistilBERT_(ML_NLP_Engineer).ipynb

### **Model Performance**

| **Metric**   | **Value** |
|--------------|-----------|
| Accuracy     | ✅ 88%     |
| Precision    | ✅ 0.87    |
| Recall       | ✅ 1.00    |
| F1 Score     | ✅ 0.90    |

* **High Recall**: The model correctly identified all positive reviews in the test set.  
* **Moderate Precision**: Some negative reviews were misclassified as positive.  
* **Overall**: The model performed exceptionally well for a distilled transformer, showing both high efficiency and reliable accuracy.

---

### **Key Concepts Applied**

* Transformer-based NLP using pre-trained models  
* Fine-tuning DistilBERT on domain-specific tasks  
* Tokenization using Hugging Face’s tokenizer  
* Hugging Face Trainer API for model training and evaluation  
* Binary text classification pipeline  
* Evaluation metrics: Accuracy, Precision, Recall, F1 Score  
* Handling balanced datasets and batching  
* Overfitting control using techniques like early stopping

