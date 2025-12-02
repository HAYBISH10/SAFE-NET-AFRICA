# SAFE-NET AFRICA  
**AI-Powered Digital Safety & Survivor Support Platform for Women & Girls**

---

## 1. Overview

**Title:** SAFE-NET AFRICA  
**Subtitle:** AI-Powered Digital Safety & Survivor Support Platform for Women & Girls  
**Themes:**  
- Digital Literacy  
- Survivor Support  
- Safety-by-Design  

**Mission:**  
SAFE-NET AFRICA uses **Data Science, Machine Learning, Deep Learning and NLP** to help protect women and girls online by:

- Detecting **phishing / scam messages**
- Detecting **abusive / harmful content**
- Providing **anonymous, trauma-informed guidance**
- Operating with a strong **privacy-first, no-data-storage** policy

The platform is built as a **simple Streamlit web app** backed by ML models trained in a single Jupyter notebook.

---

## 2. Problem Statement

Women and girls across Africa increasingly face:

- Cyber harassment and online abuse  
- Phishing, scams, and identity theft  
- Online stalking and doxxing  
- Misinformation and manipulation  
- Technology-facilitated gender-based violence (TFGBV)

Key gaps:

- Low **digital literacy** around online safety  
- Lack of **safe, anonymous tools** to check if something is dangerous  
- Limited access to **supportive, non-judgmental guidance**  
- Fear of exposure when seeking help  

**SAFE-NET AFRICA** addresses these issues by providing anonymous AI tools that allow users to:

- Check if a message or link might be a **scam**  
- Check if a message might be **abusive or harmful**  
- Receive **educational tips and survivor-support guidance**

---

## 3. Project Goals

1. Build a **phishing / scam detection model** using real-world SMS spam data.  
2. Build a **toxicity / abuse detection model** using a public toxic comments dataset.  
3. Wrap both models in a **simple, safe Streamlit app**.  
4. Provide **clear, supportive explanations** instead of just “YES/NO” answers.  
5. Follow **Safety-by-Design** and **privacy-first** principles throughout.

---

## 4. Tech Stack

- **Language:** Python  
- **ML / NLP:** scikit-learn, pandas, numpy  
- **Model Persistence:** joblib  
- **Web App:** Streamlit  
- **Environment:** Jupyter Notebook / VS Code  

---

## 5. Project Structure

```bash
safe_net_africa/
│
├── app.py                       # Main Streamlit app (UI + model inference)
├── SAFE_NET_models.ipynb        # Jupyter notebook (data prep + model training)
│
├── data/
│   ├── spam.csv                 # Kaggle SMS spam/phishing dataset
│   └── train.csv                # Kaggle toxic comments dataset
│
├── models/
│   ├── phishing_model.joblib    # Saved phishing/scam detection model
│   └── toxicity_model.joblib    # Saved abuse/toxicity detection model
│
├── README.md                # This file (full project documentation)
│
├── utils/
│   └── utils.py                 # Helper functions (preprocess, explain, guidance, etc.)
│
├── logs/
│   └── error.log                # Internal error logs (no user text)
│
├── requirements.txt             # Python dependencies
└── LICENSE                      # MIT or chosen license
