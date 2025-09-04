# Student Risk Prediction System  

## Overview  
The **Student Risk Prediction System** is a machine learning–based project designed to classify students into **High Risk, Medium Risk, and Low Risk** categories. The system leverages both **structured academic data** (grades, attendance, demographics) and **unstructured feedback text** (NLP sentiment analysis) to provide actionable insights for early intervention.  

---

## Features  
- ✅ **Structured Data Analysis** with multiple ML models:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
  - Neural Network (Keras/TensorFlow)  

- ✅ **Natural Language Processing (NLP)** for student feedback:  
  - Text cleaning & preprocessing (NLTK)  
  - TF-IDF vectorization  
  - Logistic Regression classifier  
  - Sentiment analysis with VADER  

- ✅ **Web Application**  
  - Built with Flask (Python backend + Jinja2 templating)  
  - Interactive form for student data input  
  - Risk prediction with confidence score  
  - REST API for integration  

- ✅ **Visualizations**  
  - Model performance comparison  
  - Confusion matrices  
  - Risk distribution plots  

- ✅ **Ethics in AI**  
  - Ensures **anonymization** (no student names)  
  - Addresses **bias** (gender, socioeconomic factors)  
  - Emphasizes **responsible use** as decision-support, not replacement  

```
## Project Structure  

├── data/                                        # Dataset ( raw )
├── models/                                      # Saved ML & NLP models
├── student_preformance(final_project).ipynb     # Jupyter notebooks (experiments, preprocessing, training)
├── templates/                                   # HTML templates
│   ├── index.html                               # main page 
│   └── result.html                              # result page   
├── app.py                                       # Main Flask app
├── requirements.txt                             # Dependencies
└──  README.md                                   # Project documentation
```
##  Installation 
1.Install dependencies:
  ```
  pip install -r requirements.txt
  ```
2.Download required NLTK data (first run only):
```
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")
```
## Usage
1. Train & Evaluate Models

  Run Jupyter notebooks in the notebooks/ directory to:
  - Preprocess data
  - Train ML models
  - Evaluate performance (Accuracy, Precision, Recall, F1-score)

2. Run Flask Web App
   ```
   cd app
   python app.py
   ```
   Access the app at http://127.0.0.1:5000/
## Results
Structured Data Models
| Model                  | Accuracy  | Precision | Recall | F1-Score |
| ---------------------- | --------- | --------- | ------ | -------- |
| Decision Tree          | **93.3%** | 93.2%     | 93.3%  | 93.2%    |
| Random Forest          | 88.5%     | 88.6%     | 88.5%  | 88.4%    |
| Logistic Regression    | 84.2%     | 85.6%     | 84.2%  | 84.0%    |
| Support Vector Machine | 83.7%     | 85.3%     | 83.7%  | 83.5%    |

Deep Learning Model (Keras)

- Accuracy: 87%

NLP Model (Feedback Sentiment)

- Accuracy: 93.5% (imbalanced dataset)
- Strong performance in Low Risk detection
- Needs improvement for High Risk classification

## Ethics in AI

- Data anonymization (no student names/IDs)
- Bias monitoring (gender, socioeconomic factors)
- Predictions support teacher decisions, not replace them
- Transparency with explainable predictions
