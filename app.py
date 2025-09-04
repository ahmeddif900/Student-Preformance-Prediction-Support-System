from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import re
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.models import load_model


# Download NLTK resources (run once)
try:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except:
    pass

app = Flask(__name__)

# Load all models
models = joblib.load('models\student_model.pkl')
MODELS = {
    'Logistic Regression': models['Logistic Regression'],
    'Decision Tree': models['Decision Tree'],
    'Random Forest': models['Random Forest'],
    'SVM': models['SVM'],
    'Neural Network': tf.keras.models.load_model('models\deep_learning_model.h5')
}
scaler = joblib.load('models\scaler.pkl')

# Define the required features
REQUIRED_FEATURES = [
    'school', 'schoolsup','age', 'G1','G2',  'absences','Fjob_health', 'Fjob_services', 
    'goout',    'paid','higher', 'attendance_ratio', 'AvgGrade'
]

# Risk categories
RISK_CATEGORIES = {
    2: {'name': 'High Risk (Fail <10)', 'color': '#dc3545', 'icon': '⚠️'},
    1: {'name': 'Medium Risk (Pass 10–13)', 'color': '#ffc107', 'icon': '🔶'},
    0: {'name': 'Low Risk (Pass 14+)', 'color': '#28a745', 'icon': '✅'},
    'High': {'name': 'High Risk (Fail <10)', 'color': '#dc3545', 'icon': '⚠️'},
    'Medium': {'name': 'Medium Risk (Pass 10–13)', 'color': '#ffc107', 'icon': '🔶'},
    'low':{'name': 'Low Risk (Pass 14+)', 'color': '#28a745', 'icon': '✅'}
}

# Initialize custom cleaner using NLTK
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Custom text cleaner matching your pipeline"""
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove non-letters
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Load NLP pipeline if available
try:
    NLP_PIPELINE = joblib.load('models\ logreg_pipeline.pkl')
    HAS_NLP_MODEL = True
    print("NLP pipeline loaded successfully")
except Exception as e:
    HAS_NLP_MODEL = False
    print(f"NLP pipeline not found: {e}")
    # Create a placeholder pipeline structure
    NLP_PIPELINE = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

# Feature descriptions and metadata
FEATURE_DETAILS = {
    'schoolsup': {
        'type': 'binary',
        'options': ['yes', 'no'],
        'description': 'Extra educational support',
        'placeholder': 'Does the student receive extra educational support?'
    },
    'age': {
        'type': 'numeric',
        'min': 15,
        'max': 22,
        'description': "Student's age",
        'placeholder': 'Enter age (15-22)'
    },
    'school': {
        'type': 'binary',
        'options': ['GP', 'MS'],
        'description': "Student's school",
        'placeholder': 'GP - Gabriel Pereira or MS - Mousinho da Silveira'
    },
    'absences': {
        'type': 'numeric',
        'min': 0,
        'max': 93,
        'description': 'Number of school absences',
        'placeholder': 'Enter number of absences (0-93)'
    },
    'Fjob_services': {
        'type': 'binary',
        'options': ['yes', 'no'],
        'description': "Father's job in services",
        'placeholder': 'Is father working in civil services?'
    },
    'Fjob_health': {
        'type': 'binary',
        'options': ['yes', 'no'],
        'description': "Father's job in health",
        'placeholder': 'Is father working in health care?'
    },
    'goout': {
        'type': 'numeric',
        'min': 1,
        'max': 5,
        'description': 'Going out with friends frequency',
        'placeholder': '1 (very low) to 5 (very high)'
    },
    'G2': {
        'type': 'numeric',
        'min': 0,
        'max': 20,
        'description': 'Second period grade',
        'placeholder': 'Enter G2 grade (0-20)'
    },
    'G1': {
        'type': 'numeric',
        'min': 0,
        'max': 20,
        'description': 'First period grade',
        'placeholder': 'Enter G1 grade (0-20)'
    },
    'higher': {
        'type': 'binary',
        'options': ['yes', 'no'],
        'description': 'Wants to take higher education',
        'placeholder': 'Does student want higher education?'
    },
    'paid': {
        'type': 'binary',
        'options': ['yes', 'no'],
        'description': 'Extra paid classes',
        'placeholder': 'Does student take extra paid classes?'
    },
    'attendance_ratio': {
        'type': 'numeric',
        'min': 0,
        'max': 1,
        'step': 0.01,
        'description': 'Attendance ratio (1 - absences/total_days)',
        'placeholder': 'Enter attendance ratio (0-1)'
    },
    'AvgGrade': {
        'type': 'numeric',
        'min': 0,
        'max': 20,
        'description': 'Average of G1 and G2 grades',
        'placeholder': 'Auto-calculated average grade'
    }
}

def preprocess_input(data):
    """Preprocess input data for prediction"""
    input_data = []
    for feature in REQUIRED_FEATURES:
        value = data.get(feature, '')
        
        # Handle different feature types
        if FEATURE_DETAILS[feature]['type'] == 'binary':
            input_data.append(1 if str(value).lower() == 'yes' else 0)
        elif feature == 'school':
            input_data.append(1 if str(value).upper() == 'GP' else 0)
        else:
            try:
                input_data.append(float(value))
            except:
                input_data.append(0.0)
    
    # Calculate AvgGrade if not provided
    if 'AvgGrade' in data and data['AvgGrade']:
        pass
    else:
        g1_val = float(data.get('G1', 0))
        g2_val = float(data.get('G2', 0))
        avg_grade = (g1_val + g2_val) / 2
        input_data[-1] = avg_grade
    
    return scaler.transform(np.array(input_data).reshape(1, -1))

def predict_with_model(model, input_array, model_name):
    """Make prediction with the specified model"""
    try:
        if model_name == 'Neural Network':
            # Neural network prediction - assuming it outputs probabilities for 3 classes
            prediction_proba = model.predict(input_array)
            predicted_class = np.argmax(prediction_proba, axis=1)[0]
            confidence = float(np.max(prediction_proba))
        else:
            # Scikit-learn models
            if hasattr(model, 'predict_proba'):
                prediction = model.predict(input_array)
                probability = model.predict_proba(input_array)
                confidence = float(np.max(probability))
                predicted_class = int(prediction[0])
            else:
                prediction = model.predict(input_array)
                predicted_class = int(prediction[0])
                confidence = None
        
        # Get risk category details
        risk_category = RISK_CATEGORIES.get(predicted_class, RISK_CATEGORIES[0])
        
        return predicted_class, risk_category, confidence
    except Exception as e:
        raise Exception(f"Error predicting with {model_name}: {str(e)}")

def analyze_text_nlp(text):
    """Analyze text using NLP techniques"""
    if not text or len(text.strip()) < 10:
        return None
    
    # Clean text using your custom cleaner
    cleaned_text = clean_text(text)
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Get sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    
    # TextBlob sentiment
    blob = TextBlob(text)
    blob_sentiment = blob.sentiment
    
    # Keyword analysis for educational context
    positive_edu_keywords = ['study', 'learn', 'improve', 'progress', 'understand', 'achieve', 
                           'success', 'hard work', 'dedicated', 'motivated', 'interested']
    negative_edu_keywords = ['struggle', 'difficult', 'problem', 'fail', 'poor', 'weak', 
                           'challenge', 'stress', 'anxiety', 'behind', 'lack']
    
    positive_count = sum(1 for word in positive_edu_keywords if word in text.lower())
    negative_count = sum(1 for word in negative_edu_keywords if word in text.lower())
    
    # Determine overall sentiment
    compound_score = sentiment_scores['compound']
    
    if compound_score >= 0.05:
        sentiment = 'Positive'
        sentiment_icon = '😊'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
        sentiment_icon = '😔'
    else:
        sentiment = 'Neutral'
        sentiment_icon = '😐'
    
    # Educational risk assessment based on sentiment
    if sentiment == 'Negative' or negative_count > positive_count:
        edu_risk = 'High'
        risk_icon = '⚠️'
    elif sentiment == 'Neutral' and negative_count == positive_count:
        edu_risk = 'Medium'
        risk_icon = '🔶'
    else:
        edu_risk = 'Low'
        risk_icon = '✅'
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'sentiment_icon': sentiment_icon,
        'compound_score': compound_score,
        'positive_keywords': positive_count,
        'negative_keywords': negative_count,
        'education_risk': edu_risk,
        'risk_icon': risk_icon,
        'scores': {
            'vader_positive': sentiment_scores['pos'],
            'vader_negative': sentiment_scores['neg'],
            'vader_neutral': sentiment_scores['neu'],
            'textblob_polarity': blob_sentiment.polarity,
            'textblob_subjectivity': blob_sentiment.subjectivity
        }
    }

def predict_with_nlp_pipeline(text):
    """Predict using your trained NLP pipeline"""
    if not HAS_NLP_MODEL or not text:
        return None
    
    try:
        # Clean text using your custom cleaner
        cleaned_text = clean_text(text)
        
        # Predict using the pipeline (it will handle TF-IDF and classification)
        prediction = NLP_PIPELINE.predict([cleaned_text])
        
        if hasattr(NLP_PIPELINE, 'predict_proba'):
            prediction_proba = NLP_PIPELINE.predict_proba([cleaned_text])
            confidence = float(np.max(prediction_proba))
            probabilities = prediction_proba[0].tolist()
        else:
            confidence = None
            probabilities = None
        
        return {
            'predicted_class': int(prediction[0]),
            'confidence': confidence,
            'probabilities': probabilities,
            'cleaned_text': cleaned_text
        }
    except Exception as e:
        print(f"NLP pipeline prediction error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html', 
                         features=REQUIRED_FEATURES, 
                         feature_details=FEATURE_DETAILS,
                         models=list(MODELS.keys()),
                         risk_categories=RISK_CATEGORIES,
                         has_nlp_model=HAS_NLP_MODEL)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        selected_model = data.get('model', 'Random Forest')
        nlp_text = data.get('nlp_text', '').strip()
        
        # Check if this is text-only analysis (from NLP tab)
        is_text_only_analysis = 'analyze_text_only' in data
        
        if selected_model not in MODELS and not is_text_only_analysis:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Handle text-only analysis
        if is_text_only_analysis:
            # For text-only analysis, we only use NLP results
            nlp_analysis = analyze_text_nlp(nlp_text)
            nlp_model_prediction = predict_with_nlp_pipeline(nlp_text) if nlp_text else None
            
            # Create result structure for text-only analysis
            result = {
                'is_text_only': True,
                'nlp_analysis': nlp_analysis,
                'nlp_model_prediction': nlp_model_prediction,
                'input_features': data,
                'model_used': 'Text Analysis NLP'
            }
            
            # Add predicted class and risk category if NLP prediction exists
            if nlp_model_prediction:
                result['predicted_class'] = nlp_model_prediction['predicted_class']
                result['risk_category'] = RISK_CATEGORIES.get(nlp_model_prediction['predicted_class'], RISK_CATEGORIES[1])
                result['confidence'] = nlp_model_prediction.get('confidence')
            else:
                # Fallback if no NLP prediction
                result['predicted_class'] = 1
                result['risk_category'] = RISK_CATEGORIES[1]
                result['confidence'] = None
            
            return render_template('result.html', result=result, feature_details=FEATURE_DETAILS, features=REQUIRED_FEATURES, risk_categories=RISK_CATEGORIES)
        
        # Normal structured data prediction
        input_array = preprocess_input(data)
        model = MODELS[selected_model]
        predicted_class, risk_category, confidence = predict_with_model(model, input_array, selected_model)
        
        # NLP analysis (if text provided)
        nlp_analysis = analyze_text_nlp(nlp_text)
        nlp_model_prediction = predict_with_nlp_pipeline(nlp_text) if nlp_text else None
        
        # Return results for structured data analysis
        result = {
            'is_text_only': False,
            'predicted_class': predicted_class,
            'risk_category': risk_category,
            'confidence': confidence,
            'model_used': selected_model,
            'input_features': data,
            'nlp_analysis': nlp_analysis,
            'nlp_model_prediction': nlp_model_prediction
        }
        
        return render_template('result.html', result=result, feature_details=FEATURE_DETAILS, features=REQUIRED_FEATURES, risk_categories=RISK_CATEGORIES)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        selected_model = data.get('model', 'Random Forest')
        nlp_text = data.get('nlp_text', '').strip()
        
        if selected_model not in MODELS:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Preprocess input
        input_array = preprocess_input(data)
        
        # Make prediction
        model = MODELS[selected_model]
        predicted_class, risk_category, confidence = predict_with_model(model, input_array, selected_model)
        
        # NLP analysis
        nlp_analysis = analyze_text_nlp(nlp_text)
        nlp_model_prediction = predict_with_nlp_pipeline(nlp_text) if nlp_text else None
        
        result = {
            'predicted_class': predicted_class,
            'risk_category': risk_category,
            'confidence': confidence,
            'model_used': selected_model,
            'features_used': REQUIRED_FEATURES,
            'nlp_analysis': nlp_analysis,
            'nlp_model_prediction': nlp_model_prediction
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/nlp/analyze', methods=['POST'])
def api_nlp_analyze():
    """API endpoint for standalone NLP analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Perform NLP analysis
        nlp_analysis = analyze_text_nlp(text)
        nlp_model_prediction = predict_with_nlp_pipeline(text)
        
        result = {
            'text_analysis': nlp_analysis,
            'model_prediction': nlp_model_prediction
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/compare', methods=['POST'])
def compare_models():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess input
        input_array = preprocess_input(data)
        
        # Get predictions from all models
        comparisons = []
        for model_name, model in MODELS.items():
            predicted_class, risk_category, confidence = predict_with_model(model, input_array, model_name)
            comparisons.append({
                'model': model_name,
                'predicted_class': predicted_class,
                'risk_category': risk_category,
                'confidence': confidence
            })
        
        return jsonify({
            'comparisons': comparisons,
            'input_features': {k: data.get(k, '') for k in REQUIRED_FEATURES}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


#viste http://127.0.0.1:5000 to go to web