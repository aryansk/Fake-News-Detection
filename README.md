# NewsTruthDetector 🕵️‍♀️🔍

## 📰 Project Overview

A sophisticated machine learning solution to detect fake news using multiple classification algorithms. Identify the credibility of news articles with advanced text analysis techniques!

## 🛠️ Key Features

- Multiple machine learning classifiers
- Text preprocessing and feature extraction
- TF-IDF vectorization
- Comprehensive model evaluation
- Manual news verification

## 🤖 Algorithms Used

1. Logistic Regression
2. Decision Tree Classifier
3. Gradient Boosting Classifier
4. Random Forest Classifier

## 📊 Performance Metrics

The system evaluates news credibility using:
- Accuracy scores
- Precision
- Recall
- F1-Score

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### Installation

```bash
git clone https://github.com/yourusername/news-truth-detector.git
cd news-truth-detector
pip install -r requirements.txt
```

## 💡 Usage

```python
# Load your news text
news_text = "Your news article text here"
manual_testing(news_text)
```

## 🔍 How It Works

1. Text Preprocessing
   - Lowercase conversion
   - Special character removal
   - URL elimination
   - Punctuation cleaning

2. Feature Extraction
   - TF-IDF Vectorization

3. Classification
   - Multiple algorithm prediction
   - Majority voting mechanism

## 🧪 Testing

Manually test news articles by running the script and inputting text for verification.

## 📝 Limitations

- Requires substantial training data
- Performance depends on dataset quality
- May have bias from training data

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines.

## 📜 License

MIT License
