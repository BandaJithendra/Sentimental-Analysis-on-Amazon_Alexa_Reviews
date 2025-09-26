# Sentimental Analysis on Amazon Alexa Reviews

## Project Overview

This project focuses on **sentiment analysis of customer reviews** for the Amazon Alexa product. The goal is to automatically classify each review as **positive** or **negative** using machine learning and natural language processing (NLP) techniques.

## Features

* **Preprocessing of text data** using NLP techniques:

  * Stopwords removal
  * Tokenization (`word_tokenize`)
  * Lemmatization
* **Machine learning models implemented**:

  * Logistic Regression
  * Multinomial Naive Bayes (MultinomialNB)
  * Random Forest Classifier
* **Best performing model:** Random Forest (saved using `pickle` for future use)

## Model Performance

| Model               | Training Accuracy (%) | Testing Accuracy (%) |
| ------------------- | --------------------- | -------------------- |
| Logistic Regression | 94.97                 | 88.94                |
| MultinomialNB       | 93.53                 | 90.85                |
| Random Forest       | 98.67                 | 91.76                |

* Random Forest gave the highest accuracy on the testing set, so it was chosen as the final model.

## Technologies Used

* **Programming Language:** Python
* **Libraries:** scikit-learn, NLTK, pandas, NumPy, pickle
* **Techniques:** NLP preprocessing, classification models, model serialization

## Usage

1. Clone the repository:

```bash
git clone https://github.com/BandaJithendra/Sentimental-Analysis-on-Amazon_Alexa_Reviews.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script to classify new reviews using the saved Random Forest model:

```python
import pickle

# Load the saved model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict sentiment on new data
predictions = model.predict(new_review_vectors)
```

## Conclusion

This project demonstrates how NLP techniques and machine learning models can be applied to **analyze customer feedback**. Among the models tested, **Random Forest** provided the most accurate results for classifying Amazon Alexa reviews into positive or negative sentiments.
