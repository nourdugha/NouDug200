# This helper file that dose the model prediction
import pickle
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


__version__ = "0.1.0"

# gives the absolute path of the parent directory of the script or module.
BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_pipeline_{__version__}.pkl","rb") as f:
    model = pickle.load(f)

classes = ["ch","cnc","ct","ft","mr","pkg"]

def process_text_string(text):
    # Step 1: Convert the text to lowercase
    text = text.lower()
    # Step 2: Remove special characters and numbers "Remove punctuation"
    text = re.sub(r'[^a-zäöüß\s]', '', text)
    # Step 3: Tokenization and remove stopwords
    stop_words = set(stopwords.words('german'))
    text = ' '.join([word for word in word_tokenize(text) if word.lower() not in stop_words])
    # Step 4: Stemming using PorterStemmer
    porter_stemmer = PorterStemmer()
    text = ' '.join([porter_stemmer.stem(word) for word in word_tokenize(text)])
    
    return text


def predict_pipeline(text):
    processed_text = process_text_string(text)
    pred = model.predict([processed_text])
    return classes[pred[0]]

