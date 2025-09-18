import pandas as pd
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text Preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# 📂 Load your dataset (adjust file path if needed)
df = pd.read_csv("spam.csv", encoding="latin-1")  
df = df[['v1','v2']]   # keep only useful columns
df.rename(columns={'v1':'label','v2':'message'}, inplace=True)

# Preprocess
df['transformed'] = df['message'].apply(transform_text)

# Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transformed'])
y = df['label'].map({'ham':0, 'spam':1})

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# ✅ Save fitted vectorizer & model
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model and Vectorizer trained & saved successfully!")
