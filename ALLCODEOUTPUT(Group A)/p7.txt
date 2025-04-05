import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Step 1: Load the sample document
with open('text.txt', 'r') as file:
    text = file.read()

print("\nOriginal Text:\n", text)

# Step 2: Tokenization
tokens = word_tokenize(text)
print("\nTokens:\n", tokens)

# Step 3: POS Tagging
pos_tags = pos_tag(tokens)
print("\nPOS Tags:\n", pos_tags)

# Step 4: Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\nAfter Stop Words Removal:\n", filtered_tokens)

# Step 5: Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\nStemmed Words:\n", stemmed_words)

# Step 6: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\nLemmatized Words:\n", lemmatized_words)

# Step 7: TF and IDF Calculation using sklearn
# We'll use the original sentence and another one for comparison
documents = [
    text,
    "Chatbots use natural language processing to respond to user queries."
]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

# Print TF-IDF values
print("\nTF-IDF Matrix:")
for word, index in tfidf.vocabulary_.items():
    print(f"{word}: {tfidf_matrix[0, index]:.4f}")
