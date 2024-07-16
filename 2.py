import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stemmer = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# TODO:
p = """
Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language. 
"""

sentences = nltk.sent_tokenize(p)
print(sentences)

word = nltk.word_tokenize(p)

stemmer = PorterStemmer()
for i in range(len(sentences)):
  words = nltk.word_tokenize(sentences[i])
  words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
  sentences[i] = ' '.join(words)





from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
for i in range(len(sentencess)):
  words = nltk.word_tokenize(sentencess[i])
  words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
  sentencess[i] = ' '.join(words)


# Cleaning
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
  review = re.sub('[^a-zA-Z]', ' ', sentences[i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

#Creating the Bag of words model
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  x = cv.fit_transform(corpus).toarray()



sentencess = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
  review = re.sub('[^a-zA-Z]', ' ', sentencess[i])
  review = review.lower()
  review = review.split()
  review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

#Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x = tf.fit_transform(corpus).toarray()
