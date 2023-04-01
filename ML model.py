
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# load the dataset
df = pd.read_csv('/kaggle/input/fake-news/FakeNewsNet.csv')

# extract the labels and text content from the dataset
labels = df['real'].tolist()
articles = df['title'].tolist()

# convert the labels to 0 for fake and 1 for real
labels = [0 if label == 'fake' else 1 for label in labels]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(articles, labels, test_size=0.2, random_state=42)

# create a TfidfVectorizer to convert the text content into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# train a Multinomial Naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# get the URL input from the user
url = input("Enter a news article URL: ")

try:
    # download the webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # extract the text content from the webpage
    text = ''
    for p in soup.find_all('p'):
        text += p.get_text()

    # remove any special characters and convert to lowercase for comparison
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()

    # use the trained model to predict if the article is fake or real
    X_tfidf = tfidf_vectorizer.transform([text])
    prediction = clf.predict(X_tfidf)[0]

    if prediction == 0:
        print("This news article is fake.")
    else:
        print("This news article is real.")

except requests.exceptions.MissingSchema:
    print("Given News is fake")

except requests.exceptions.ConnectionError:
    print("Could not connect to the server. Please check your internet connection and try again.")
