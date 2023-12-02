import gensim.downloader as api

# Download the "Google News" Word2Vec model
model = api.load("word2vec-google-news-300")

# Now, 'model' is the loaded Word2Vec model

