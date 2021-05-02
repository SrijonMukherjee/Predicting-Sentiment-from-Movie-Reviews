# Predict Sentiment from Movie Reviews
Sentiment analysis using IMDB dataset with CNN and LSTM

- Get data: download [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) and [Glove embeddings](http://nlp.stanford.edu/data/glove.6B.zip)
- Modify variables dataset_dir and glove_dir in saveData.py with your absolute path to dataset and embedding directory
- Run saveData.py to load the dataset and embedding
- Then run one of the following: CNNnonstatic.py, CNNstatic.py, LSTM.py, CNN_LSTM.py
