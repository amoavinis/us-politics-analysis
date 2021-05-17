import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score


if __name__ == "__main__":
    CLUSTERS = 2
    REDUCTED_TEXT_FOLDER = 'data/user-clustering/'
    DIM_REDUCTION = 10
    REDUCTED_TEXT_FILE = REDUCTED_TEXT_FOLDER + 'reducted-txt-' + str(DIM_REDUCTION) + '.pkl'

    if not os.path.exists(REDUCTED_TEXT_FOLDER):
        os.mkdir(REDUCTED_TEXT_FOLDER)

    if not os.path.exists(REDUCTED_TEXT_FILE):
        # We read the inferenced sub-dataset of sentiment
        # because in that file it already has the pre-processed cleaned text.
        df_sent = pd.read_pickle('data/inferences/tweets-with-sentiment.pkl')
        cleaned_text_scalar = df_sent['cleaned_text']
        
        tfidf = TfidfVectorizer()
        svd = TruncatedSVD(n_components=10, random_state=1)
        
        tf_idf_text = tfidf.fit_transform(cleaned_text_scalar)
        reducted_text = svd.fit_transform(tf_idf_text)

        with open(REDUCTED_TEXT_FILE, 'wb') as f:
            pickle.dump(reducted_text, f)
    else:
        print("Loading reducted text...")
        with open(REDUCTED_TEXT_FILE, 'rb') as f:
            reducted_text = pickle.load(f)

    kmeans = MiniBatchKMeans(n_clusters=CLUSTERS, random_state=1, max_iter=2, verbose=1)
    
    kmeans.fit(reducted_text)

    # Save fitted model
    with open('data/user-clustering/kmeans-{}.pkl'.format(CLUSTERS), 'wb') as f:
        pickle.dump(kmeans, f)

    predicted_labels = kmeans.predict(reducted_text)

    print("Silhouette score:", silhouette_score(reducted_text, predicted_labels, random_state=1))

