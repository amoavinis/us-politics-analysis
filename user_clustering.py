import os
import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from tqdm import tqdm


if __name__ == "__main__":
    CLUSTERS = [i for i in range(2, 11)] # 2 to 10 clusters to evaluate
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

    scores_clusters = list()
    for cluster in tqdm(CLUSTERS):
        kmeans = KMeans(n_clusters=cluster)

        kmeans.fit(reducted_text)

        predicted_labels = kmeans.labels_

        score = calinski_harabasz_score(reducted_text, predicted_labels)
        
        score_cluster = (cluster, score)
        scores_clusters.append(score_cluster)

    print("Scores per number of clusters")
    print(scores_clusters)
