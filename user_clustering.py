import os
import pickle
from numpy import tile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm


def get_preprocessed_users_text():
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
    
    return reducted_text
            

def visually_determine_best_k_clusters(reducted_text):
    ch_scores_clusters = list()
    db_scores_clusters = list()
    inertias = list()
    for k in tqdm(CLUSTERS):
        if os.path.exists('pretrained-models/user-clustering/{}-means.pkl'.format(k)):
            print("Loading {}-means model".format(k))
            kmeans = pickle.load(open('pretrained-models/user-clustering/{}-means.pkl'.format(k), 'rb'))
        else:
            print("Training {}-means model".format(k))
            kmeans = KMeans(n_clusters=k, random_state=1)
            kmeans.fit(reducted_text)
            pickle.dump(kmeans, open('pretrained-models/user-clustering/{}-means.pkl'.format(k), 'wb'))

        predicted_labels = kmeans.labels_
        inertia = kmeans.inertia_

        ch_score = calinski_harabasz_score(reducted_text, predicted_labels)
        ch_scores_clusters.append(ch_score)

        db_score = davies_bouldin_score(reducted_text, predicted_labels)
        db_scores_clusters.append(db_score)

        inertias.append(inertia)

    sub_titles = [
        'Number of clusters VS Inertia',
        'Number of clusters VS Calinski-Harabasz (More is better)',
        'Number of clusters VS Davies-bouldin (Less is better)'
    ]
    fig = make_subplots(rows=3, cols=1, subplot_titles=sub_titles)
    fig.add_trace(go.Scatter(x=CLUSTERS, y=inertias), row=1, col=1)
    fig.add_trace(go.Scatter(x=CLUSTERS, y=ch_scores_clusters), row=2, col=1)
    fig.add_trace(go.Scatter(x=CLUSTERS, y=db_scores_clusters), row=3, col=1)
    fig.show()


def get_wordclouds_in_clusters(k):
    pass


if __name__ == "__main__":
    CLUSTERS = [i for i in range(2, 11)] # 2 to 10 clusters to evaluate
    REDUCTED_TEXT_FOLDER = 'data/user-clustering/'
    DIM_REDUCTION = 10
    REDUCTED_TEXT_FILE = REDUCTED_TEXT_FOLDER + 'reducted-txt-' + str(DIM_REDUCTION) + '.pkl'

    data = get_preprocessed_users_text()
    visually_determine_best_k_clusters(data)
    