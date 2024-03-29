import pandas as pd

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

dataset = pd.read_excel("finalh.xlsx")

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,4), min_df=0)
tfidf_matrix = tf.fit_transform(dataset['description'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in dataset.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], dataset['id'][i]) for i in similar_indices]

    results[row['id']] = similar_items[1:]
    
def item(id):
    return dataset.loc[dataset['id'] == id]['description'].tolist()[0].split(' - ')[0]

def recommend(item_id, num):
    print("Recommending " + str(num) + " person with similar interest as " + item(item_id))
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]))

recommend(item_id=103, num=6)
