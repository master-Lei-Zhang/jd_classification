import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

df = pd.read_csv('jd_labeled_all.csv')

# Read vocalbulary from files
df_words_da = pd.read_csv('da_keywords.txt')
df_words_ds = pd.read_csv('ds_keywords.txt')
df_words_mle = pd.read_csv('mle_keywords.txt')
df_words_de = pd.read_csv('de_keywords.txt')
words = df_words_da['words'].tolist()+df_words_ds['words'].tolist() + \
    df_words_mle['words'].tolist()+df_words_de['words'].tolist()
words.sort()
print('Before:', len(words))
words = list(set(words))
print('After:', len(words))
print(len(words))
n = len(words)
vocabulary = {}

for i in range(n):
    vocabulary[words[i]] = i

# This is the Tfidf setup with volcabulary input
vectorizer = TfidfVectorizer(
    vocabulary=vocabulary)

X = df["description"]
y = df.y

counts = df.y.value_counts()
weight = {}
n_all = len(df)
for c in df.y.unique():
    weight[c] = n_all/counts[c]
print(weight)
weight = 'balanced'

model = Pipeline([('cv', vectorizer),
                  ('lr', LogisticRegression(penalty="l1",
                                            C=10,
                                            solver='saga',
                                            #solver = 'liblinear',
                                            # solver='newton-cg',
                                            multi_class='auto',
                                            class_weight=weight))])

model.fit(df.description, df.y)

print(classification_report(df.y, model.predict(df.description)))

# save the model to disk
filename = 'jd_classifier_model.sav'
pickle.dump(model, open(filename, 'wb'))