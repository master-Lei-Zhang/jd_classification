import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

frames = []
file_list = ['labels/jd_labeled_1.csv', 'labels/jd_labeled_2.csv', 'labels/jd_labeled_3.csv', 'labels/jd_labeled_4.csv', 'labels/jd_labeled_round2_1.csv',
             'labels/jd_labeled_round2_2.csv', 'labels/jd_labeled_round2_3.csv', 'labels/jd_labeled_mannual_keywords_round1.csv', 'labels/jd_labeled_mannual_browse.csv']
for file in file_list:
    df_mannual = pd.read_csv(file)
    frames.append(df_mannual)
df_mannual = pd.concat(frames)
df_mannual.to_csv('jd_labeled_mannual.csv', index=False)
df_mannual = pd.read_csv('jd_labeled_mannual.csv')

# combine df
frames = []
file_list = ['labels/label_sample_all_sure2.csv']
for file in file_list:
    df = pd.read_csv(file)
    frames.append(df)
df = pd.concat(frames)

# Remove test sample from training sample
index_drop = []
for i in range(len(df)):
    if df['description'][i] in df_mannual['description'].tolist():
        index_drop.append(i)

print(len(df))
df = df.drop(df.index[index_drop])
print(len(df))


df.to_csv('jd_labeled_all.csv', index=False)
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

# This is the normal Tfidf setup to derive volcabulary automatically

# vectorizer = TfidfVectorizer(
#    ngram_range=(1,2),
#    binary=True,
#    token_pattern = '[a-z]+\w*',
#    stop_words="english",
#    min_df=50,
#    max_df=0.5#,
#    #max_features=500
# )

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
y_mannual = df_mannual['y']
res_mannual = model.predict(df_mannual['description'])
print(classification_report(y_mannual, res_mannual))

# save the model to disk
filename = 'jd_classifier_model.sav'
pickle.dump(model, open(filename, 'wb'))

#from sklearn.metrics import roc_auc_score
#y_mannual_prob = model.predict_proba(df_mannual['description'])
#y_mannual_prob = np.transpose([pred[:, 1] for pred in y_mannual_prob])
#roc_auc=roc_auc_score(y_mannual, y_mannual_prob, multi_class='ovr')
#print(roc_auc_score)
#import pdb;pdb.set_trace()