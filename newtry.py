import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sea

train = pd.read_json("train.json")
train= train[["listing_id","description","interest_level"]]
train["flag"]="train"

test = pd.read_json("test.json")
test= test[["listing_id","description"]]
test["flag"]="test"

full=pd.concat([train, test], axis=0)

stemmer=PorterStemmer()
#stemmer=LancasterStemmer()

def clean(x):
    regex=re.compile(r"[^a-zA-Z]")
    i = regex.sub(' ', x).lower()
    i = i.split(" ")
    i = [stemmer.stem(l) for l in i]
    i = " ".join([l.strip() for l in i if (len(l) > 2)])    # keeping words that have length greater than 2
    return i

"""def clean(s):
# Remove html tags:
cleaned = re.sub(r"(?s)<.*?>", " ", s)
# Keep only regular chars:
cleaned = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", cleaned)
# Remove unicode chars
cleaned = re.sub("\\\\u(.){4}", " ", cleaned)
# Remove extra whitespace
cleaned = re.sub(r"&nbsp;", " ", cleaned)
cleaned = re.sub(r"\s{2,}", " ", cleaned)
return s.strip()"""

full["description_new"]=full.description.apply(lambda x: clean(x))
ct=CountVectorizer(stop_words="english", max_features=300)
full_sparse=ct.fit_transform(full.description_new)
colname=["desc_" + i for i in ct.get_feature_names()]
count_df=pd.DataFrame(data=full_sparse.todense(), columns=colname)
full=pd.concat([full.reset_index(), count_df], axis=1)
labels={"high":0, "medium":1, "low":2}
train=(full[full.flag=="train"])
test=(full[full.flag=="test"])
train.loc["interest_level"]=train.interest_level.apply(lambda x: labels[x])
features=train.drop(['interest_level','flag','listing_id','description','index','description_new'], axis=1)\
    .columns.values

train.drop(['interest_level','flag','listing_id','description','index','description_new'], axis=1)\
    .to_csv("desc_features_.csv", sep=",")
test.drop(['interest_level','flag','listing_id','description','index','description_new'], axis=1)\
    .to_csv("desc_features_test.csv", sep=",")


def run_mod(train_X, test_X, train_Y):
    reg=GB(max_features="auto", n_estimators=300, random_state=1)
    reg.fit(train_X, train_Y)
    pred=reg.predict_proba(test_X)
    #pred=reg.predict(test_X)         # predict class
    imp=reg.feature_importances_
    return pred, imp


def cross_val(train, feature, split):
    cv_scores=[]
    importances=[]
    train_x=train[feature].as_matrix()
    train_y=train["interest_level"].as_matrix()
    #test_x=test[feature].as_matrix()
    skf=StratifiedKFold(n_splits=split, shuffle=True, random_state=1)
    for train_idx, val_idx in skf.split(train_x, train_y):
        train_x_, test_x_=train_x[train_idx, :], train_x[val_idx, :]
        train_y_, test_y_=train_y[train_idx, ], train_y[val_idx, ]
        pred, imp=run_mod(train_x_, test_x_, train_y_)
        cv_scores.append(log_loss(test_y_, pred))
        importances.append(imp)
        #cv_scores.append(np.mean(np.array(pred)!=np.array(test_y_)))      # error
    return np.mean(cv_scores), importances

loss, imp=cross_val(train, features, 3)  # log loss: 0.730792105001  cv error: 0.301629125874
print(loss)                             #  300 features: 0.729377748743   0.301021246541
importances=list(np.average(imp, axis=0))    # lancaster: 0.301953320958
features = ct.get_feature_names()
imp_df=pd.DataFrame(data={"words":features, "importance": importances}).sort_values(by="importance", ascending=False).\
    head(30)
plt.figure(figsize=(12,10))
sea.barplot(x=imp_df.importance, y=imp_df.words)
sea.plt.show()
