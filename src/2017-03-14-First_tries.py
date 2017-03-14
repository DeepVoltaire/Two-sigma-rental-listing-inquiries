import matplotlib
matplotlib.use("Pdf")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy import sparse
import xgboost as xgb
import matplotlib.pyplot as plt
import time

def preprocess():
    train = pd.read_json("../data/train.json")
    test = pd.read_json("../data/test.json")

    features = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

    train["num_photos"] = train["photos"].apply(len)
    test["num_photos"] = test["photos"].apply(len)

    train["num_features"] = train["features"].apply(len)
    test["num_features"] = test["features"].apply(len)

    train["num_description_words"] = train["description"].apply(lambda x: len(x.split(" ")))
    test["num_description_words"] = test["description"].apply(lambda x: len(x.split(" ")))

    train["created"] = pd.to_datetime(train["created"])
    test["created"] = pd.to_datetime(test["created"])

    # random split, no time split, so we can randomly split train/eval
    print(train["created"].min())
    print(train["created"].max())
    print(test["created"].min())
    print(test["created"].max())

    train["created_year"] = train["created"].dt.year
    test["created_year"] = test["created"].dt.year
    train["created_month"] = train["created"].dt.month
    test["created_month"] = test["created"].dt.month
    train["created_day"] = train["created"].dt.day
    test["created_day"] = test["created"].dt.day
    train["created_weekday"] = train["created"].dt.weekday
    test["created_weekday"] = test["created"].dt.weekday
    train["created_hour"] = train["created"].dt.hour
    test["created_hour"] = test["created"].dt.hour

    features.extend(["num_photos", "num_features", "num_description_words","created_year",
                     "created_month", "created_day", "listing_id", "created_hour"])

    # label encode categorical features = make them numerical even if its not making sense
    categorical = ["display_address", "manager_id", "building_id", "street_address"]
    for var in categorical:
        lbl = LabelEncoder()
        # Wir nehmen hier die Testwerte auch mit auf, auch wenn sie keine Einfluss auf das Train
        # ing haben. Man könnte die Werte, die nur im Test vorkommen einfach auf -1 setzen, aber
        # das ist hier einfacher so
        lbl.fit(list(train[var].values) + list(test[var].values))
        train[var] = lbl.transform(list(train[var].values))
        test[var] = lbl.transform(list(test[var].values))
        features.append(var)

    # Count Vectorizer auf die Features = Erstellen Features, die angeben, wie oft jedes Wort aller
    # Wörter des Corpus der Features in jedem Listing vorkommen
    # Dafür müssen wir die Liste der Features zu einem String zusammenfassen, also zuerst
    # die zusammengehörigen Featurebegriffe mit _ verbinden und dann die Liste der Features zu
    # einem String machen
    train["features"] = train["features"].apply(
        lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    test["features"] = test["features"].apply(
        lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    count_vector = CountVectorizer(stop_words="english", max_features=100)
    train_sparse = count_vector.fit_transform(train["features"])
    test_sparse = count_vector.transform(test["features"])

    # sparse und dense Daten zusammenbringen, tocsr = to compressed sparse row format
    train_X = sparse.hstack([train[features], train_sparse]).tocsr()
    test_X = sparse.hstack([test[features], test_sparse]).tocsr()

    target_num_map = {"high": 0, "medium": 1, "low": 2}
    train_y = np.array(train["interest_level"].apply(lambda x: target_num_map[x]))
    print(train_X.shape, test_X.shape)
    return train_X, test_X, train_y, test


def runXGB(train_X, train_y, test_X, test_y=None, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = 1337
    param["nthread"] = 20
    num_rounds = num_rounds
    
    dtrain = xgb.DMatrix(train_X, train_y)
    if test_y is not None:
        dtest = xgb.DMatrix(test_X, test_y)
        watchlist = [(dtrain, "train"), (dtest, "eval")]
        model = xgb.train(param, dtrain, num_rounds, watchlist, early_stopping_rounds=20, verbose_eval=50)
    else:
        dtest = xgb.DMatrix(test_X)
        model = xgb.train(param, dtrain, num_rounds)
        
    pred_test_y = model.predict(dtest)
    return pred_test_y, model

def train_and_submit(train_X, test_X, train_y, test):
    cv_scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1992)
    for train_index, eval_index in kf.split(train_X, train_y):
        tr_X, eval_X = train_X[train_index,:], train_X[eval_index,:]
        tr_y, eval_y = train_y[train_index], train_y[eval_index]
        preds, model = runXGB(tr_X, tr_y, eval_X, eval_y)
        cv_scores.append(log_loss(eval_y, preds))
        print(cv_scores)

    preds, model = runXGB(train_X, train_y, test_X, num_rounds=400)
    subm = pd.DataFrame(preds)
    subm.columns = ["high", "medium", "low"]
    subm["listing_id"] = test.listing_id.values
    subm.to_csv("xgb_copy1.csv.gz", compression="gzip", index=False)

    # Feature Importance
    gain = pd.Series(model.get_score(importance_type="gain"))*pd.Series(model.get_score(importance_type="weight"))
    gain = gain.reset_index()
    gain.columns = ["features", "gain"]
    gain.sort_values(by="gain", inplace=True)

    # top 20
    gain = gain[-30:]

    val = gain["gain"]    # the bar lengths
    pos = np.arange(len(gain))+.5    # the bar centers on the y axis

    plt.figure(figsize=(16,12))
    plt.barh(pos, val, align="center")
    plt.yticks(pos, gain.features.tolist())
    # featplot = gain.plot(kind="barh", x="features", y="gain", legend=False, figsize=(10,25))
    plt.title("XGBoost Total Gain")
    plt.xticks(size=20)
    plt.yticks(size=18)
    plt.xlabel("Total Gain")
    # fig_featplot = featplot.get_figure()
    plt.savefig("../figures/XGBOOST_GAIN_" + time.strftime("%Y_%m_%d_%H") + ".png", dpi=150, bbox_inches="tight", pad_inches=1)
    plt.show()

train_X, test_X, train_y, test = preprocess()
train_and_submit(train_X, test_X, train_y, test)


# for col in ["topleveldomain", "domain", "landingpage", "Vorwahl", "Browser", 'Channel',
#             'Devicetyp', 'Kampagne', 'Anzeigeninhalt', 'Anzeigengruppe', 'Keyword']:
#     counts = X_train.groupby(col)["mean_BZ"].count().to_frame().reset_index().rename(
#         columns={"mean_BZ": col+"_count"})
#     X_train = X_train.merge(counts, how="left", on=col)
#     X_test = X_test.merge(counts, how="left", on=col)
# #     X_test[col+"_count"] = X_test[col+"_count"].fillna(0)

#     commons = X_train[col].value_counts()[:5].index
#     X_train.loc[(X_train[col].notnull()) & (~X_train[col].isin(commons)),
#                 col] = "other"
#     X_train = pd.get_dummies(X_train, columns=[col] , prefix=col)

#     X_test.loc[(X_test[col].notnull()) & (~X_test[col].isin(commons)),
#                col] = "other"
#     X_test = pd.get_dummies(X_test, columns=[col] , prefix=col)

