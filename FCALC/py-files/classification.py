import fcalc
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
args = parser.parse_args()

def model_test_CV(X, y, cat_cols=None, method="standard", alpha=0., n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    Accuracy = []
    F1_score = []
    exec_time = []
    Unclassified = []
    for train_index, test_index in kf.split(X):
        pat_cls = fcalc.classifier.PatternClassifier(X[train_index], y[train_index], 
                                                     categorical=cat_cols, method=method,
                                                     alpha=alpha)
        start = time.time()
        pat_cls.predict(X[test_index])
        end = time.time()
        Accuracy.append(accuracy_score(y[test_index], pat_cls.predictions))
        F1_score.append(f1_score(y[test_index], pat_cls.predictions, average='macro'))
        Unclassified.append((pat_cls.predictions==-1).sum()/pat_cls.predictions.shape[0])
        exec_time.append(end-start)
    
    Accuracy.append(round(np.mean(Accuracy),4)); F1_score.append(round(np.mean(F1_score),4))
    Unclassified.append(round(np.mean(Unclassified),4)); exec_time.append(round(np.mean(exec_time),4))
    return pd.DataFrame(zip(Accuracy,F1_score,Unclassified,exec_time),
                        columns=["Accuracy","F1 score", "Unclassified", "time (sec.)"],
                        index=[x+1 for x in range(kf.get_n_splits())]+["mean"])

df = pd.read_csv(f'data/{args.dataset}.csv')
X = df.drop('class', axis=1).values
y = LabelEncoder().fit_transform(df['class'].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1998, stratify=y)
methods = ["standard", "standard-support", "ratio-support"]
result = []

for m in methods:
    pat_cls = fcalc.classifier.PatternClassifier(X_train, y_train, method=m)
    start = time.time()
    pat_cls.predict(X_test)
    end = time.time()
    result.append([round(accuracy_score(y_test, pat_cls.predictions), 4), round(f1_score(y_test, pat_cls.predictions, average='macro'), 4),
		  round((pat_cls.predictions==-1).sum()/pat_cls.predictions.shape[0], 4), round(end-start, 2)])
    # res = model_test_CV(X, y, method=m)
    # result.append(res.loc["mean"].values)

result=pd.DataFrame(result, index=methods,columns=["Accuracy", "F1 score", 
                                                   "Unclassified", "time (sec.)"])

result.to_csv(f'results/{args.dataset}-res.csv')
