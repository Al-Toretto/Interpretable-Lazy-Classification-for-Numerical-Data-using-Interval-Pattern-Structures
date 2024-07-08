import fcalc
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('method', type=str)
args = parser.parse_args()

def model_test_CV(X, y, method=args.method, alpha=0.,
                  randomize=False, num_iters=10, subsample_size=1e-2,
                  n_splits=5, seed=42):
    kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    Accuracy = []
    F1_score = []
    Unclassified = []
    exec_time = []
    for train_index, test_index in kf.split(X,y):
        pat_cls = fcalc.classifier.PatternClassifier(X[train_index], y[train_index], 
                                                     method=method, alpha=alpha,
                                                     randomize=randomize, num_iters=num_iters, 
                                                     subsample_size=subsample_size)
        start = time.time()
        pat_cls.predict(X[test_index])
        end = time.time()
        Accuracy.append(accuracy_score(y[test_index], pat_cls.predictions))
        F1_score.append(f1_score(y[test_index], pat_cls.predictions, average='macro'))
        Unclassified.append((pat_cls.predictions==-1).sum()/pat_cls.predictions.shape[0])
        exec_time.append(end-start)
    
    Accuracy.append(np.mean(Accuracy)); F1_score.append(np.mean(F1_score))
    Unclassified.append(np.mean(Unclassified)); exec_time.append(np.mean(exec_time))
    return pd.DataFrame(zip(Accuracy,F1_score,Unclassified,exec_time),
                        columns=["Accuracy","F1 score", "Unclassified", "time (sec.)"],
                        index=[x+1 for x in range(kf.get_n_splits())]+["mean"])

df = pd.read_csv(f'data/{args.dataset}.csv')
X = df.drop('class', axis=1).values
y = LabelEncoder().fit_transform(df['class'].values)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1998, stratify=y)

n_iters = np.arange(10,51,10)
s_size = np.arange(1,6) 
# result = []
best_n = 0
best_s = 0
best_score = 0.0

for n in n_iters:
    for s in s_size:
        res = model_test_CV(X_train,y_train,randomize=True,num_iters=n,subsample_size=s, seed=1998)
        if res.loc["mean"].values[1] > best_score:
            best_score = res.loc["mean"].values[1]
            best_n = n
            best_s = s

pat_cls = fcalc.classifier.PatternClassifier(X_train, y_train, 
                                             method=args.method, randomize=True,
					     num_iters=best_n, 
                                             subsample_size=best_s)
start = time.time()
pat_cls.predict(X_test)
end = time.time()
result = pd.DataFrame([[round(accuracy_score(y_test, pat_cls.predictions),4), round(f1_score(y_test, pat_cls.predictions, average='macro'),4), (best_n, best_s), round(end-start, 2)]],
		      columns=["Accuracy","F1 score", "size","time (sec.)"]) 

        # result.append(res.loc["mean"].values)
# result=pd.DataFrame(result,columns=["Accuracy","F1 score", "Unclassified","time (sec.)"], 
#                     index=pd.MultiIndex.from_product([n_iters, s_size], names=["Number of iterations","Subsample size"]))
# result[["Accuracy", "F1 score", "Unclassified"]] = result[["Accuracy", "F1 score", "Unclassified"]].round(4)
# result["time (sec.)"] = result["time (sec.)"].round(2)
result.to_csv(f"results-randomized/{args.dataset}-{args.method}-res.csv", index=False)
#               index_label=["Number of iterations","Subsample size"])
