import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('method', type=str)
args = parser.parse_args()

def supp_prox(context, labels, test, method='proximity',
              num_iters=10, subsample_size=1, seed=42):
    classes = np.unique(labels)
    class_lengths = np.array([len(context[labels == c]) for c in classes])
    support = []
    distances = []
    rng = np.random.default_rng(seed=seed)

    if method=='proximity':
        for c in classes:
            train_pos = context[labels==c]
            pos_dists = np.zeros(shape=(len(test), num_iters))

            train_pos_sampled = np.zeros(shape=(num_iters, subsample_size, context.shape[1]))
            for j in range(num_iters):
                train_pos_sampled[j] = rng.choice(train_pos, size=subsample_size,
                                                  replace=False, shuffle=True)
            for i in range(len(test)):
                for j in range(num_iters):
                    low = np.minimum(test[i], np.min(train_pos_sampled[j], axis=0))
                    high = np.maximum(test[i], np.max(train_pos_sampled[j], axis=0))
                    pos_mask = (~((low <= train_pos) & (train_pos <= high))).sum(axis=1) == 0
                    pos_dists[i][j] = 1-np.linalg.norm(train_pos[pos_mask]-test[i], axis=1).mean() / np.sqrt(context.shape[1])
            distances.append(pos_dists)
    else:
        for c in classes:
            train_pos = context[labels==c]
            train_neg = context[labels!=c]

            positive_support = np.zeros(shape=(len(test), num_iters))
            positive_counter = np.zeros(shape=(len(test), num_iters))
            pos_dists = np.zeros(shape=(len(test), num_iters))
            train_pos_sampled = np.zeros(shape=(num_iters, subsample_size, context.shape[1]))
            for j in range(num_iters):
                train_pos_sampled[j] = rng.choice(train_pos, size=subsample_size,
                                                  replace=False, shuffle=True)
            for i in range(len(test)):
                for j in range(num_iters):
                    low = np.minimum(test[i], np.min(train_pos_sampled[j], axis=0))
                    high = np.maximum(test[i], np.max(train_pos_sampled[j], axis=0))
                    pos_mask = (~((low <= train_pos) & (train_pos <= high))).sum(axis=1) == 0
                    cnt_mask = (~((low <= train_neg) & (train_neg <= high))).sum(axis=1) == 0
                    pos_dists[i][j] = 1-np.linalg.norm(train_pos[pos_mask]-test[i], axis=1).mean() / np.sqrt(context.shape[1])
                    
                    positive_support[i][j] = pos_mask.sum()
                    positive_counter[i][j] = cnt_mask.sum()

            support.append(np.array((positive_support, positive_counter)))
            distances.append(pos_dists)

    return support, distances, classes, class_lengths

def proximity_based(proximity, support, classes, class_lengths):
    preds = np.full(proximity[0].shape[0], -1.)
    criter = np.zeros(shape=(len(classes),proximity[0].shape[0]))
    for j in range(len(classes)):
        criter[j] = proximity[j].mean(axis=1)
    criter = criter.T
    pred_mask = (np.max(criter,axis=1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def proximity_non_falsified(proximity, support, classes, class_lengths):
    preds = np.full(proximity[0].shape[0], -1.)
    criter = np.zeros(shape=(len(classes),proximity[0].shape[0]))
    for j in range(len(classes)):
        criter[j] = (proximity[j]*(support[j][1] == 0)).mean(axis=1)
    criter = criter.T
    pred_mask = (np.max(criter,axis=1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def proximity_support(proximity, support, classes, class_lengths):
    preds = np.full(proximity[0].shape[0], -1.)
    criter = np.zeros(shape=(len(classes),proximity[0].shape[0]))
    for j in range(len(classes)):
        criter[j] = (support[j][0]*proximity[j]*(support[j][1] == 0)).sum(axis=1)
    criter = criter.T / class_lengths
    pred_mask = (np.max(criter,axis=1)[:,None]==criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds

def prox_cv(X,y, method, num_iters=10, subsample_size=1):    
    kf = StratifiedKFold(n_splits=5, random_state=1998, shuffle=True)
    Accuracy = 0.
    F1_score = 0.
    Unclassified = 0.
    exec_time = 0.
    for train_index, test_index in kf.split(X,y):
        start = time.time()
        support, proximity, classes, cl = supp_prox(X[train_index], y[train_index],
                                                    X[test_index], method=method,
						    num_iters=num_iters, subsample_size=subsample_size)
        if method == 'proximity':
            predictions = proximity_based(proximity, support, classes, cl)
        elif method == 'proximity-non-falsified':
            predictions = proximity_non_falsified(proximity, support, classes, cl)
        elif method == 'proximity-support':
            predictions = proximity_support(proximity, support, classes, cl)
        end = time.time()
        Accuracy+=accuracy_score(y[test_index], predictions)
        F1_score+=f1_score(y[test_index], predictions, average='macro')
        Unclassified += (predictions==-1).sum()
        exec_time+=end-start
    Accuracy=round(Accuracy/kf.get_n_splits(),4)
    F1_score=round(F1_score/kf.get_n_splits(),4)
    Unclassified = round(Unclassified/kf.get_n_splits(),4)
    exec_time=round(exec_time/kf.get_n_splits(),2)
    return np.array([Accuracy, F1_score, Unclassified, exec_time])

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
best_score = 0.

for n in n_iters:
    for s in s_size:
        res = prox_cv(X_train,y_train,method=args.method,num_iters=n,subsample_size=s)
        if res[1] > best_score:
            best_n = n
            best_s = s
            best_score = res[1]
start = time.time()
support, proximity, classes, cl = supp_prox(X_train, y_train,
                                            X_test, method=args.method, 
                                            num_iters=best_n, subsample_size=best_s)
if args.method == 'proximity':
    predictions = proximity_based(proximity, support, classes, cl)
elif args.method == 'proximity-non-falsified':
    predictions = proximity_non_falsified(proximity, support, classes, cl)
elif args.method == 'proximity-support':
    predictions = proximity_support(proximity, support, classes, cl)
end = time.time()
result=pd.DataFrame([[round(accuracy_score(y_test, predictions),4), round(f1_score(y_test, predictions, average='macro'),4), (best_n,best_s), round(end-start,2)]],columns=["Accuracy","F1 score", "Size","time (sec.)"])

#         result.append(res)
# result=pd.DataFrame(result,columns=["Accuracy","F1 score", "Unclassified","time (sec.)"], 
#                     index=pd.MultiIndex.from_product([n_iters, s_size], names=["Number of iterations","Subsample size"]))
# result[["Accuracy", "F1 score", "Unclassified"]] = result[["Accuracy", "F1 score", "Unclassified"]].round(4)
# result["time (sec.)"] = result["time (sec.)"].round(2)
result.to_csv(f"results-randomized/{args.dataset}-res-{args.method}.csv", index=True)
#               index_label=["Number of iterations","Subsample size"])
