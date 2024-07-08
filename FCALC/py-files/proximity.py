import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
args = parser.parse_args()

def supp_prox(context, labels, test, method='proximity'):
    classes = np.unique(labels)
    class_lengths = np.array([len(context[labels == c]) for c in classes])
    support = []
    distances = []
    if method=='proximity':
        for c in classes:
            train_pos = context[labels==c]
            pos_dists = np.zeros(shape=(len(test), len(train_pos)))

            for i in range(len(test)):
                for j in range(len(train_pos)):
                    low = np.minimum(test[i],train_pos[j])
                    high = np.maximum(test[i],train_pos[j])
                    pos_mask = (~((low <= train_pos) & (train_pos <= high))).sum(axis=1) == 0
                    pos_dists[i][j] = 1-np.linalg.norm(train_pos[pos_mask]-test[i], axis=1).mean() / np.sqrt(context.shape[1])
            distances.append(pos_dists)
    else:
        for c in classes:
            train_pos = context[labels==c]
            train_neg = context[labels!=c]

            positive_support = np.zeros(shape=(len(test), len(train_pos)))
            positive_counter = np.zeros(shape=(len(test), len(train_pos)))
            pos_dists = np.zeros(shape=(len(test), len(train_pos)))

            for i in range(len(test)):
                for j in range(len(train_pos)):
                    low = np.minimum(test[i],train_pos[j])
                    high = np.maximum(test[i],train_pos[j])
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

def prox_cv(X,y, method):    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    Accuracy = 0.
    F1_score = 0.
    Unclassified = 0.
    exec_time = 0.
    for train_index, test_index in kf.split(X):
        start = time.time()
        support, proximity, classes, cl = supp_prox(X[train_index], y[train_index], X[test_index], method=method)
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

methods = ["proximity", "proximity-non-falsified", "proximity-support"]
result = []
for m in methods:
    start = time.time()
    support, proximity, classes, cl = supp_prox(X_train, y_train, X_test, method=m)
    if m == 'proximity':
        predictions = proximity_based(proximity, support, classes, cl)
    elif m == 'proximity-non-falsified':
        predictions = proximity_non_falsified(proximity, support, classes, cl)
    elif m == 'proximity-support':
        predictions = proximity_support(proximity, support, classes, cl)
    end = time.time()
    result.append([round(accuracy_score(y_test, predictions),4), round(f1_score(y_test, predictions, average='macro'),4),
		   round((predictions==-1).sum()/len(predictions),4), round(end-start, 2)])
    # res = prox_cv(X, y, method=m)
    # result.append(res)

result=pd.DataFrame(result, index=methods,columns=["Accuracy", "F1 score", 
                                                   "Unclassified", "time (sec.)"])

result.to_csv(f'results/{args.dataset}-prox-res.csv')
