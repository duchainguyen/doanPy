# svc.py
from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
import pandas as pd
from data import _data

def _Svc():
    X_train, X_test, y_train, y_test = _data()
    clf_svc = SVC(gamma='auto')
    clf_svc.fit(X_train, y_train)
    y_pred_svc = clf_svc.predict(X_test)

    y_test = y_test.reset_index(drop=True)
    y_pred_svc = pd.Series(y_pred_svc)
    count_svc = 0
    for i in range(0, len(y_pred_svc)):
        if y_test[i] == y_pred_svc[i]:    
            count_svc = count_svc + 1
    predictCorrect = count_svc/len(y_pred_svc)
    # print("count", count_svc)
    print('Ty le du doan dung SVC là: ',predictCorrect )

    # Các độ đo 
    from sklearn.metrics import precision_score
    precision_score_svc = precision_score(y_test, y_pred_svc, average='macro')
    print('Độ chính xác Perceptron tính theo SVC: ', precision_score_svc)

    from sklearn.metrics import recall_score
    recall_score_svc = recall_score(y_test, y_pred_svc, average='macro')
    print('Độ chính xác Recall tính theo SVC: ', recall_score_svc)

    from sklearn.metrics import f1_score
    f1_score_svc = f1_score(y_test, y_pred_svc, average='macro')
    print('Độ chính xác F1 tính theo SVC: ', f1_score_svc)
    return predictCorrect,precision_score_svc, recall_score_svc, f1_score_svc
if __name__ == "__main__":
    _Svc()
