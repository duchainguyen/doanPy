# data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def _data():
    # Đọc dữ liệu từ tệp CSV
    df = pd.read_csv(r'F:\Đồ Án\kag_risk_factors_cervical_cancer.csv', sep=';')

    # Thay thế "?" bằng NaN
    df.replace('?', np.nan, inplace=True)
    # Xử lý dữ liệu thiếu
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm thử
    X_data = df[['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)']]
    y_data = df['Biopsy']
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, shuffle=True)
    
    return X_train, X_test, y_train, y_test
