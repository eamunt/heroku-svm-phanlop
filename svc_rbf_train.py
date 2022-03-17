# Nạp các gói thư viện cần thiết
from math import gamma
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


df = pd.read_csv("data/data-set.csv")
X = df.iloc[:,[1,3,4]]
y = df.nhom

scale = X.astype('float32')

# scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(scale)

X_new = pd.DataFrame(scaled, columns = ['gioitinh', 'diemToan', 'diemVan'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state= 390)

#Cân bằng dữ liệu bằng SMOTE
smt = SMOTE(random_state=102)
X_train, y_train = smt.fit_resample(X_train, y_train)

# Xây dựng bằng mô hình svc-rbf
rbf_svc = SVC(kernel='rbf', C = 1000, gamma=100)
rbf_svc.fit(X_train, y_train)
# Dự đoán nhãn tập kiểm tra
y_pred = rbf_svc.predict(X_test)
print("y_pred:",y_pred)
# accuracy score svc-rbf
a_s = accuracy_score(y_test, y_pred)
print("accuracy_score:", a_s)

#filename = 'svc_rbf_model_df.pkl'
#pickle.dump(rbf_svc, open(filename, 'wb'))