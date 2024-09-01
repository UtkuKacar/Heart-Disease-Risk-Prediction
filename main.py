import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Veri setini oku
data = pd.read_csv("heart.csv")

# Veri setinin ilk 5 satırına bak
print("Veri Setinin İlk 5 Satırı:")
print(data.head())
# print(data.info())

# Eksik verilerin sayısına bak
missing_value = data.isnull().sum()
print("\n Eksik Veriler:")
print(missing_value)

# Sütun bazında benzersiz değer sayısına bak
unique_counts = data.nunique()
print("\n Sütundaki benzersiz değer sayısı:")
print(unique_counts)

# Veri tipleri kontrol et
data_types = data.dtypes
print("\n Veri Tipleri:")
print(data_types)

# Kategorik verileri sayısal verilere dönüştür (One-Hot Encoding)
data_encoded = pd.get_dummies(data, columns= ["cp", "restecg", "slope", "thal", "ca"])

# One-Hot Encoding sonrası ilk 5 satıra bak
print("\n One-Hot Encoding sonrası ilk 5 satır:")
print(data_encoded.head())

# Sayısal verileri standartlaştırma
numeric_colums = ["age", "trestbps", "chol", "thalach", "oldpeak"]
scaler = StandardScaler()
data_encoded[numeric_colums] = scaler.fit_transform(data_encoded[numeric_colums])

# Veri setini özellikler (X) ve hedef değişken (Y) olarak ayır
X = data_encoded.drop("target", axis= 1)
y = data_encoded["target"]

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

# Eğitim ve test setlerinin boyutlarını kontrol et
print("\n Eğitim Seti Boyutu:", X_train.shape)
print("\n Test Seti Boyutu:", X_test.shape)

# Logistic Regression modelini oluştur ve eğit
model = LogisticRegression(max_iter= 1000)
model.fit(X_train, y_train)

# Modeli test seti üzerinden değerlendir
y_pred = model.predict(X_test)

# Model doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print("\n Model Accuracy Score:", accuracy)

# Sınıflandırma raporu
print("\n Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Karmaşıklık matrisi
print("\n Karmmaşıklık Matrisi:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Karmaşıklık matrisini görselleştir
plt.figure(figsize= (6, 4))
sns.heatmap(cm, annot= True, fmt= "d", cmap= "Blues", xticklabels= ["No Disease", "Disease"], yticklabels= ["No Disease", "Disease"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.title("Karmaşıklık Matrisi")
plt.show()

# ROC eğrisi ve AUC score hesapla
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ROC eğrisi görselleştir
plt.figure(figsize= (6, 4))
plt.plot(fpr, tpr, color= "blue", lw= 2, label= "ROC Eğrisi (AUC = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color= "gray", lw= 2, linestyle= "--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Yanlış Pozitif Oranı")
plt.ylabel("Doğru Pozitif Oranı")
plt.title("ROC Eğrisi")
plt.legend(loc= "lower right")
plt.show()

# Sayısal özelliklerin histogram grafiği
data_encoded[numeric_colums].hist(bins= 15, figsize= (10, 6), layout= (2, 3))
plt.suptitle("Sayısal Özelliklerin Dağılımı")
plt.show()

# Veri üzerinden test etme
sample_data = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])  # Tahminin 0 olması gerek
sample_data_encoded = pd.DataFrame(sample_data, columns= ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
sample_data_encoded = pd.get_dummies(sample_data_encoded, columns= ["cp", "restecg", "slope", "thal", "ca"])

# Eksik verileri 0 ile doldur
expected_colums = X.columns
missing_cols = set(expected_colums) - set(sample_data_encoded.columns)

for col in missing_cols:
    sample_data_encoded[col] = 0

# Sütunları aynı sıraya getir
sample_data_encoded = sample_data_encoded[expected_colums]

# Verileri standartlaştır
sample_data_encoded[numeric_colums] = scaler.transform(sample_data_encoded[numeric_colums])

# Tahmin yap
prediction = model.predict(sample_data_encoded)
prediction_proba = model.predict_proba(sample_data_encoded)

print("\n Tahmin (1 - Hastalık Var, 0 - Hastalık Yok):", prediction[0])
print("\n Hastalık Olma Olasılığı:", prediction_proba[0][1])
print("\n Hastalık Olmama Olasılığı:", prediction_proba[0][0])