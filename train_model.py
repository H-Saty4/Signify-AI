import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle, os, glob

DATA_PATH = "data_sequences_final" 
MODEL_PATH = "model_pkl" 

os.makedirs(MODEL_PATH, exist_ok=True)
print(f"[INFO] Membaca data CSV dari folder '{DATA_PATH}'...")
csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
if not csv_files:
    print(f"[ERROR] Tidak ada file .csv ditemukan di '{DATA_PATH}'. Jalankan perekam data dulu."); exit()

X_list, y_list = [], []
for file in csv_files:
    label = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    X_list.append(df.values)
    y_list.extend([label] * len(df))

if not X_list:
    print("[ERROR] Gagal memuat data dari file CSV. Pastikan file tidak kosong."); exit()

X = np.concatenate(X_list, axis=0)
y = np.array(y_list)

print(f"[INFO] Total {len(X)} sampel dimuat untuk {len(np.unique(y))} kelas.")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


min_samples_per_class = np.min(np.bincount(y_encoded))
if min_samples_per_class < 2:
    print("\n[PERINGATAN] Beberapa kelas hanya memiliki 1 sampel. stratify dinonaktifkan.")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("[INFO] Melatih model Random Forest...")
model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"\n[SUCCESS] Pelatihan Selesai. Akurasi Model: {acc * 100:.2f}%\n")

with open(os.path.join(MODEL_PATH, "signify_model_rf.pkl"), 'wb') as f: pickle.dump(model, f)
with open(os.path.join(MODEL_PATH, "label_encoder_rf.pkl"), 'wb') as f: pickle.dump(encoder, f)

print(f"[SUCCESS] Model dan encoder tersimpan di folder '{MODEL_PATH}'.")