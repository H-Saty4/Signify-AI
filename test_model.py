import os
import pickle
import numpy as np
import tensorflow as tf
import random

MODEL_PATH = "model_pkl"
DATA_PATH = "data_sequences_final"

print("[INFO] Memuat model AI dan encoder...")

try:
    with open(os.path.join(MODEL_PATH, 'signify_model_rf.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(MODEL_PATH, 'label_encoder_rf.pkl'), 'rb') as f:
        encoder = pickle.load(f)

    print("[SUCCESS] Model berhasil dimuat.")

except Exception as e:
    print(f"\n[FATAL ERROR] Gagal memuat model. Pastikan Anda sudah menjalankan 'train_model.py'.")
    print(f"Detail Error: {e}\n")
    exit()

if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
    print(f"[FATAL ERROR] Folder data '{DATA_PATH}' kosong.")
    exit()

random_label = random.choice(os.listdir(DATA_PATH))
label_path = os.path.join(DATA_PATH, random_label)

random_sequence_file = random.choice(os.listdir(label_path))
sequence_path = os.path.join(label_path, random_sequence_file)

print(f"\n[INFO] Menguji dengan satu sampel acak:")
print(f"  -> File: {sequence_path}")
print(f"  -> Seharusnya Terdeteksi Sebagai: '{random_label}'")

sequence_data = np.load(sequence_path)

SEQUENCE_LENGTH = 30
if len(sequence_data) > SEQUENCE_LENGTH:
    sequence_data = sequence_data[:SEQUENCE_LENGTH]
elif len(sequence_data) < SEQUENCE_LENGTH:
    padding = np.zeros((SEQUENCE_LENGTH - len(sequence_data), sequence_data.shape[1]))
    sequence_data = np.vstack((sequence_data, padding))

test_data = np.expand_dims(sequence_data, axis=0)

prediction = model.predict(test_data)[0]
confidence = np.max(prediction)
predicted_label_index = np.argmax(prediction)
predicted_label = encoder.inverse_transform([predicted_label_index])[0]

print("\n----------- HASIL INTEROGASI AI -----------")
print(f">>> Prediksi Model: '{predicted_label}'")
print(f">>> Tingkat Keyakinan: {confidence * 100:.2f}%")
print("-----------------------------------------")

if predicted_label == random_label and confidence > 0.9:
    print("\n[KESIMPULAN] Model AI Anda SEHAT dan PINTAR. Masalahnya kemungkinan ada di aplikasi penerjemah.")
else:
    print("\n[KESIMPULAN] Model AI Anda BERMASALAH. Ia tidak bisa mengenali datanya sendiri. Anda perlu melatih ulang (mungkin dengan lebih banyak data atau epoch).")