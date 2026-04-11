# src/model_architectures.py
# ============================================================
# [TV2] Kiến trúc các mô hình Deep Learning
#   - CNN: 3 lớp tích chập + 3 lớp pooling + fully connected
#   - MLP: 2 lớp ẩn (hidden layers)
# Tham khảo: Ding Chen et al., J. Phys.: Conf. Ser. 1757 (2021) 012055
# ============================================================

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from config import VECTOR_SIZE, MAX_SEQUENCE_LENGTH, NUM_CLASSES


# ------------------------------------------------------------
# 1. MÔ HÌNH CNN
# ------------------------------------------------------------
def build_cnn_model(input_shape=(MAX_SEQUENCE_LENGTH, VECTOR_SIZE)):
    """
    Xây dựng mô hình CNN để phát hiện SQL Injection.

    Kiến trúc (theo bài báo):
      - Conv1D layer 1 : 16 kernels, size 3
      - MaxPooling1D   : pool_size 2
      - Conv1D layer 2 : 32 kernels, size 4
      - MaxPooling1D   : pool_size 2
      - Conv1D layer 3 : 64 kernels, size 5
      - MaxPooling1D   : pool_size 2
      - Flatten
      - Dense (fully connected) + ReLU
      - Dropout
      - Dense output   + Softmax

    Args:
        input_shape: (MAX_SEQUENCE_LENGTH, VECTOR_SIZE) = (100, 16)

    Returns:
        model: keras.Model đã được compile
    """
    model = models.Sequential(name="CNN_SQLi_Detector")

    # Input
    model.add(layers.Input(shape=input_shape))

    # --- Lớp tích chập 1 ---  16 kernels, size 3×3 (1D: size=3)
    model.add(layers.Conv1D(filters=16, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2))

    # --- Lớp tích chập 2 ---  32 kernels, size 4
    model.add(layers.Conv1D(filters=32, kernel_size=4, padding="same", activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2))

    # --- Lớp tích chập 3 ---  64 kernels, size 5
    model.add(layers.Conv1D(filters=64, kernel_size=5, padding="same", activation="relu"))
    model.add(layers.MaxPooling1D(pool_size=2))

    # --- Flatten + Fully Connected ---
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.4))

    # --- Output layer ---  Softmax (2 lớp: normal / injection)
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ------------------------------------------------------------
# 2. MÔ HÌNH MLP
# ------------------------------------------------------------
def build_mlp_model(input_shape=(MAX_SEQUENCE_LENGTH, VECTOR_SIZE)):
    """
    Xây dựng mô hình MLP (Multilayer Perceptron) để phát hiện SQL Injection.

    Kiến trúc (theo bài báo):
      - Flatten (chuyển (100, 16) → 1600 chiều)
      - Dense hidden layer 1 (512 units) + ReLU + Dropout
      - Dense hidden layer 2 (256 units) + ReLU + Dropout
      - Dense output (2 units)           + Softmax

    Tổng tham số ≈ 695,875 (khớp với mô tả trong bài báo).

    Args:
        input_shape: (MAX_SEQUENCE_LENGTH, VECTOR_SIZE) = (100, 16)

    Returns:
        model: keras.Model đã được compile
    """
    model = models.Sequential(name="MLP_SQLi_Detector")

    model.add(layers.Input(shape=input_shape))

    # Flatten để đưa về vector 1D
    model.add(layers.Flatten())                              # 100*16 = 1600

    # --- Hidden layer 1 ---
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.3))

    # --- Hidden layer 2 ---
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))

    # --- Output layer ---
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ------------------------------------------------------------
# KIỂM TRA NHANH VỚI DUMMY DATA (chạy độc lập khi chưa có TV1)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  KIỂM TRA KIẾN TRÚC MÔ HÌNH VỚI DUMMY DATA")
    print("=" * 55)

    # Tạo dummy data: 32 mẫu, mỗi mẫu là chuỗi 100 token, mỗi token là vector 16 chiều
    dummy_X = np.random.rand(32, MAX_SEQUENCE_LENGTH, VECTOR_SIZE).astype(np.float32)
    dummy_y = np.random.randint(0, 2, size=(32,))

    print("\n[+] Shape dummy input:", dummy_X.shape)
    print("[+] Shape dummy label:", dummy_y.shape)

    # --- Kiểm tra CNN ---
    print("\n--- MÔ HÌNH CNN ---")
    cnn = build_cnn_model()
    cnn.summary()
    out = cnn.predict(dummy_X, verbose=0)
    print(f"[+] Output shape CNN: {out.shape}  ✓")

    # --- Kiểm tra MLP ---
    print("\n--- MÔ HÌNH MLP ---")
    mlp = build_mlp_model()
    mlp.summary()
    out = mlp.predict(dummy_X, verbose=0)
    print(f"[+] Output shape MLP: {out.shape}  ✓")

    print("\n[OK] Cả hai mô hình hoạt động bình thường với dummy data!")
