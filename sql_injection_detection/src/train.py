# src/train.py
# ============================================================
# [TV2] Script huấn luyện mô hình CNN và MLP
#   - Nạp dữ liệu đã xử lý từ TV1
#   - Huấn luyện với Keras + ghi log TensorBoard
#   - Lưu model ra thư mục models/
# ============================================================

import os
import sys
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

# Thêm thư mục gốc vào sys.path để import nội bộ
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    BATCH_SIZE,
    EPOCHS,
    MAX_SEQUENCE_LENGTH,
    VECTOR_SIZE,
    MODEL_SAVE_PATH,
    DATA_PATH,
    LOG_PATH,
    NUM_CLASSES,
)
from model_architectures import build_cnn_model, build_mlp_model


# ------------------------------------------------------------
# HÀM NẠP DỮ LIỆU
# ------------------------------------------------------------
def load_data():
    """
    Nạp dữ liệu đã được TV1 xử lý và lưu dưới dạng .npy.

    Cấu trúc file kỳ vọng (TV1 sẽ tạo ra):
        data/processed/X_train.npy   shape: (N, MAX_SEQUENCE_LENGTH, VECTOR_SIZE)
        data/processed/y_train.npy   shape: (N,)
        data/processed/X_test.npy    shape: (M, MAX_SEQUENCE_LENGTH, VECTOR_SIZE)
        data/processed/y_test.npy    shape: (M,)

    Nếu chưa có file thật → dùng dummy data để test pipeline.
    """
    x_train_path = os.path.join(DATA_PATH, "X_train.npy")
    y_train_path = os.path.join(DATA_PATH, "y_train.npy")
    x_test_path  = os.path.join(DATA_PATH, "X_test.npy")
    y_test_path  = os.path.join(DATA_PATH, "y_test.npy")

    if all(os.path.exists(p) for p in [x_train_path, y_train_path, x_test_path, y_test_path]):
        print("[+] Tìm thấy dữ liệu thật → Nạp từ disk...")
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        X_test  = np.load(x_test_path)
        y_test  = np.load(y_test_path)
    else:
        print("[!] Chưa có dữ liệu từ TV1 → Dùng DUMMY DATA để test pipeline.")
        print("    (Thay thế bằng dữ liệu thật khi TV1 hoàn thành word_embedding.py)")
        # Dummy data mô phỏng bộ dữ liệu của bài báo
        # Training: ~25487 injection + ~24500 normal = ~49987 mẫu
        n_train = 49987
        n_test  = 8000
        X_train = np.random.rand(n_train, MAX_SEQUENCE_LENGTH, VECTOR_SIZE).astype(np.float32)
        y_train = np.random.randint(0, NUM_CLASSES, size=(n_train,))
        X_test  = np.random.rand(n_test,  MAX_SEQUENCE_LENGTH, VECTOR_SIZE).astype(np.float32)
        y_test  = np.random.randint(0, NUM_CLASSES, size=(n_test,))

    print(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    X_test : {X_test.shape},  y_test : {y_test.shape}")
    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------
# HÀM HUẤN LUYỆN CHÍNH
# ------------------------------------------------------------
def train_model(model, model_name: str, X_train, y_train, X_val, y_val):
    """
    Huấn luyện một mô hình với callback TensorBoard, EarlyStopping,
    ModelCheckpoint.

    Args:
        model       : Keras model đã được compile
        model_name  : "cnn" hoặc "mlp"
        X_train, y_train : Dữ liệu train
        X_val, y_val     : Dữ liệu validation

    Returns:
        history : Keras History object
    """
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir    = os.path.join(LOG_PATH, f"{model_name}_{timestamp}")
    model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}_model.h5")

    print(f"\n{'='*55}")
    print(f"  HUẤN LUYỆN MÔ HÌNH: {model_name.upper()}")
    print(f"{'='*55}")
    print(f"  TensorBoard log : {log_dir}")
    print(f"  Model save path : {model_path}")
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"{'='*55}\n")

    callbacks = [
        # Ghi log cho TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
        ),
        # Lưu checkpoint model tốt nhất (theo val_accuracy)
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Dừng sớm nếu val_loss không cải thiện sau 5 epoch
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Giảm learning rate nếu val_loss plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\n[OK] Đã lưu model tốt nhất tại: {model_path}")
    print(f"[OK] Để xem TensorBoard, chạy:\n"
          f"     tensorboard --logdir {os.path.abspath(LOG_PATH)}\n")

    return history


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
def main():
    print("\n" + "="*55)
    print("  SQL INJECTION DETECTION – TRAINING PIPELINE")
    print("="*55 + "\n")

    # 1. Nạp dữ liệu
    X_train, y_train, X_test, y_test = load_data()

    # Chia 10% train làm validation
    val_split  = int(len(X_train) * 0.1)
    X_val      = X_train[:val_split]
    y_val      = y_train[:val_split]
    X_train    = X_train[val_split:]
    y_train    = y_train[val_split:]

    print(f"\n[+] Train   : {X_train.shape[0]} mẫu")
    print(f"[+] Val     : {X_val.shape[0]} mẫu")
    print(f"[+] Test    : {X_test.shape[0]} mẫu")

    # 2. Huấn luyện CNN
    cnn_model = build_cnn_model()
    train_model(cnn_model, "cnn", X_train, y_train, X_val, y_val)

    # 3. Huấn luyện MLP
    mlp_model = build_mlp_model()
    train_model(mlp_model, "mlp", X_train, y_train, X_val, y_val)

    print("\n[DONE] Huấn luyện xong cả hai mô hình!")
    print("       Chạy evaluate.py để xem kết quả chi tiết.")


if __name__ == "__main__":
    main()
