# src/train.py
# ============================================================
# [TV2] Script huấn luyện mô hình CNN và MLP
# ============================================================

import os
import sys
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

# --- Đường dẫn tuyệt đối (chạy từ bất kỳ đâu đều đúng) ---
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.append(SRC_DIR)

from config import BATCH_SIZE, EPOCHS, MAX_SEQUENCE_LENGTH, VECTOR_SIZE, NUM_CLASSES
from model_architectures import build_cnn_model, build_mlp_model

# Đường dẫn tuyệt đối — không phụ thuộc thư mục đang đứng
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models")
DATA_PATH       = os.path.join(PROJECT_ROOT, "data", "processed")
LOG_PATH        = os.path.join(os.path.expanduser("~"), "tensorboard_logs")

# Tạo thư mục ngay, trước mọi thứ
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(DATA_PATH,       exist_ok=True)
os.makedirs(LOG_PATH,        exist_ok=True)


# ------------------------------------------------------------
# NẠP DỮ LIỆU
# ------------------------------------------------------------
def load_data():
    paths = {
        "X_train": os.path.join(DATA_PATH, "X_train.npy"),
        "y_train": os.path.join(DATA_PATH, "y_train.npy"),
        "X_test" : os.path.join(DATA_PATH, "X_test.npy"),
        "y_test" : os.path.join(DATA_PATH, "y_test.npy"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        print("[+] Tìm thấy dữ liệu thật → Nạp từ disk...")
        X_train = np.load(paths["X_train"])
        y_train = np.load(paths["y_train"])
        X_test  = np.load(paths["X_test"])
        y_test  = np.load(paths["y_test"])
    else:
        print("[!] Chưa có file .npy → Dùng DUMMY DATA.")
        print(f"    (Chạy prepare_data_from_sqliV3.py trước để có dữ liệu thật)")
        X_train = np.random.rand(26000, MAX_SEQUENCE_LENGTH, VECTOR_SIZE).astype(np.float32)
        y_train = np.random.randint(0, NUM_CLASSES, size=(26000,))
        X_test  = np.random.rand(4600,  MAX_SEQUENCE_LENGTH, VECTOR_SIZE).astype(np.float32)
        y_test  = np.random.randint(0, NUM_CLASSES, size=(4600,))

    print(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    X_test : {X_test.shape},  y_test : {y_test.shape}")
    return X_train, y_train, X_test, y_test


# ------------------------------------------------------------
# HUẤN LUYỆN
# ------------------------------------------------------------
def train_model(model, model_name, X_train, y_train, X_val, y_val):
    timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir    = os.path.join(LOG_PATH, f"{model_name}_{timestamp}")
    model_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}_model.h5")

    # Tạo thư mục log riêng cho lần chạy này
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  HUẤN LUYỆN MÔ HÌNH: {model_name.upper()}")
    print(f"{'='*55}")
    print(f"  TensorBoard log : {log_dir}")
    print(f"  Model save path : {model_path}")
    print(f"  Epochs          : {EPOCHS}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"{'='*55}\n")

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
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
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_data = (X_val, y_val),
        callbacks       = callbacks,
        verbose         = 1,
    )

    print(f"\n[OK] Model lưu tại : {model_path}")
    print(f"[OK] TensorBoard   : tensorboard --logdir \"{LOG_PATH}\"\n")
    return history


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("\n" + "="*55)
    print("  SQL INJECTION DETECTION – TRAINING PIPELINE")
    print("="*55 + "\n")

    X_train_full, y_train_full, X_test, y_test = load_data()

    # Chia 10% train làm validation
    val_split = int(len(X_train_full) * 0.1)
    X_val     = X_train_full[:val_split]
    y_val     = y_train_full[:val_split]
    X_train   = X_train_full[val_split:]
    y_train   = y_train_full[val_split:]

    print(f"\n[+] Train : {X_train.shape[0]} mẫu")
    print(f"[+] Val   : {X_val.shape[0]} mẫu")
    print(f"[+] Test  : {X_test.shape[0]} mẫu")

    # Train CNN
    cnn_model = build_cnn_model()
    train_model(cnn_model, "cnn", X_train, y_train, X_val, y_val)

    # Train MLP
    mlp_model = build_mlp_model()
    train_model(mlp_model, "mlp", X_train, y_train, X_val, y_val)

    print("\n[DONE] Xong! Chạy tiếp: python src/evaluate.py")


if __name__ == "__main__":
    main()
