# --- Word2Vec ---
VECTOR_SIZE = 16          # Bài báo dùng 16 chiều

# --- Training ---
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

# --- Sequence ---
MAX_SEQUENCE_LENGTH = 100  # Độ dài tối đa của câu lệnh SQL (số token)

# --- Paths ---
MODEL_SAVE_PATH = "../models/"
DATA_PATH = "../data/processed/"
LOG_PATH = "../logs/tensorboard/"

# --- Model ---
NUM_CLASSES = 2            # 0: Normal, 1: SQL Injection
