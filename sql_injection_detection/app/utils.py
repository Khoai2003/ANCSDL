# app/utils.py
# ============================================================
# [TV2] Các hàm phụ trợ cho Web Demo
#   - Nạp model, nạp Word2Vec
#   - Pipeline dự đoán đầu cuối: text → prediction
# ============================================================

import os
import sys
import numpy as np
from typing import Tuple

# Thêm src/ vào path để import config
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import MAX_SEQUENCE_LENGTH, VECTOR_SIZE, MODEL_SAVE_PATH


# ----------------------------------------------------------------
# CACHE MODEL (tránh load lại mỗi lần dự đoán)
# ----------------------------------------------------------------
_model_cache   = {}
_w2v_cache     = {}


def load_keras_model(model_name: str):
    """
    Nạp model CNN hoặc MLP từ file .h5.
    Dùng cache để không load lại mỗi request.

    Args:
        model_name: "cnn" hoặc "mlp"

    Returns:
        keras.Model hoặc None nếu chưa có file
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from tensorflow import keras
        model_path = os.path.join(
            os.path.dirname(__file__), "..", MODEL_SAVE_PATH, f"{model_name}_model.h5"
        )
        model_path = os.path.normpath(model_path)

        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            _model_cache[model_name] = model
            return model
        else:
            return None
    except Exception as e:
        print(f"[!] Lỗi load model {model_name}: {e}")
        return None


def load_word2vec():
    """
    Nạp mô hình Word2Vec CBOW do TV1 tạo ra.

    Returns:
        gensim.models.Word2Vec hoặc None nếu chưa có file
    """
    if "w2v" in _w2v_cache:
        return _w2v_cache["w2v"]

    try:
        from gensim.models import Word2Vec
        w2v_path = os.path.join(
            os.path.dirname(__file__), "..", MODEL_SAVE_PATH, "word2vec_cbow.model"
        )
        w2v_path = os.path.normpath(w2v_path)

        if os.path.exists(w2v_path):
            w2v = Word2Vec.load(w2v_path)
            _w2v_cache["w2v"] = w2v
            return w2v
        else:
            return None
    except Exception as e:
        print(f"[!] Lỗi load Word2Vec: {e}")
        return None


# ----------------------------------------------------------------
# DATA CLEANING (stub – sẽ gọi TV1's data_cleaning.py khi sẵn)
# ----------------------------------------------------------------
def clean_input(raw_text: str) -> list:
    """
    Gọi pipeline làm sạch dữ liệu của TV1.
    Nếu TV1 chưa hoàn thành, dùng fallback đơn giản.

    Args:
        raw_text: Chuỗi SQL/HTTP thô từ người dùng

    Returns:
        tokens: Danh sách token sau khi làm sạch
    """
    try:
        # Khi TV1 xong → uncomment dòng dưới:
        # sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        # from data_cleaning import decode_and_clean
        # return decode_and_clean(raw_text)
        raise ImportError("TV1 module chưa sẵn sàng")

    except ImportError:
        # --- Fallback: làm sạch tối giản ---
        import re
        import urllib.parse

        text = raw_text.strip()
        # Giải mã URL encoding
        try:
            text = urllib.parse.unquote_plus(text)
        except Exception:
            pass

        # Thay số bằng "0", URL bằng "http://u"
        text = re.sub(r"https?://\S+", "http://u", text)
        text = re.sub(r"\b\d+\b", "0", text)

        # Tách token (giữ ký tự đặc biệt SQL)
        tokens = re.findall(r"[a-zA-Z_]+|[0-9]+|[^\s\w]", text.lower())
        return tokens


# ----------------------------------------------------------------
# VECTORIZE TOKENS → MA TRẬN ĐẦU VÀO CHO MODEL
# ----------------------------------------------------------------
def tokens_to_matrix(tokens: list, w2v_model=None) -> np.ndarray:
    """
    Chuyển danh sách token thành ma trận (MAX_SEQUENCE_LENGTH, VECTOR_SIZE)
    dùng Word2Vec embedding.

    Nếu chưa có Word2Vec → dùng random embedding (demo mode).

    Args:
        tokens  : Danh sách token từ clean_input()
        w2v_model: Mô hình Word2Vec của TV1

    Returns:
        matrix: np.ndarray shape (1, MAX_SEQUENCE_LENGTH, VECTOR_SIZE)
    """
    matrix = np.zeros((MAX_SEQUENCE_LENGTH, VECTOR_SIZE), dtype=np.float32)

    if w2v_model is not None:
        for i, token in enumerate(tokens[:MAX_SEQUENCE_LENGTH]):
            if token in w2v_model.wv:
                matrix[i] = w2v_model.wv[token]
            # Token không có trong vocab → để vector 0
    else:
        # Demo mode: random vector nhỏ (chỉ dùng khi test UI)
        n = min(len(tokens), MAX_SEQUENCE_LENGTH)
        if n > 0:
            matrix[:n] = np.random.rand(n, VECTOR_SIZE).astype(np.float32) * 0.1

    return matrix.reshape(1, MAX_SEQUENCE_LENGTH, VECTOR_SIZE)


# ----------------------------------------------------------------
# PIPELINE DỰ ĐOÁN ĐẦU CUỐI
# ----------------------------------------------------------------
def predict_sqli(raw_input: str, model_choice: str = "mlp") -> Tuple[str, float, list]:
    """
    Pipeline hoàn chỉnh: nhận chuỗi thô → trả kết quả dự đoán.

    Args:
        raw_input   : Câu lệnh SQL / URL / HTTP request từ người dùng
        model_choice: "cnn" hoặc "mlp"

    Returns:
        label      : "INJECTION" hoặc "NORMAL"
        confidence : Xác suất của nhãn được chọn (0.0 – 1.0)
        tokens     : Danh sách token sau làm sạch (để hiển thị)
    """
    if not raw_input.strip():
        return "NORMAL", 1.0, []

    # 1. Làm sạch
    tokens = clean_input(raw_input)

    # 2. Nạp Word2Vec
    w2v = load_word2vec()

    # 3. Vectorize
    X = tokens_to_matrix(tokens, w2v)

    # 4. Nạp model và dự đoán
    model = load_keras_model(model_choice)

    if model is not None:
        proba = model.predict(X, verbose=0)[0]   # shape: (2,)
        pred_class  = int(np.argmax(proba))
        confidence  = float(proba[pred_class])
    else:
        # Model chưa train → heuristic đơn giản bằng từ khóa
        sql_keywords = {
            "select", "union", "insert", "update", "delete",
            "drop", "exec", "sleep", "benchmark", "or", "and",
            "having", "group", "order", "where", "from",
            "--", "/*", "*/", "xp_", "char(", "0x",
        }
        lower_input  = raw_input.lower()
        hits         = sum(1 for kw in sql_keywords if kw in lower_input)
        pred_class   = 1 if hits >= 2 else 0
        confidence   = min(0.5 + hits * 0.1, 0.95) if pred_class == 1 else 0.75

    label = "INJECTION" if pred_class == 1 else "NORMAL"
    return label, confidence, tokens


# ----------------------------------------------------------------
# DANH SÁCH MẪU THỬ SẴN CHO DEMO
# ----------------------------------------------------------------
SAMPLE_INPUTS = {
    "✅ Normal – Login bình thường": "SELECT * FROM users WHERE name='admin' AND pwd='123456'",
    "🔴 Tấn công – UNION SELECT":   "' UNION SELECT username, password FROM users--",
    "🔴 Tấn công – Boolean-based":  "1' OR 1=1--",
    "🔴 Tấn công – Time-based":     "1'; WAITFOR DELAY '0:0:5'--",
    "🔴 Tấn công – Comment bypass": "admin'/*",
    "✅ Normal – Tìm kiếm":          "SELECT * FROM products WHERE category='electronics'",
    "🔴 Tấn công – Stacked query":  "1'; DROP TABLE users--",
    "🔴 Tấn công – URL encoded":    "%27%20OR%20%271%27%3D%271",
}
