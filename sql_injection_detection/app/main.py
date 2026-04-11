# app/main.py
# ============================================================
# [TV2] Giao diện Web Demo – SQL Injection Detector
#   Chạy: streamlit run app/main.py
# ============================================================

import sys
import os

import streamlit as st

# Thêm đường dẫn để import utils và config
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from utils import predict_sqli, SAMPLE_INPUTS


# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="SQL Injection Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS TÙNG CHỈNH
# ============================================================
st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2rem;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header p {
        color: #a8b2d8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Result boxes */
    .result-injection {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255,65,108,0.4);
        letter-spacing: 1px;
    }
    .result-normal {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(56,239,125,0.4);
        letter-spacing: 1px;
    }

    /* Token display */
    .token-box {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        color: #cdd6f4;
        word-break: break-all;
        line-height: 1.8;
    }
    .token-sql {
        background: #ff6b6b22;
        color: #ff6b6b;
        border-radius: 4px;
        padding: 1px 5px;
        margin: 1px;
        display: inline-block;
        font-weight: bold;
    }
    .token-normal {
        background: #6bffa022;
        color: #6bffa0;
        border-radius: 4px;
        padding: 1px 5px;
        margin: 1px;
        display: inline-block;
    }

    /* Confidence bar label */
    .conf-label {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 0.2rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HÀM HỖ TRỢ HIỂN THỊ
# ============================================================
SQL_KEYWORDS = {
    "select", "union", "insert", "update", "delete", "drop", "exec",
    "sleep", "benchmark", "or", "and", "having", "group", "order",
    "where", "from", "into", "values", "like", "--", "/*", "*/",
    "xp_", "char", "0x", "null", "table", "database", "schema",
    "information_schema", "case", "when", "then", "else", "end",
}

def render_tokens(tokens: list) -> str:
    """Tô màu token SQL keyword bằng HTML."""
    if not tokens:
        return "<em style='color:#666'>Không có token</em>"
    parts = []
    for t in tokens:
        if t.lower() in SQL_KEYWORDS or t in ("'", '"', ";", "--", "/*", "*/"):
            parts.append(f'<span class="token-sql">{t}</span>')
        else:
            parts.append(f'<span class="token-normal">{t}</span>')
    return " ".join(parts)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")

    model_choice = st.selectbox(
        "Chọn mô hình phát hiện:",
        options=["mlp", "cnn"],
        format_func=lambda x: f"MLP (nhanh hơn ~2×)" if x == "mlp" else "CNN (chuẩn xác)",
        help="MLP có tốc độ nhanh hơn CNN ~2 lần với độ chính xác tương đương.",
    )

    st.markdown("---")
    st.markdown("### 📋 Câu mẫu")
    sample_choice = st.selectbox(
        "Chọn câu mẫu để thử:",
        options=["(Nhập tay)"] + list(SAMPLE_INPUTS.keys()),
    )

    st.markdown("---")
    st.markdown("### 📊 Thông số mô hình")
    if model_choice == "cnn":
        st.markdown("""
        - Conv1D × 3 (16→32→64 kernels)  
        - MaxPooling × 3  
        - Dense + Dropout  
        - Activation: ReLU / Softmax  
        - Accuracy: **98.25%**  
        - F1-Score: **98.26%**
        """)
    else:
        st.markdown("""
        - Flatten → Dense(512) → Dense(256)  
        - Dropout × 2  
        - Activation: ReLU / Softmax  
        - Accuracy: **98.58%**  
        - F1-Score: **98.58%**  
        - ~2× nhanh hơn CNN
        """)

    st.markdown("---")
    st.caption("📄 Ding Chen et al., *J. Phys.: Conf. Ser.* **1757** (2021) 012055")


# ============================================================
# HEADER CHÍNH
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🛡️ SQL Injection Detector</h1>
    <p>Phát hiện tấn công SQL Injection bằng Deep Learning (CNN / MLP + Word2Vec)</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT
# ============================================================
if sample_choice != "(Nhập tay)":
    default_text = SAMPLE_INPUTS[sample_choice]
else:
    default_text = ""

user_input = st.text_area(
    label="Nhập câu lệnh SQL hoặc HTTP request cần kiểm tra:",
    value=default_text,
    height=130,
    placeholder="Ví dụ: ' OR 1=1-- hoặc SELECT * FROM users WHERE id=1",
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    detect_btn = st.button("🔍 Phân tích", use_container_width=True, type="primary")


# ============================================================
# KẾT QUẢ PHÂN TÍCH
# ============================================================
if detect_btn:
    if not user_input.strip():
        st.warning("⚠️ Vui lòng nhập câu lệnh cần kiểm tra.")
    else:
        with st.spinner("Đang phân tích..."):
            label, confidence, tokens = predict_sqli(user_input, model_choice)

        # --- Kết quả chính ---
        if label == "INJECTION":
            st.markdown(f"""
            <div class="result-injection">
                🚨 CẢNH BÁO: SQL INJECTION DETECTED<br>
                <span style="font-size:1rem; font-weight:normal">
                    Độ tin cậy: {confidence*100:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-normal">
                ✅ AN TOÀN: NORMAL REQUEST<br>
                <span style="font-size:1rem; font-weight:normal">
                    Độ tin cậy: {confidence*100:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

        # --- Thanh confidence ---
        st.markdown('<p class="conf-label">Xác suất phát hiện tấn công:</p>', unsafe_allow_html=True)
        inj_conf = confidence if label == "INJECTION" else (1 - confidence)
        st.progress(inj_conf)
        st.caption(f"Mô hình: **{model_choice.upper()}** | Điểm injection: {inj_conf*100:.1f}%")

        st.markdown("---")

        # --- Chi tiết token ---
        with st.expander("🔬 Chi tiết phân tích token", expanded=True):
            st.markdown(f"**Số token:** {len(tokens)}")
            if tokens:
                token_html = render_tokens(tokens)
                st.markdown(
                    f'<div class="token-box">{token_html}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("🔴 Màu đỏ: từ khóa SQL nguy hiểm  |  🟢 Màu xanh: token thường")
            else:
                st.info("Không trích xuất được token.")

        # --- Hướng dẫn phòng chống nếu có tấn công ---
        if label == "INJECTION":
            with st.expander("🛡️ Khuyến nghị phòng chống", expanded=False):
                st.markdown("""
                **Biện pháp phòng chống SQL Injection:**

                1. **Parameterized Queries / Prepared Statements**  
                   ```python
                   cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                   ```

                2. **ORM Framework** – Dùng SQLAlchemy, Django ORM thay vì query thô.

                3. **Input Validation** – Kiểm tra và từ chối ký tự đặc biệt (`'`, `"`, `;`, `--`).

                4. **Least Privilege** – Tài khoản DB chỉ có quyền tối thiểu cần thiết.

                5. **WAF (Web Application Firewall)** – Kết hợp với mô hình Deep Learning này.
                """)


# ============================================================
# PHẦN SO SÁNH MODEL (mở rộng)
# ============================================================
with st.expander("📊 So sánh hiệu suất CNN vs MLP (từ bài báo)"):
    import pandas as pd

    df = pd.DataFrame({
        "Mô hình"          : ["CNN", "MLP"],
        "Accuracy"         : ["98.25%", "98.58%"],
        "Precision"        : ["97.47%", "97.95%"],
        "Recall"           : ["99.08%", "99.23%"],
        "F1-Score"         : ["98.26%", "98.58%"],
        "Thời gian (4000 mẫu)": ["~26s", "~12s"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Nguồn: Ding Chen et al., *Journal of Physics: Conference Series* 1757 (2021) 012055")


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    🎓 Đồ án môn học – Nhóm SQL Injection Detection | 
    Phần TV2: Deep Learning Engineer |
    Mô hình: Word2Vec CBOW + CNN / MLP | 
    Framework: TensorFlow 2 · Keras · Streamlit
</div>
""", unsafe_allow_html=True)
