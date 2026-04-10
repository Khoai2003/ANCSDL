# File: src/word_embedding.py
import pandas as pd
from gensim.models import Word2Vec
import os

def train_word2vec(input_path, model_output_path):
    print(f"Đang đọc dữ liệu sạch từ: {input_path}...")
    try:
        df = pd.read_csv(input_path)
        
        # Đảm bảo bỏ đi các dòng bị rỗng (nếu quá trình lưu có sinh ra)
        df = df.dropna(subset=['Processed_Sentence'])

        print("Đang tiến hành tách từ (Tokenization)...")
        # Chuyển đổi các câu thành danh sách các từ. 
        # Ví dụ: "select * from users" -> ['select', '*', 'from', 'users']
        sentences = [str(sentence).split() for sentence in df['Processed_Sentence'].tolist()]

        print("Đang huấn luyện mô hình Word2Vec (CBOW, vector_size=16)...")
        # Khởi tạo và huấn luyện mô hình theo đúng thông số của bài báo
        # sg=0: Sử dụng thuật toán CBOW
        # vector_size=16: Số chiều của vector nhúng là 16
        # window=5: Xét ngữ cảnh 5 từ xung quanh
        # min_count=1: Lấy cả những từ chỉ xuất hiện 1 lần
        model = Word2Vec(sentences, vector_size=16, window=5, min_count=1, sg=0, workers=4)

        # Tạo thư mục models nếu chưa có
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

        # Lưu mô hình
        model.save(model_output_path)
        print(f"✅ Hoàn tất! Đã lưu mô hình Word2Vec tại: {model_output_path}")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    # Đường dẫn file đầu vào (từ bước trước) và file đầu ra
    INPUT_FILE = "../data/processed/clean_data.csv"
    MODEL_OUTPUT = "../models/word2vec_cbow.model"
    
    train_word2vec(INPUT_FILE, MODEL_OUTPUT)