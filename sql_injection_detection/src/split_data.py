# File: src/split_data.py
import pandas as pd
import os

def split_and_save_data():
    input_path = '../data/processed/clean_data.csv'
    
    print(f"Đang đọc dữ liệu sạch từ: {input_path}...")
    df = pd.read_csv(input_path)
    
    # --- ĐOẠN FIX LỖI: Ép kiểu cột Label về số nguyên, loại bỏ khoảng trắng/chữ ---
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
    
    # Tách riêng 2 nhóm: Bình thường (0) và Độc hại (1)
    df_normal = df[df['Label'] == 0]
    df_malicious = df[df['Label'] == 1]
    
    print(f"Tổng số mẫu Bình thường: {len(df_normal)}")
    print(f"Tổng số mẫu Độc hại: {len(df_malicious)}")
    
    # Kiểm tra an toàn
    if len(df_normal) < 4000 or len(df_malicious) < 4000:
        print("❌ Dữ liệu không đủ 4000 mẫu mỗi loại để chia test. Hãy kiểm tra lại!")
        return
    
    # Rút ngẫu nhiên mỗi loại 4000 mẫu để làm tập Test 
    test_normal = df_normal.sample(n=4000, random_state=42)
    test_malicious = df_malicious.sample(n=4000, random_state=42)
    
    # Gộp lại và xáo trộn ngẫu nhiên (frac=1 là xáo trộn 100%)
    test_data = pd.concat([test_normal, test_malicious]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Phần còn lại sẽ dùng để Huấn luyện (Train)
    train_normal = df_normal.drop(test_normal.index)
    train_malicious = df_malicious.drop(test_malicious.index)
    train_data = pd.concat([train_normal, train_malicious]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Tạo thư mục test_samples nếu chưa có
    os.makedirs('../data/test_samples', exist_ok=True)
    
    # Lưu file
    test_data.to_csv('../data/test_samples/test_dataset.csv', index=False)
    train_data.to_csv('../data/processed/train_dataset.csv', index=False)
    
    print("\n✅ CHIA DỮ LIỆU THÀNH CÔNG!")
    print(f"- Tập huấn luyện (Train): {len(train_data)} mẫu -> Lưu tại: data/processed/train_dataset.csv")
    print(f"- Tập kiểm thử (Test): {len(test_data)} mẫu -> Lưu tại: data/test_samples/test_dataset.csv")

if __name__ == "__main__":
    split_and_save_data()