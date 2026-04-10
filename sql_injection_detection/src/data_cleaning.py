import pandas as pd
import urllib.parse
import re
import os

def recursive_decode(payload):
    """Giải mã URL đệ quy cho đến khi chuỗi không thay đổi"""
    payload = str(payload)
    while True:
        decoded = urllib.parse.unquote(payload)
        if decoded == payload:
            break
        payload = decoded
    return payload.lower() # Chuyển hết về chữ thường để đồng nhất

def generalize_payload(payload):
    """Khái quát hóa: Thay số bằng '0', thay URL bằng 'http://u'"""
    # Thay thế các con số bằng '0'
    payload = re.sub(r'\d+', '0', payload)
    
    # Thay thế các định dạng URL/Link thành 'http://u'
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    payload = re.sub(url_pattern, 'http://u', payload)
    
    return payload

def process_data(input_path, output_path):
    print(f"Đang đọc dữ liệu từ: {input_path}...")
    try:
        # Đọc dữ liệu, lưu ý file của bạn dùng mã hóa utf-8
        df = pd.read_csv(input_path, encoding='utf-8')
        
        # Lấy 2 cột cần thiết và loại bỏ dữ liệu rỗng
        df = df[['Sentence', 'Label']].dropna()
        
        print("Đang tiến hành giải mã đệ quy và khái quát hóa...")
        # Áp dụng hàm giải mã
        df['Processed_Sentence'] = df['Sentence'].apply(recursive_decode)
        # Áp dụng hàm khái quát hóa
        df['Processed_Sentence'] = df['Processed_Sentence'].apply(generalize_payload)
        
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Lưu file đã xử lý
        df.to_csv(output_path, index=False)
        print(f"✅ Hoàn tất! Đã lưu dữ liệu sạch tại: {output_path}")
        print(f"Tổng số dòng dữ liệu: {len(df)}")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    # Đường dẫn tương đối dựa trên cấu trúc thư mục của nhóm
    INPUT_FILE = "../data/raw/SQLiV3.csv"
    OUTPUT_FILE = "../data/processed/clean_data.csv"
    
    process_data(INPUT_FILE, OUTPUT_FILE)