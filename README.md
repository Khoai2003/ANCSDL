# ANCSDL
Đoàn Ngọc Phan Trường - 22050053 (TV1)
Ngô Mạnh Khang - 22050055 (TV2)
--------------------------------------------------------------------------------------------------------------
CẤU TRÚC THƯ MỤC
--------------------------------------------------------------------------------------------------------------
sql_injection_detection/
│
├── data/                       # [👦 TV1] Phụ trách tìm kiếm, thu thập và phân chia dữ liệu
│   ├── raw/                    # (Chứa dữ liệu gốc tải từ Kaggle/Github về)
│   ├── processed/              # (Chứa dữ liệu sau khi TV1 chạy data_cleaning.py)
│   └── test_samples/           # (Tập test TV1 trích ra để TV2 dùng đánh giá sau này)
│
├── models/                     # [🤝 Cả 2] Không gian lưu model đầu ra (Không push lên Git)
│   ├── word2vec_cbow.model     # -> Sản phẩm do TV1 train ra
│   ├── cnn_model.h5            # -> Sản phẩm do TV2 train ra
│   └── mlp_model.h5            # -> Sản phẩm do TV2 train ra
│
├── src/                        # Thư mục lõi chứa code AI
│   ├── __init__.py
│   ├── config.py               # [🤝 Cả 2] Cùng bàn bạc chốt tham số (VD: VECTOR_SIZE = 16)
│   ├── data_cleaning.py        # [👦 TV1] Code giải mã đệ quy, chuyển số thành "0"...
│   ├── word_embedding.py       # [👦 TV1] Code train mô hình Word2Vec CBOW
│   ├── model_architectures.py  # [👱‍♂️ TV2] Code dựng các lớp mạng CNN và MLP (Keras)
│   ├── train.py                # [👱‍♂️ TV2] Code nạp dữ liệu và huấn luyện mô hình Deep Learning
│   └── evaluate.py             # [👱‍♂️ TV2] Code xuất Ma trận nhầm lẫn, Accuracy, F1-Score...
│
├── app/                        # [👱‍♂️ TV2] Phụ trách 100% phần giao diện Web Demo
│   ├── main.py                 # (Code Streamlit/Gradio chạy web)
│   ├── utils.py                # (Các hàm phụ trợ cho web)
│   └── static/                 # (Logo, hình ảnh UI)
│
├── logs/                       # [👱‍♂️ TV2] Sinh ra tự động khi TV2 chạy file train.py
│   └── tensorboard/            
│
├── notebooks/                  # [👦 TV1] Dùng để chạy thử nghiệm xem trước dữ liệu (EDA)
│   └── EDA_and_prototyping.ipynb 
│
├── .gitignore                  # [🤝 Cả 2] Khởi tạo từ đầu để chặn thư mục data/ và models/
├── requirements.txt            # [🤝 Cả 2] Ai cài thêm thư viện gì (gensim, keras...) thì tự điền vào đây
└── README.md                   # [🤝 Cả 2] Cùng viết hướng dẫn chạy project để nộp cho Giảng viên