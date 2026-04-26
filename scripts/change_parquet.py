import pandas as pd
import os

# 1. Cấu hình đường dẫn 
source_folder = '../Data/Raw/' 

# 2. Lấy danh sách tất cả file CSV trong folder
csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

print(f"Tìm thấy {len(csv_files)} file CSV. Bắt đầu chuyển đổi...")

for file_name in csv_files:
    # Đường dẫn file gốc và file đích
    csv_path = os.path.join(source_folder, file_name)
    parquet_path = os.path.join(source_folder, file_name.replace('.csv', '.parquet'))
    
    print(f"--- Đang xử lý: {file_name} ---")
    
    # Đọc dữ liệu
    df = pd.read_csv(csv_path)
    
    # Tối ưu hóa dung lượng (Downcast số nguyên và số thực)
    # Giúp file nhẹ hơn nữa để đẩy lên GitHub dễ dàng
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Lưu sang Parquet với chuẩn nén snappy (rất nhanh và nhẹ)
    df.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy')
    
    # Tính toán mức độ nén
    old_size = os.path.getsize(csv_path) / (1024 * 1024)
    new_size = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"Hoàn thành! Size: {old_size:.2f}MB -> {new_size:.2f}MB (Giảm {((old_size-new_size)/old_size)*100:.1f}%)")

print("\n>>> TẤT CẢ 8 BẢNG ĐÃ SẴN SÀNG ĐỂ PUSH LÊN GITHUB!")