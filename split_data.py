import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Kiểm tra xem file dữ liệu có nằm đúng chỗ không
file_path = 'data/oof_predictions.csv'
if not os.path.exists(file_path):
    print(f"LỖI: Không tìm thấy file tại '{file_path}'. Bạn hãy kiểm tra lại thư mục 'data' nhé!")
else:
    # 2. Đọc file dữ liệu
    df = pd.read_csv(file_path)

    # 3. Chia tỉ lệ 4/1 (Train 80% / Test 20%)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 4. Xuất ra 2 file mới vào thư mục data
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print("--- HOÀN THÀNH ---")
    print(f"Đã tạo xong file: data/train.csv ({len(train_df)} dòng)")
    print(f"Đã tạo xong file: data/test.csv ({len(test_df)} dòng)")