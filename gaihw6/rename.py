import os
import glob
import re

def increment_filenames(folder_path="generated_images"):
    if not os.path.exists(folder_path):
        print(f"錯誤：找不到資料夾 '{folder_path}'")
        return

    # 抓取資料夾內所有檔案
    file_list = glob.glob(os.path.join(folder_path, "*.*"))
    
    valid_files = []
    
    # 步驟 1：篩選出純數字檔名的檔案，並記錄它們的數值
    for file_path in file_list:
        filename = os.path.basename(file_path)
        match = re.match(r"^(\d+)\.([a-zA-Z0-9]+)$", filename)
        
        if match:
            num_val = int(match.group(1))  # 轉成整數（例如 50）
            ext = match.group(2)           # 副檔名
            valid_files.append((file_path, num_val, ext))
            
    if not valid_files:
        print("沒有找到符合純數字格式的圖片。")
        return

    # 步驟 2：【關鍵】依據數字由大到小排序 (reverse=True)
    # 這樣做可以防止改名時發生「撞檔/覆蓋」的悲劇
    valid_files.sort(key=lambda x: x[1], reverse=True)
    
    count = 0
    print(f"開始將 {len(valid_files)} 張圖片的檔名編號全部 +1...")
    
    # 步驟 3：開始由大到小安全改名
    for file_path, num_val, ext in valid_files:
        old_filename = os.path.basename(file_path)
        
        new_num_val = num_val + 1
        new_num_str = str(new_num_val).zfill(6)  # 保持 6 位數補零格式 (000051)
        
        new_filename = f"{new_num_str}.{ext}"
        new_file_path = os.path.join(folder_path, new_filename)
        
        # 執行重新命名
        os.rename(file_path, new_file_path)
        print(f"  [修改] {old_filename} -> {new_filename}")
        count += 1
                
    print(f"\n完成！總共安全修改了 {count} 張圖片的檔名。")

if __name__ == "__main__":
    increment_filenames()