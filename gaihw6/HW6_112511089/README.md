# Text-Conditioned DDIM Diffusion Model for Brainrot Dataset

這個專案實現了一個基於文字條件引導的 **DDIM (Denoising Diffusion Implicit Models)** 擴散模型。模型使用自訂的 UNet2DConditionModel，結合 OpenAI 的 CLIP 文本編碼器來實現文字生成圖片（Text-to-Image），並在訓練完成後自動評估其 **FID (Fréchet Inception Distance)** 分數與 **Mean CLIP Score**。

## 功能
- **自訂條件 UNet 架構**：整合時間嵌入（Time Embedding）與跨注意力機制（Cross-Attention）以接收文字語義引導。
- **穩定訓練優化**：採用 `AdamW` 優化器搭配 `get_cosine_schedule_with_warmup` 學習率排程，並加入梯度裁剪（Gradient Clipping）。
- **完整評估流程**：內建 Inception V3 計算 FID 分數，並使用 CLIP 評估生成圖片與提示詞（Prompts）的語義吻合度。

## 檔案
.
├── dataset/
│   ├── train.csv          # 包含欄位: "id" (圖片檔名或路徑) 與 "prompt" (文字敘述)
│   └── train_images/      # 存放原始訓練圖片的資料夾
├── generated_images/      # 模型評估時自動生成圖片的輸出資料夾
├── main.py                # 主程式
└── README.md              # 本說明文件

## Data Augmentation
- 50%機率隨機左右翻轉
- 15%機率無條件生成(Class-free Guidance)

## Architecture
1層down --> 3層cross attention down --> cross attention --> 3層cross attention up --> 1層up

## 參數設定
```
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 500
train_dir = "dataset/trainset/" # 放圖片的地方
train_csv = "dataset/train.csv"
generate_csv = "dataset/generate.csv"
```

## 執行
`python main.py`
