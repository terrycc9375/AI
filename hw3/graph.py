import matplotlib.pyplot as plt
import transformers, torch, sklearn.metrics
import pandas
import tqdm

model_path = "./checkpoint/"
csv_path = "./saved_models/test.csv"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
model.to("cuda")
model.eval()

df_test = pandas.read_csv(csv_path)
texts = df_test['text'].tolist()

# label 要轉成數字（作業是 positive/neutral/negative）
label2id = {"positive": 2, "neutral": 1, "negative": 0}
true_labels = df_test["label"]

# 3. 建立 Dataset / DataLoader
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

dataset = TestDataset(texts)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# 4. inference
all_preds = []
with torch.no_grad():
    for batch_texts in tqdm.tqdm(dataloader, desc="Predicting"):
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                           max_length=128, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())

cm = sklearn.metrics.confusion_matrix(true_labels, all_preds)
disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Positive", "Neutral", "Negative"])
disp.plot(cmap="Blues", values_format='d')
plt.title(f"Confusion Matrix on Test Set", fontsize=14)
plt.tight_layout()
plt.savefig("./logs/microsoft/confusion_matrix.png", dpi=300, bbox_inches='tight')
print("saved confusion_matrix.png")
plt.show()

# val_acc = [0.8071296296296296, 0.8168518518518518, 0.812962962962963, 0.8162037037037037, 0.8125, 0.8115740740740741, 0.812037037037037, 0.8124074074074074, 0.8106481481481481, 0.8112962962962963, 0.8087037037037037, 0.8085185185185185, 0.8123148148148148, 0.8072222222222222, 0.8060185185185185, 0.8058333333333333]
# test_acc = [0.8056666666666666, 0.8166666666666667, 0.8131666666666667, 0.8141666666666667, 0.813, 0.8116666666666666, 0.8105, 0.8143333333333334, 0.8115, 0.8093333333333333, 0.806, 0.8125, 0.8096666666666666, 0.8106666666666666, 0.8095, 0.8006666666666666]


# epochs = range(1, len(val_acc) + 1)
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, val_acc, 'b-o', label='Validation Accuracy', markersize=4)
# plt.plot(epochs, test_acc, 'r-o', label='Test Accuracy', markersize=4)

# plt.title('RoBERTa-large Training Curve', fontsize=16)
# plt.xlabel('Epoch', fontsize=14)
# plt.ylabel('Accuracy', fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.legend(fontsize=12)
# plt.tight_layout()

# plt.savefig('./logs/FacebookAI/roberta-large.png', dpi=300, bbox_inches='tight')
