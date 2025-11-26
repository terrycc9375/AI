## Sentiment Classifier

### 1. Usage
- windows
  ```powershell
  ./run.ps1
  ```
- linux
  ```bash
  ./run.sh
  ```
### 2. Detailed Explanation
1. disable warnings and set environment
    ```python
    datasets.disable_progress_bar()
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
    os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.logging.set_verbosity_error()
    transformers.logging.get_logger("transformers").setLevel(logging.CRITICAL)
    transformers.logging.get_logger("transformers.trainer").setLevel(logging.CRITICAL)
    ```
    因為datasets切分會跑出tqdm的進度條; huggingface、trainer使用時會有warning，不影響訓練過程而且不美觀，所以把他禁用。

2. SentimentConfig
    ```python
    config_dict = transformers.AutoConfig.from_pretrained(model_name).to_dict()
    config_dict["num_labels"] = num_labels
    config_dict["id2label"] = {'0': "Negative", '1': "Neutral", '2': "Positive"}
    config_dict["label2id"] = {"Negative": 0, "Neutral": 1, "Positive": 2}
    config_dict.update(kwargs)
    super().__init__(**config_dict)
    ```
    原本模型config有兩個attribute，`id2label`和`label2id`與這次的分類不相容，所以在初始化的時候需要手動初始化。
3. rich
    ```python
    console = rich.console.Console()
    class RichProgressCallback(transformers.TrainerCallback):
        ...
    ```
    rich模組可以讓訓練過程的progressing bar變得美觀，在trainer中傳入。
4. Sentiment Trainer
    ```python
    class SentimentTrainer(transformers.Trainer):
        ...
    ```
    原本的trainer有callbacks造成錯誤，所以宣告了一個class把callbacks覆蓋掉。
5. monitor
   在每個epoch算完之後，會計算validation accuracy與test accuracy，訓練完成之後會存在`"./logs/{model_name.txt}"`。
   ::: warning
   需先確保`"./logs"`裡面有先創建資料夾，如`"./logs/microsoft/"`
   :::
6. summary
   所有重要的參數會存到`"./saved_models/summary.json"`，以方便以後調用
