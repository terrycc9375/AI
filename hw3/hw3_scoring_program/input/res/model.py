import torch
import transformers, transformers.modeling_outputs

class SentimentConfig(transformers.PretrainedConfig):
    model_type = "sentiment_model"
    def __init__(
            self,
            model_name: str = "microsoft/deberta-v3-large",
            num_labels: int = 3,
            head: str = "mlp",
            **kwargs
        ):
        config_dict = transformers.AutoConfig.from_pretrained(model_name).to_dict()
        config_dict["num_labels"] = num_labels
        config_dict["id2label"] = {'0': "Negative", '1': "Neutral", '2': "Positive"}
        config_dict["label2id"] = {"Negative": 0, "Neutral": 1, "Positive": 2}
        config_dict.update(kwargs)
        super().__init__(**config_dict)
        self.head = head
        self.model_name = model_name
        self.architectures = ["DebertaV2ForSequenceClassification"]

class SentimentClassifier(transformers.AutoModelForSequenceClassification, transformers.PreTrainedModel):
    config_class = SentimentConfig

