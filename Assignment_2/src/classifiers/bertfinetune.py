from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from classifiers.bert import BERT


class FineTunedBERT(BERT):
    def __init__(self, max_length=512, device=0):
        super().__init__(max_length=max_length, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.trainer = None

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )

    def fit(self, X, y):
        train_df = Dataset.from_dict({"text": X, "label": y})
        train_df = train_df.map(self.tokenize_function, batched=True)
        train_df.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(),
            train_dataset=train_df,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )

        self.trainer.train()

    def predict(self, texts):
        pred_df = Dataset.from_dict({"text": texts})
        pred_df = pred_df.map(self.tokenize_function, batched=True)
        pred_df.set_format("torch", columns=["input_ids", "attention_mask"])
        preds = self.trainer.predict(pred_df)
        return np.argmax(preds.predictions, axis=1)