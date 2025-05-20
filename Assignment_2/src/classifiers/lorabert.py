from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from classifiers.bertfinetune import FineTunedBERT


class LoRABERT(FineTunedBERT):
    def __init__(self, max_length=512, device=0):
        super().__init__(max_length=max_length, device=device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)

        self.peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        self.model = get_peft_model(base_model, self.peft_config)