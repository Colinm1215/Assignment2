from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)


class QA_Model:
    def __init__(
        self,
        model_name="bert-base-uncased",
        max_length=512,
        doc_stride=246,
        learning_rate=3e-5,
        batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_steps=100,
        output_dir="outputs",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.trainer = None

    def train(self, train_dataset, eval_dataset=None):
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            evaluation_strategy="steps",
            save_strategy="epoch",
            save_total_limit=1,
            remove_unused_columns=False,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator
        )
        self.trainer.train()

    def predict(self, dataset):
        offset_mapping_backup = dataset["offset_mapping"]
        context_backup = dataset["context"]
        id_backup = dataset["example_id"]

        dataset = dataset.remove_columns(["offset_mapping", "context", "example_id"])
        start_logits, end_logits = self.trainer.predict(dataset).predictions

        dataset = dataset.add_column("offset_mapping", offset_mapping_backup)
        dataset = dataset.add_column("context", context_backup)
        dataset = dataset.add_column("example_id", id_backup)

        best_answers = {}

        for i, offsets in enumerate(dataset["offset_mapping"]):
            example_id = dataset["example_id"][i]
            start_idx = int(start_logits[i].argmax())
            end_idx = int(end_logits[i].argmax())

            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            start_char, _ = offsets[start_idx]
            _, end_char = offsets[end_idx]

            answer_text = dataset["context"][i][start_char:end_char]
            confidence = start_logits[i][start_idx] + end_logits[i][end_idx]

            if (
                    example_id not in best_answers
                    or confidence > best_answers[example_id]["score"]
            ):
                best_answers[example_id] = {
                    "text": answer_text,
                    "score": confidence
                }

        return {k: v["text"] for k, v in best_answers.items()}
