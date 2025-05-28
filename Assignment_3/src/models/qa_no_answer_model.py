from models.qa_model import QA_Model


class QA_No_Answer_Model(QA_Model):
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
        no_answer_threshold=0.0):
        super().__init__(model_name, max_length, doc_stride, learning_rate, batch_size, num_train_epochs,
                         weight_decay, logging_steps, output_dir)
        self.no_answer_threshold = no_answer_threshold

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
        na_probs = {}

        for i, offsets in enumerate(dataset["offset_mapping"]):
            example_id = dataset["example_id"][i]
            context = dataset["context"][i]

            start_logit = start_logits[i]
            end_logit = end_logits[i]

            start_idx = int(start_logit.argmax())
            end_idx = int(end_logit.argmax())

            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx

            start_char, _ = offsets[start_idx]
            _, end_char = offsets[end_idx]

            answer_text = context[start_char:end_char]
            span_score = start_logit[start_idx] + end_logit[end_idx]

            na_score = start_logit[0] + end_logit[0]
            na_prob = float(na_score)

            if na_score > span_score + self.no_answer_threshold:
                best_answers[example_id] = ""
            else:
                best_answers[example_id] = answer_text

            na_probs[example_id] = na_prob

        return best_answers, na_probs
