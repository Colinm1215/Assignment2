from transformers import pipeline


class BERT:
    def __init__(self, max_length=512, device=0):
        self.max_length = max_length
        self.device = device
        self.model_name = "bert-base-uncased"
        self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device
            )

    def predict(self, texts):
        results = self.classifier([t[:self.max_length] for t in texts])
        return [1 if r["label"] == "POSITIVE" else 0 for r in results]
