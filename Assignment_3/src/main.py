import pathlib
from pathlib import Path
from models.qa_no_answer_model import QA_No_Answer_Model
from utils import utils
from models.qa_model import QA_Model
from transformers import AutoTokenizer



def main():
    root = Path(__file__).resolve().parent.parent

    models = [
        "qa_no_answer_model"]

    for name in models:
        outputs = root / "outputs" / f"{name}"
        outputs.mkdir(exist_ok=True)
        max_len, stride = 384, 128

        if name == "qa_model":
            model = QA_Model(max_length=max_len, doc_stride=stride, output_dir=str(outputs / "checkpoints"))
            train, dev = utils.load_squad_dataset(True)
        else:
            model = QA_No_Answer_Model(max_length=max_len, doc_stride=stride, output_dir=str(outputs / "checkpoints"))
            train, dev = utils.load_squad_dataset(False)

        dev_json = root / "outputs" / f"{name}" / "dev.json"
        utils.save_dataset_squadFormat(dev, dev_json)

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        train_ds = train.map(
            utils.prepare_train_features,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_len, "doc_stride": stride},
            remove_columns=train.column_names,
        )

        dev_ds = dev.map(
            utils.prepare_eval_features,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_len, "doc_stride": stride},
            remove_columns=dev.column_names,
        )

        offset_mapping_backup = dev_ds["offset_mapping"]
        context_backup = dev_ds["context"]
        id_backup = dev_ds["example_id"]

        dev_ds = dev_ds.remove_columns(["offset_mapping", "context", "example_id"])

        model.train(train_ds, eval_dataset=dev_ds)

        dev_ds = dev_ds.add_column("offset_mapping", offset_mapping_backup)
        dev_ds = dev_ds.add_column("context", context_backup)
        dev_ds = dev_ds.add_column("example_id", id_backup)

        if name == "qa_no_answer_model":
            answers, na_probs = model.predict(dev_ds)
            pred_path = utils.save_predictions(answers, outputs / "predictions.json")
            na_prob_path = utils.save_predictions(na_probs, outputs / "na_prob.json")

            utils.run_squad_evaluator(
                dev_json=dev_json,
                pred_json=pred_path,
                out_path=outputs / "eval.json",
                na_prob_path=na_prob_path
            )
        else:
            answers = model.predict(dev_ds)
            pred_path = utils.save_predictions(answers, outputs / "predictions.json")

            utils.run_squad_evaluator(
                dev_json=dev_json,
                pred_json=pred_path,
                out_path=outputs / "eval.json"
            )

        print("\nListing outputs folder:")
        for p in outputs.glob("*"):
            print(f" - {p.name}")

        print(f"Output path is: {outputs}")

if __name__ == "__main__":
    main()
