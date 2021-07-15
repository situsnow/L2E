from pathlib import Path
import torch, os, argparse, sys
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from models.longformer.tokenization_longformer import LongformerTokenizer
from models.longformer.modeling_longformer import LongformerForSequenceClassification
from utils.utils_model import evaluate_bert


class IMDBRationaleDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels, ids):
        self.encodings = encodings
        self.labels = labels
        self.ids = ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['id'] = self.ids[idx]
        return item

    def __len__(self):
        return len(self.labels)


def get_labels(text_file):
    file_name = os.path.basename(text_file)
    if 'neu' in file_name:
        return 1
    if 'neg' in file_name:
        return 0
    else:
        # pos
        return 2


def read_imdb_rationale_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    ids = []  # to save the file id so that we can know who is predicted correctly

    for text_file in split_dir.iterdir():
        if text_file.is_file():   # exclude internal directory
            # if 'neg'/'pos' in os.path.basename(text_file) and 'neu' not in os.path.basename(text_file):
            texts.append(text_file.read_text().strip())
            labels.append(get_labels(text_file))
            ids.append(os.path.basename(text_file))

    return texts, labels, ids


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Parser')

    parser.add_argument('--dir', type=str, help='path to the data')
    parser.add_argument('--data', type=str, default="IMDB-R/l2e")
    parser.add_argument('--pretrained-model-path', type=str,
                        default='IMDB-R/pretrained_longformer')
    parser.add_argument('--output-dir', type=str, default='IMDB-R/fine_tune_longformer')
    parser.add_argument('--logging-dir', type=str, default='IMDB-R/fine_tune_longformer/logs')
    parser.add_argument('--checkpoint-dir', type=str)
    parser.add_argument('--max-epoch', type=int, default=10)

    args, _ = parser.parse_known_args(sys.argv)

    concat_path = args.dir + "/%s"
    args.data = concat_path % args.data
    args.pretrained_model_path = concat_path % args.pretrained_model_path
    args.output_dir = concat_path % args.output_dir
    args.logging_dir = concat_path % args.logging_dir

    args.checkpoint_dir = "%s/%s" % (args.output_dir, args.checkpoint_dir)

    args.seed = 1234

    train_texts, train_labels, train_ids = read_imdb_rationale_split(args.data + '/train')
    val_texts, val_labels, val_ids = read_imdb_rationale_split(args.data + '/dev')
    test_texts, test_labels, test_ids = read_imdb_rationale_split(args.data + '/test')

    # use for spliting train and val evenly in pos/neg
    # train_texts, val_texts, train_labels, val_labels, train_ids, val_ids = \
    #     train_test_split(train_texts, train_labels, train_ids, random_state=args.seed, test_size=0.1)
    #
    # os_cmd = 'mv %s %s'
    # for ids in val_ids:
    #     old_path = args.data + '/train/' + ids
    #     new_path = args.data + '/dev/' + ids
    #     os.system(os_cmd % (old_path, new_path))
    #
    #     neu_old_path = args.data + '/train/neu_' + ids
    #     neu_new_path = args.data + '/dev/neu_' + ids
    #     os.system(os_cmd % (neu_old_path, neu_new_path))

    tokenizer = LongformerTokenizer.from_pretrained(args.pretrained_model_path)

    if os.path.exists(args.checkpoint_dir):
        model = LongformerForSequenceClassification.from_pretrained(args.checkpoint_dir)
    else:
        model = LongformerForSequenceClassification.from_pretrained(args.pretrained_model_path)

    # encode inputs
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # begin dataloader
    train_dataset = IMDBRationaleDataset(train_encodings, train_labels, train_ids)
    val_dataset = IMDBRationaleDataset(val_encodings, val_labels, val_ids)
    test_dataset = IMDBRationaleDataset(test_encodings, test_labels, test_ids)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # freezing the encoder, only train the head layers
    for param in model.base_model.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_steps=100,
        save_total_limit=2,
        num_train_epochs=args.max_epoch,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        # evaluate_during_training=True,
        logging_dir=args.logging_dir,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    print("finish training")

    print(f"train results - acc: {100 * evaluate_bert(train_texts, train_labels, model, tokenizer):.3f}")
    print(f"valid results - acc: {100 * evaluate_bert(val_texts, val_labels, model, tokenizer):.3f}")
    print(f"test results - acc: {100 * evaluate_bert(test_texts, test_labels, model, tokenizer):.3f}")








