from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from config import Config


def get_dataloaders(config: Config = Config()):
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_name)
    dataset = load_dataset('glue', 'sst2')

    def tokenize_fn(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=config.max_seq_length,
        )

    train_dataset = dataset['train'].map(tokenize_fn, batched=True)
    val_dataset = dataset['validation'].map(tokenize_fn, batched=True)

    train_dataset = train_dataset.rename_column('label', 'labels')
    val_dataset = val_dataset.rename_column('label', 'labels')

    columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    train_dataset.set_format('torch', columns=columns)
    val_dataset.set_format('torch', columns=columns)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader
