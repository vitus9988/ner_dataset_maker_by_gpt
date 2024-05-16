import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import torch.nn.functional as F
from seqeval.metrics import classification_report, f1_score as ner_f1_score, precision_score, recall_score, accuracy_score
import itertools
from sklearn.metrics import f1_score

origin = pd.read_csv('20240514_1k_ko.csv', encoding='utf-8')
origin['labels'] = origin['labels'].apply(literal_eval)
data = origin.to_dict('records')

LABELS = ['B-HT', 'I-HT', 'B-RT', 'I-RT', 'B-PS', 'I-PS', 'B-DT', 'I-DT', 'B-SP', 'I-SP', 'O']

def pad_sequences(seqs, pad_val, max_length):     
    _max_length = max([len(s) for s in seqs])
    max_length = min(max_length, _max_length)
    
    padded_seqs = []
    for seq in seqs:
        seq = seq[:max_length]
        pads = [pad_val] * (max_length - len(seq))
        seq = seq + pads
        padded_seqs.append(seq)

    return padded_seqs

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length, split='train'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        char_labels = item['labels']

        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        labels = []
        for i in range(input_ids.shape[0]):
            span = inputs.token_to_chars(i)
            if span is None or span.start >= len(char_labels):
                labels.append(len(LABELS)-1) # O
            else:
                labels.append(char_labels[span.start])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def collate_fn(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
            
    
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-small')

rand_idxs = np.random.permutation(range(len(data)))
train_idxs = rand_idxs[100:]
valid_idxs = rand_idxs[:100]

train_data = [data[i] for i in train_idxs]
valid_data = [data[i] for i in valid_idxs]

train_dataset = Dataset(train_data, tokenizer, 256)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn)

valid_dataset = Dataset(valid_data, tokenizer, 256)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=valid_dataset.collate_fn)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=2)
    
    preds = [pred[label_mask] for pred, label_mask in zip(preds, [np.array(label) != -100 for label in labels])]
    labels = [label[label_mask] for label, label_mask in zip(labels, [np.array(label) != -100 for label in labels])]
    
    preds = [[LABELS[p] for p in pred] for pred in preds]
    targets = [[LABELS[t] for t in target] for target in labels]

    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = ner_f1_score(targets, preds)
    accuracy = accuracy_score(targets, preds)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }




num_labels = len(LABELS)
id2label = {i:l for i,l in enumerate(LABELS)}
label2id = {l:i for i,l in enumerate(LABELS)}

model = AutoModelForTokenClassification.from_pretrained('klue/roberta-small', num_labels=num_labels, ignore_mismatched_sizes=True,id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    'upload model name',
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()
