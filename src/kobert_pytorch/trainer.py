import torch
from torch import nn
import gluonnlp as nlp
from tqdm.notebook import tqdm
from kobert.pytorch_kobert import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from dataloader import BERTDataset
from ..preprocess.processor import NLPdata
from model import BERTClassifier

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]

    return train_acc


def torch_save(model, optimizer, save_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )


def train_torch_bert(model, dataloader, save_path):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    t_total = len(dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total
    )
    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print(
                    "epoch {} batch id {} loss {} train acc {}".format(
                        e + 1,
                        batch_id + 1,
                        loss.data.cpu().numpy(),
                        train_acc / (batch_id + 1),
                    )
                )
        print("Epoch {}: train acc=={}".format(e + 1, train_acc / (batch_id + 1)))

        model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            test_acc += calc_accuracy(out, label)
        print("Epoch {}: test acc=={}".format(e + 1, test_acc / (batch_id + 1)))

        if save_path is not None:
            torch_save(model, optimizer, save_path)


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    dataset_test = NLPdata.TSVDdataset(
        "../data/test_set.csv", "cleanBody", "category", "mode3"
    )
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)
    train_torch_bert(model, test_dataloader, None)
