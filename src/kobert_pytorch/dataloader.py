from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm
from kobert.pytorch_kobert import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.pytorch_kobert import get_pytorch_kobert_model
import pandas as pd
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
