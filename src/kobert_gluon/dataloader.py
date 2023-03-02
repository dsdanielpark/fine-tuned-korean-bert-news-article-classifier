import gc
gc.collect()
import numpy as np
import gluonnlp as nlp
from tqdm.notebook import tqdm
from mxnet import gluon
import mxnet as mx


class BERTDataset(mx.gluon.data.Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        sent_dataset = gluon.data.SimpleDataset([[
            i[sent_idx],
        ] for i in dataset])
        self.sentences = sent_dataset.transform(transform)
        self.labels = gluon.data.SimpleDataset(
            [np.array(np.int32(i[label_idx])) for i in dataset])

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


