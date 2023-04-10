import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))
import pandas as pd
import numpy as np
import gluonnlp as nlp
import mxnet as mx
from kobert_gluon.model import BERTClassifier
from kobert_gluon.dataloader import BERTDataset
from src.preprocess.processor import NLPdata
from kobert.mxnet_kobert import get_mxnet_kobert_model
from kobert.mxnet_kobert import get_tokenizer


class GluonBERTClassifierInferencer(BERTClassifier, NLPdata):
    def __init__(self, cfg, model_path, data_path, save_path, prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        super(NLPdata).__init__()
        self.cfg = cfg
        self.device = mx.cpu()
        self.bert_base, self.vocab = get_mxnet_kobert_model(
            use_decoder=False, use_classifier=False, ctx=mx.cpu(), cachedir=".cache"
        )
        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)
        self.model = BERTClassifier(self.bert_base, num_classes=8, dropout=0.1)
        self.model.load_parameters(model_path)
        self.dataset_test = NLPdata().TSVDdataset(
            data_path, "cleanBody", "category", "mode2", None
        )
        self.data_test = BERTDataset(
            self.dataset_test, 0, 1, self.tok, cfg["max_len"], True, False
        )
        self.data_iter = mx.gluon.data.DataLoader(
            self.data_test, batch_size=int(cfg["batch_size"] / 2)
        )
        self.save_path = save_path

    def infer(self) -> List[list, list]:
        i = 0
        cls_dense_layers_val_list = []
        for i, (t, v, s, label) in enumerate(self.data_iter):
            if i > 1000:
                break
            i += 1
            token_ids = t.as_in_context(self.device)
            valid_length = v.as_in_context(self.device)
            segment_ids = s.as_in_context(self.device)
            label = label.as_in_context(self.device)
            output = self.model(token_ids, segment_ids, valid_length.astype("float32"))

            cls_dense_layers_val_list.extend(output.asnumpy())
            predicted_y = [np.argmax(x) for x in cls_dense_layers_val_list]
            class_dict = {
                0: "international",
                1: "economy",
                2: "society",
                3: "sport",
                4: "it",
                5: "politics",
                6: "entertain",
                7: "culture",
            }
            df_epocheval = pd.DataFrame(predicted_y, columns=["predicted"])
            df_epocheval.reset_index(inplace=True)
            df_epocheval["predicted"] = [
                class_dict[int(x)] for x in df_epocheval["predicted"]
            ]
            if self.save_path is not None:
                df_epocheval.to_csv(self.save_path, encoding="utf-8-sig")
            else:
                print(f"Predicted news topic: {class_dict[predicted_y[0]]}")

        return cls_dense_layers_val_list, predicted_y


# unit test
# from model import BERTClassifier
# from dataloader import BERTDataset
# from preprocess.processor import NLPdata
# if __name__ == "__main__":
#     cfg = {'max_len': 128, 'batch_size':32}
#     gluon_bert_inferencer = GluonBERTClassifierInferencer(cfg, "./weights/ko-news-clf-gluon-weight.pth", "./data/sample.csv", None)
#     gluon_bert_inferencer.infer()
