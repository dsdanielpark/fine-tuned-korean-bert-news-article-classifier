import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))
import pandas as pd
import numpy as np
import gluonnlp as nlp
import mxnet as mx
from model import BERTClassifier
from torch.utils.data import DataLoader
from kobert.mxnet_kobert import get_mxnet_kobert_model
from kobert.mxnet_kobert import get_mxnet_kobert_model
from kobert.mxnet_kobert import get_tokenizer
from dataloader import BERTDataset
from preprocess.processor import NLPdata 
ctx = mx.cpu()


def gluon_infer(model: object, data_iter: DataLoader, save_path:str, ctx=ctx) -> (list, list):
    i = 0
    cls_dense_layers_val_list = []
    for i, (t,v,s, label) in enumerate(data_iter):
        if i > 1000:
            break
        i += 1
        token_ids = t.as_in_context(ctx)
        valid_length = v.as_in_context(ctx)
        segment_ids = s.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = model(token_ids, segment_ids, valid_length.astype('float32'))
        
        cls_dense_layers_val_list.extend(output.asnumpy())
        predicted_y = [np.argmax(x) for x in cls_dense_layers_val_list]
        class_dict = {0: 'international', 1: 'economy', 2: 'society', 3: 'sport', 4: 'it', 5: 'politics', 6: 'entertain', 7: 'culture'}
        df_epocheval = pd.DataFrame(predicted_y, columns=['predicted'])
        df_epocheval.reset_index(inplace=True)
        df_epocheval['predicted'] = [class_dict[int(x)] for x in df_epocheval['predicted']]
        if save_path is not None:
            df_epocheval.to_csv(save_path, encoding='utf-8-sig')
        else:
            print(f'Predicted news topic: {class_dict[predicted_y[0]]}')

    return cls_dense_layers_val_list, predicted_y





if __name__ == "__main__":
    max_len = 128
    batch_size = 32
    bert_base, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=ctx, cachedir=".cache")
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    dataset_test = NLPdata().TSVDdataset("./data/sample.csv", 'cleanBody', 'category', 'mode2', None)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=int(batch_size/2))
    bert_base, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=ctx, cachedir=".cache")
    model = BERTClassifier(bert_base, num_classes=8, dropout=0.1)
    model.load_parameters("./weights/ko-news-clf-gluon-weight.pth")
    # cls_dense_layers_val_list, predicted_y = gluon_infer(model, test_dataloader, "gloun_infer_result.csv", ctx=ctx)
    cls_dense_layers_val_list, predicted_y = gluon_infer(model, test_dataloader, None, ctx=ctx) 