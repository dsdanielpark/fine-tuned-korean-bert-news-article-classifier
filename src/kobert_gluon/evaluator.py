# TEMPORAL CLASS, will not be refactored.




































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


class GluonBERTClassifierEvaluator(BERTClassifier, NLPdata):
    def __init__(self, cfg, model_path, data_path, save_path, prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        super(NLPdata).__init__()
        self.cfg = cfg 
        self.device = mx.cpu()
        self.bert_base, self.vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=mx.cpu(), cachedir=".cache")
        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)
        self.model = BERTClassifier(self.bert_base, num_classes=8, dropout=0.1)
        self.model.load_parameters(model_path)
        self.dataset_test = NLPdata().TSVDdataset(data_path, 'cleanBody', 'category', 'mode2', None)
        self.data_test = BERTDataset(self.dataset_test, 0, 1, self.tok, cfg['max_len'], True, False)
        self.data_iter = mx.gluon.data.DataLoader(self.data_test, batch_size=int(cfg['batch_size']/2))
        self.save_path = save_path

    
    def _mxnet_inference(self, model: object, save_path:str) -> (list, list):
        class_dict = {0: 'international', 1: 'economy', 2: 'society', 3: 'sport', 4: 'it', 5: 'politics', 6: 'entertain', 7: 'culture'}
        output_pred_arr = []
        for i, (t,v,s, label) in enumerate(self.data_iter):
            token_ids = t.as_in_context(self.ctx)
            valid_length = v.as_in_context(self.ctx)
            segment_ids = s.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            output = model(token_ids, segment_ids, valid_length.astype('float32'))
            output_pred_arr.extend(output.asnumpy())
        predicted_y = [np.argmax(x) for x in output_pred_arr]
        df_eval = pd.DataFrame([output_pred_arr, predicted_y])
        df_eval = df_eval.T
        df_eval.columns = ['pred', 'predicted_y']
        df_eval.reset_index(inplace=True)
        df_eval['predicted_y'] = [class_dict[int(x)] for x in df_eval['predicted_y']]
        # df_eval.to_csv(save_path, encoding='utf-8-sig', index=False)
        
        return df_eval
    
    def make_infer_result_df(self, model_path, target_data_f_path, result_save_f_path, mode, num_cluster):
        bert_base, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=self.ctx, cachedir=".cache")
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        if mode == 'mode2':
            dataset_test = NLPdata().TSVDdataset(target_data_f_path, 'title', 'category', mode)
        elif mode == 'mode4':
            dataset_test = NLPdata().TSVDdataset(target_data_f_path, 'cleanBody', 'category', mode, num_cluster)
        else:
            dataset_test = NLPdata().TSVDdataset(target_data_f_path, 'cleanBody', 'category', mode)
        data_test = BERTDataset(dataset_test, 0, 1, tok, self.cfg['max_len'], True, False)
        test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=int(self.cfg['batch_size']/2))
        bert_base, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=self.ctx, cachedir=".cache")
        model = BERTClassifier(bert_base, num_classes=8, dropout=0.1)
        model.load_parameters(model_path)
        df_eval = self._mxnet_inference(model, test_dataloader, None, ctx=self.ctx) 
        df_test = pd.read_csv(target_data_f_path)
        df_test.reset_index(inplace=True)
        infer_result_df = df_test.merge(df_eval, on='index', how='left')
        infer_result_df.to_csv(result_save_f_path, index=False, encoding='utf-8-sig')

        return infer_result_df

    def _save_metric_report(infer_result_df, save_metric_f_name):
        y_true = infer_result_df.category
        y_pred = infer_result_df.predicted_y
        from sklearn.metrics import classification_report
        mc = classification_report(y_true, y_pred, output_dict=True)
        df_mc = pd.DataFrame(mc)
        df_mc.to_csv(save_metric_f_name)

        return mc

    def make_infer_result_df(self, model_path, target_data_f_path, result_save_f_path, mode, num_cluster):
        # pre-trained bert base down and use cached one.
        bert_base, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=ctx, cachedir=".cache")
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

        if mode == 'mode2':
            dataset_test = NLPdata().TSVDdataset(target_data_f_path, 'title', 'category', mode)
        elif mode == 'mode4':
            dataset_test = NLPdata().TSVDdataset(target_data_f_path, 'cleanBody', 'category', mode, num_cluster)
        else:
            dataset_test = NLPdata().TSVDdataset(target_data_f_path, 'cleanBody', 'category', mode)

        data_test = BERTDataset(dataset_test, 0, 1, tok, self.cfg['max_len'], True, False)
        test_dataloader = mx.gluon.data.DataLoader(data_test, batch_size=int(self.cfg['batch_size']/2))
        bert_base, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False, ctx=ctx, cachedir=".cache")
        model = BERTClassifier(bert_base, num_classes=8, dropout=0.1)
        model.load_parameters(model_path)
        df_eval = self._mxnet_inference(model, test_dataloader, None, ctx=self.ctx) 
        df_test = pd.read_csv(target_data_f_path)
        df_test.reset_index(inplace=True)
        infer_result_df = df_test.merge(df_eval, on='index', how='left')
        infer_result_df.to_csv(result_save_f_path, index=False, encoding='utf-8-sig')

        return infer_result_df
    



