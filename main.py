from src.kobert_gluon.bert_clf_infererence import GluonBERTClassifierInferencer
from config import Config as CONF


if __name__ == "__main__":
    gluon_bert_inferencer = GluonBERTClassifierInferencer(CONF.gluon_cfg, CONF.params['GLUON_WEIGHT_PATH'], CONF.params['DATA_PATH'], CONF.params['SAVE_PATH'])
    gluon_bert_inferencer.infer()