#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

import gluonnlp as nlp
from tqdm.notebook import tqdm
from kobert.pytorch_kobert import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from dataloader import BERTDataset
from preprocess.processor import NLPdata
from model import BERTClassifier


max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

def torch_bert_inference(model: object, dataloader: object, save_path: str, device: object) -> pd.DataFrame:
    class_dict = {0: 'international', 1: 'economy', 2: 'society', 3: 'sport', 4: 'it', 5: 'politics', 6: 'entertain', 7: 'culture'}
    output_pred_arr = []
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        output_pred_arr.extend(out.detach().numpy())
    predicted_y = [np.argmax(x) for x in output_pred_arr]
    df_eval = pd.DataFrame([output_pred_arr, predicted_y])
    df_eval = df_eval.T
    df_eval.columns = ['pred', 'predicted_y']
    df_eval.reset_index(inplace=True)
    df_eval['predicted_y'] = [class_dict[int(x)] for x in df_eval['predicted_y']]
    if save_path is not None:
        df_eval.to_csv(save_path, encoding='utf-8-sig', index=False)
    else:
        print(f'>>> Prdicted news topic: {class_dict[predicted_y[0]]}')
        
    return df_eval

# TEMPORAL METHOD
def single_text_inference(model: object, cfg: dict, input_text: str):
    class_dict = {0: 'international', 1: 'economy', 2: 'society', 3: 'sport', 4: 'it', 5: 'politics', 6: 'entertain', 7: 'culture'}
    dataset_inf = [[input_text, '99']]
    data_inf = BERTDataset(dataset_inf, 0, 1, tok, cfg['max_len'], True, False)
    inf_dataloader = torch.utils.data.DataLoader(data_inf, batch_size=cfg['batch_size'])
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(inf_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        out = model(token_ids, valid_length, segment_ids)
        cls_pooled_denselayer = out.detach().numpy()
        predicted_y = np.argmax(cls_pooled_denselayer)
    
    print(f'>>> Prdicted news topic: {class_dict[predicted_y]}')

    return cls_pooled_denselayer,  predicted_y


if __name__ == "__main__":
    cfg = {'max_len' : 64,
           'batch_size' : 64}
    tokenizer = get_tokenizer()
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
    dataset_test = NLPdata().TSVDdataset("./data/sample.csv", 'cleanBody', 'category', 'mode3')
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    checkpoint = torch.load('./weights/ko-news-clf-torch-weight.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # torch_bert_inference(model, test_dataloader, None, device)

    sample_input = '한양대 경제학부 하준경 교수는 BBC 코리아에 현재 50,60대에 해당하는 기성세대가 1990년대 집을 살 때는 빚을 많이 지지 않고도 연소득의 서너 배, 혹은 다섯 배 정도면 충분히 괜찮은 집을 살 수 있었지만 현재는 소득 대비 집값이 서울 같은 경우 20배씩 되는 정도기 때문에 빚을 많이 질 수 밖에 없다며 특히 청년 세대의 경우 과거에 이미 집을 싸게 사 집값 상승의 효과를 누렸던 기성세대와 달리 집을 사기 위해 빚을 많이 내야 하기 때문에 부채 증가가 빠르게 되는 부분이 있다고 설명했다 이에 대해 수원에 거주하는 61세 A씨는 1980~90년대에는 10년만 열심히 하면 집을 살 수 있다는 생각을 너도 나도 했다며 실제 같은 또래 대부분의 주변 사람들이 그렇게 약간의 빚을 내 집을 구했다고 말했다 실제 위 보고서의 분석에 따르면 전체 가구에 비해 청년 가구의 전체 자산 규모 확대 속도가 떨어지고, 전체 자산 중 실물자산이 차지하는 비중도 작은 경향을 확인할 수 있다'
    single_text_inference(model, cfg, sample_input)