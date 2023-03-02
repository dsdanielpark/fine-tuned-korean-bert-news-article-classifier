import numpy as np
import pandas as pd
import torch
import gluonnlp as nlp
from tqdm.notebook import tqdm
from kobert.pytorch_kobert import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from pytorch_kobert.dataloader import BERTDataset
from preprocess.processor import NLPdata
from pytorch_kobert.model import BERTClassifier


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
        print(class_dict[predicted_y[0]])
        
    return df_eval


if __name__ == "__main__":
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

    torch_bert_inference(model, test_dataloader, None, device)