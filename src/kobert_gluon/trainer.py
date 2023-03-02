import time
import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from tqdm.notebook import tqdm


class BERTTrainer:
    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device):
        self.cfg = cfg # config for training: see class Config
        self.model = model
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device
        self.device = mx.cpu() # device name
        self.all_model_params = model.collect_params()
        self.log_interval = 4

    @classmethod
    def fine_tunning(self, config, model, data_train, train_dataloader, test_dataloader, save_path):
        accumulate = 4
        step_size = config.batch_size * accumulate if accumulate else config.batch_size
        num_train_examples = len(data_train)
        num_train_steps = int(num_train_examples / step_size * config.num_epochs)
        warmup_ratio = 0.1
        num_warmup_steps = int(num_train_steps * warmup_ratio)
        step_num = 0
        trainer = gluon.Trainer(model.collect_params(), 'bertadam',
                        {'learning_rate': config.lr, 'epsilon': 1e-9, 'wd':0.01})
        loss_function = gluon.loss.SoftmaxCELoss()
        
        for epoch_id in range(config.config.num_epochs):
            metric = mx.metric.Accuracy()
            metric.reset()
            step_loss = 0
            for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                if step_num < num_warmup_steps:
                    new_lr = config.lr * step_num / num_warmup_steps
                else:
                    non_warmup_steps = step_num - num_warmup_steps
                    offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                    new_lr = config.lr - offset * config.lr
                trainer.set_learning_rate(new_lr)

                with mx.autograd.record():
                    token_ids = token_ids.as_in_context(self.device)
                    valid_length = valid_length.as_in_context(self.device)
                    segment_ids = segment_ids.as_in_context(self.device)
                    label = label.as_in_context(self.device)
                    out = model(token_ids, segment_ids, valid_length.astype('float32')) # forward
                    ls = loss_function(out, label).mean()

                ls.backward() # backward
                if not accumulate or (batch_id + 1) % accumulate == 0:
                    trainer.allreduce_grads()
                    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
                        v.wd_mult = 0.0
                    params = [
                        p for p in model.collect_params().values() if p.grad_req != 'null'
                    ]
                    nlp.utils.clip_grad_global_norm(params, 1)
                    trainer.update(accumulate if accumulate else 1)
                    step_num += 1
                    if accumulate and accumulate > 1:
                      self.all_model_params.zero_grad()

                step_loss += ls.asscalar() # calculate loss

                metric.update([label], [out]) # calculate metric
                if (batch_id + 1) % (50) == 0:
                    print('[Epoch {} Batch {}/{}] loss={:.4f}, config.lr={:.10f}, acc={:.3f}'
                                .format(epoch_id + 1, batch_id + 1, len(train_dataloader),
                                        step_loss / self.log_interval,
                                        trainer.learning_rate, metric.get()[1]))
                    step_loss = 0
            test_acc = self._evaluate_accuracy(model, test_dataloader, self.device)
            print('Test Accuracy: {}'.format(test_acc))
            model.save_parameters(save_path) # model save

    def _evaluate_accuracy(model, data_iter, self.device=self.device):
        acc = mx.metric.Accuracy()
        i = 0
        output_dict = {}
        for i, (t,v,s, label) in enumerate(data_iter):
            token_ids = t.as_in_context(self.device)
            valid_length = v.as_in_context(self.device)
            segment_ids = s.as_in_context(self.device)
            label = label.as_in_context(self.device)
            output = model(token_ids, segment_ids, valid_length.astype('float32'))
            acc.update(preds=output, labels=label)
            if i > 1000:
                break
            i += 1
            now = int(round(time.time() * 1000))
            pred = mx.nd.argmax(output, axis=0)
            y = np.where(pred == max(pred))[0][0]
            output_dict[i] = int(y)

        try:
            class_dict = {0: 'international', 1: 'economy', 2: 'society', 3: 'sport', 4: 'it', 5: 'politics', 6: 'entertain', 7: 'culture'}
            df_eval = pd.DataFrame(output_dict, index=['predicted_topic'])
            df_eval = df_eval.T
            df_eval.reset_index(inplace=True)
            df_eval['predicted_topic'] = [class_dict[int(x)] for x in df_eval['predicted_topic']]
            df_eval.to_csv(f"../result/{now}_temp_output.csv", encoding='utf-8-sig', index=False)
        except Exception as e:
            print(e)
            pass

        return(acc.get()[1])
