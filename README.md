# Korean-news-topic-classification-using-KO-BERT(public)

> 🚸 **Be careful when cloning this repo**: It contains large NLP model weights. (>0.3GB, [`git-lfs`](https://git-lfs.com/))

<br> 

## TASK: Multi-category(8 classes) Korean News Topic Classifier 한글 뉴스 토픽 다중 분류 모델 <Br>
- Perform a simple task to compare the performance of KO-BERT and implementations of BERT in different frameworks <br>
- FrameWork: `PyTorch`, `APACHE MXNET GluonNLP` and `Hugging-Face` for modeling, `Scikit-Learn` to calculate evaluation metric.
- Pre-trained model: [`KO-BERT`](https://github.com/SKTBrain/KoBERT), `BERT`, [`SBERT`](https://www.sbert.net/)
- Dataset: Korean News Topic Classification Dataset (특정 회사에서 수집하여 비공개로 제공) <br>
- Evaluation: Accuracy <br>
- Duration: Jan 16,2023 - Jan 19,2023 (4days) <br>
- Author: [Daniel Park, South Korea](https://github.com/DSDanielPark) <br>

<br>

# TL ; DR
### Main Task: 한국어 뉴스 기사의 토픽(8개) 분류 모델(BERT Classifier) 생성
### Sub Task 
- (프레임워크 비교) `GluonNLP`, `PyTorch`, `Hugging-Face` 프레임워크의 BERT 모델 구현 비교
- (인풋 정보량에 따른 BERT Classifier의 성능 비교) 뉴스의 제목, 제목+본문, 본문을 인풋으로 사용하였을 경우의 모델 분류 성능 정량적 비교
- (사전 군집화정보에 따른 모델 학습 속도 비교) Multilangual BERT(SBERT)를 통해 사전에 군집 정보를 주었을 경우, 모델의 학습 속도가 달라지는지 여부 비교
  - PyTorch의 경우, initial [CLS] 토큰에 정보를 임베딩하고
  - GluonNLP의 경우, 전처리 단계에서 군집에 대한 정보를 인풋에 삽입하여 비교
### 본 레포지토리는
- MXNet GluonNLP BERT weight 파일을 git-lfs을 통해서 제공하고, `data/sample.csv`를 제공합니다. 
- 추후 범용적으로 사용할 수 있는 패키지 pypi를 통해 배포합니다. 
- To-Do: Hugging Face 프레임워크, XAI, GPT2, GPT3, BERT Pipeline etc.


<br><br><br>

## Repository folder tree
```
📦fine-tuned-korean-BERT-news-article-classifier 
 ┣ 📂data
 ┃ ┣ 📂csv
 ┃ ┣ 📂imgs
 ┃ ┣ 📜sample.csv                          # sample data will provide
 ┃ ┣ 📜test_set.csv
 ┃ ┗ 📜train_set.csv
 
 ┣ 📂experiments                           # dummy for experiments
 ┃ ┣ 📂experiment_weights
 ┃ ┣ 📜exp.md
 ┃ ┗ 📜exp_metric.md
 
 ┣ 📂notebooks                             # notebook will provide

 ┣ 📂src
 ┃ ┣ 📂kobert                              # SKT KOBERT
 ┃ ┣ 📂kobert_gluon                        # gloun nlp 프레임워크 실험을 위해 생성한 모듈
 ┃ ┣ 📂kobert_hf                           # SKT KOBERT
 ┃ ┣ 📂kobert_pytorch                      # torch bert 실험을 위해 생성한 모듈
 ┃ ┣ 📂preprocess                          # 본 레포지토리 실험을 위한 전처리 클래스

 ┣ 📂weights
 ┃ ┣ 📜ko-news-clf-gluon-weight.pth        # provide throught git-lfs (0.3 GB)
 ┃ ┗ 📜ko-news-clf-torch-weight.pth        # will not provide (>1.0 GB)

 ┣ 📜.gitattributes                        # git-lfs managing
 ┣ 📜.gitignore
 ┣ 📜config.py                             # config
 ┣ 📜LICENSE
 ┣ 📜main.py                               # main.py (gluon inference만 제공)
 ┣ 📜README.md
 ┗ 📜requirements.txt
```




<br><Br>
## [Optional] Related Packages
#### 1. [`QuickShow`](https://pypi.org/project/quickshow/): pandas.DataFrame을 인풋으로 쉽고 빠르게 시각화할 수 있는 패키지
```bash
$ pip install quickshow
```
- 이번 프로젝트에 활용된 일부 시각화 모듈을 배포하였습니다. 지금까지 제가 여러 프로젝트에서 사용된 다양한 모듈을 편리하게 래핑하여 추후 업데이트할 예정입니다.


<br><br>

## Quick Start
본 레포지토리는 highly encapsulation된 gluon weight bert classifier의 inference class를 제공합니다.
```
$ git clone https://github.com/DSDanielPark/fine-tuned-korean-BERT-news-article-classifier.git
$ cd fine-tuned-korean-BERT-news-article-classifier
$ pip install -e .
$ python main.py
>>> Predicted news topic: international
```


<Br>

You use this optional args as below.
```
for gluon weight inference only

optional arguments:
  -h, --help            show this help message and exit
  --gluon_weight_path GLUON_WEIGHT_PATH                        # glouon weight file path
  --data_path DATA_PATH                                        # input csv data path
  --save_path SAVE_PATH                                        # save csv full path
```

<br>
<br>

## Remark
- *Please note that the domain was not fully searched because it is not mission critical artificial intelligence and is intended to simply identify NLP(Natural Language Processing) models in korean and perform requested tasks in a short period of time with limited data.*
- If it is used in actual service, some more tricks or experiments may be required. but as mentioned above, it is expected that the service can be maintained satisfactorily if applied in the same way with the heuristic method. There may be considerations such as continuous learning.
<!-- - *This is just a toy project! please enjoy it!* <br>
![](https://github.com/DSDanielPark/news-article-classification-using-koBERT/blob/main/imgs/enjoy2.gif) -->
<br>

## Outline
- **Problem Definition:** 
<br> Create a model that classifies articles into the following 8 categories with the title and body as input.
<br> `Categories` = ['society', 'politics', 'international', 'economy', 'sport', 'entertain', 'it', 'culture'] <br><br>
- **Data Description:**
<br> The Total 3,000 korean news dataset(Train: 2,100 articles, Test 900: articles, 7:3 split) consist of title, body and category columns. <br><br>


        ```
        title: korean article title
        cleanBody: korean article body
        category: of article (eight classes)
        ```



  |  | title |cleanBody|category|
  |---:|:---|:---|:---|
  |  0 | 보건복지부, 새해 청년과 지역이 함께 지역사회서비스를 개발제공해 사회서비스 고도화에 앞장서 | 보건복지부(장관 조규홍)는 2023년 청년사회서비스사업단(이하 ‘청년사업단’)선정 및 운영을 위해 오는 1월 27일(금)부터 2월 15일(수)까지 약 3주간 전국 17개 시도(광역 지자체)를 대상으로 청년사업단 공모를 실시한다고 밝혔습니다. ...(중략)... 본 기사는 깃허브 데이터 예시를 위해 보건복지부 보도자료를 참조하여 생성하였습니다. | society    |
  |  1 | BTS, 10년간 빌보드 'Hot 100' 차트 1위곡 최다 보유 가수로 기록돼 | 방탄소년단(이하 "BTS")가 미국 빌보드 메인 싱글 차트 'HOT 100'의 정상에 가장 많이 오른 아티스트로 선정되었습니다. BTS가 미국 유명 가수 드레이크나 아리아나 그란데 등을 제치고, 메인 싱글 차트 핫 100에 올라 다시 한번 K-POP의 위상을 세상에 떨쳤습니다. ...(중략)... BTS는 새로운 음악과 안무로 다시 컴백을 예고하고 있습니다. 이상 깃허브 뉴스 다니엘 기자였습니다. | entertain |


<br><Br>

- **EDA:**
<br> I analyzed the quality of the data, but I couldn't find any outliers because it was a very well-refined dataset. 
<br> Since this data was processed for a specific purpose by a Korean IT company, data preprocessing was unnecessary. 
<br> Replace the next image with data analysis.
<br><Br>
<img src="./data/imgs/fig.png" width="600"> <br> *Fig 1. 테스트셋과 트레인 셋의 데이터 분포(이 태스크의 경우, 추가적인 데이터 수집 및 학습이 용이하므로 imbalance는 고려하지 않음)*
<br><br>
  - Since the advent of large-scale natural language processing models, feature engineering on corpus has not been of great significance.
  - In `MODE 4` and `MODE 5`, a test was conducted to give some information on the similarity of articles in advance through k-mean clustering based on the tokens embedded from the pre-trained BERT model.

  <br>

- **Input Features list:**
  1. *input_ids* - are the ids associated with each token as per BERT vocabulary.
  2. *input_mask* - differentiate between padding and real tokens.
  3. *Segment_ids* - will be a list of 0’s, as in classification task there is only a single text.
  4. *label_id* - corresponds to Label Encoder classes
<br><br>



# Results
All model architecture and learning conditions in the whole pipeline were fixed except the initial layer in MODE5. <br>
- Mode1 to Mode4: `MXNet` Framework
- Mode5: `Pytorch` Framework to modify the initial layer
- Pre-trained Models are KO-BERT [3], SBERT [5].

  | |`MODE 1`|`MODE 2`|`MODE 3`|`MODE 4`|`MODE 5`|
  |:---:|:---:|:---:|:---:|:---:|:---:|
  |Data|article body only|articel title only|article with title|article with title which have clustered information from SBERT model|article with title|
  |Model|KO-BERT|KO-BERT|KO-BERT|KO-BERT, SBERT|KO-BERT, SBERT with clustered information in initial hidden layer|
  |TestSet Accuracy|0.8895|0.8269|0.8864|0.8895|-|
  |Remark|Model architecture and all conditions in models were fixed.|-|-|-|Model architecture and initial layer are changed only in `MODE5`.|
- As I mentioned in the 'Experiments' section, All conditions in the whole pipeline were fixed except the initial layer in MODE5 and Pre-trained Models were KO-BERT [3], SBERT [5].



<br><br>


## Experiments


- 본 실험 아이디어를 위해 참고한 논문들은 없으나, 후향적 분석과정에서 비슷한 컨셉의 논문들을 발견하여 `References`에 기재함.
- 본 실험들은 한국어를 통한 실질적인 태스크 적용시에 참고용으로 수행.<br><br>

- 실험은 크게 다음 4가지 가설을 확인하기 위하여 설계되었음. <br>
  1) 한글 기사의 본문, 제목, 본문과 제목을 통한 분류 성능의 차이 확인하고자 설계됨 - `MODE1~MODE3`
  2) 일종의 지식증류를 통한 KO-BERT의 미세 조정(fine-tunning) 과정에서 Teacher Model(SBERT)로 부터 생성한 군집에 대한 정보가 모델 수렴에 얼마나 영향을 줄 수 있는지 확인하고자 설계됨 - `MODE3~MODE4`
  3) 2번의 과정에서 SBERT로 생성된 정보(군집의 수)가 얼마나 옳은지(동일한 클래스가 동일 군집으로만 구성되어 있는지)가 모델 파라미터 미세조정에 영향을 주는지 확인하기 위해 설계됨 - `MODE4`: `EXP8~EXP11`
  4) 마지막으로 SBERT로부터 생성된 정보를 모델 인풋으로 주입시키는 것과 initial hidden layer에 주입시키는 것의 차이를 PyTorch와 MXNet 프레임워크를 통해 비교함 - `MODE4~MODE5`

<br>
<br>

The Mode 1, Mode 2, and Mode 3 experiments were designed to see how well a subject could classify 8 topics into the body and headlines of a news article.
  - `MODE 1`: It learns data containing only the body of news articles as input.
  - `MODE 2`: It learns data containing only news article titles as input.
  - `MODE 3`: It learns data containing both news article titles and body as input.

<br>

Mode 4 and Mode 5 are designed to create and experiment with a kind of distilled BERT model.
- This experiment is designed *to see if it can be fine-tuned a bit faster based on the clustering information generated by SBERT.*
- In this experiment, the information clustered by SBERT, a multilingual model, is used to fine-tune the KO-BERT model.
  - `MODE 4`: Cluster information generated from SBERT, a multilingual model, is used as model input, and this experiment used the mxnet framework in the same way as Modes 1 to 3.
  - `MODE 5`: Clustering information generated from SBERT was input to the initial layer using the PyTorch framework. The results of this experiment may be slightly different from Modes 1 to 4, so they are not described.
    - However, as in Mode 4, it was confirmed through several iterations that almost similar accuracy can be reached faster if you have cluster information generated from SBERT.


- You can see whole experiments result in [`experments/exp.md`](https://github.com/DSDanielPark/fine-tuned-korean-BERT-news-article-classifier/blob/main/experiments/exp.md) 
- In the case of EXP2 and EXP9, it was repeatedly performed to track and observe the learning rate, confirming similar learning patterns.

  | No | Condition | Best Test Accuracy | Epoch | 
  |:---:|:---|:---:|:---:|
  EXP1 | mode=1, batch_size=32 | 0.8842 | at Epoch 10
  EXP2 | mode=1, batch_size=32, epoch=20 without early stopping. | 0.8895 | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;at epoch 12&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  EXP3 | mode=1, batch_size=16 | 0.8800 | at epoch 4
  EXP4 | mode=1, batch_size=32 repeat of exp1 for check effect of randomness. | 0.8864 | at epoch 7
  EXP5 | mode=1, batch_size=32 | 0.8874 | at epoch 7 
  EXP6 | mode=2, batch_size=32 | 0.8269 | at epoch 7
  EXP7 | mode=3, batch_size=32 | 0.8864 | at epoch 6
  EXP8 | mode=4, batch_size=32, cluster_numb=8 | 0.8789 | at epoch 3
  EXP9 | mode=4, batch_size=32, cluster_numb=16 | 0.8895 | at epoch 5
  EXP10 | mode=4, batch_size=32, cluster_numb=32 | 0.8641 | at epoch 2
  EXP11 | mode=4, batch_size=32, cluster_numb=64 | 0.8885 | at epoch 7

<br>
<br>



# Evaluation
You can see evaluation metric of whole experiments in [`exp/exp_metric.md`](https://github.com/DSDanielPark/fine-tuned-korean-BERT-news-article-classifier/blob/main/experiments/exp_metric.md).

 <img src="./data/imgs/result.png" width="1000"><BR> *Fig 2. 각 실험별 F1 score, Recall, Precision*

<br><br>
<br><br>참고.
 F1 score is a performance metric for classification and is calculated as the harmonic mean of precision and recall.

<p align="center">
  <img src="./data/imgs/recall_precision.png" width="500"><br> 
  
</p>

*Fig 3. Reference figure to explain Recall, Prcision, Accuracy / Maleki, Farhad & Ovens, Katie & Najafian, Keyhan & Forghani, Behzad & Md, Caroline & Forghani, Reza. (2020). Overview of Machine Learning Part 1. Neuroimaging Clinics of North America. 30. e17-e32. 10.1016/j.nic.2020.08.007.*
 <br>

<br>

## Confusion Metrix and Heatmap in`EXP5` for each topic.
전체적인 결과는 [`exp/exp_metric.md`](https://github.com/DSDanielPark/fine-tuned-korean-BERT-news-article-classifier/blob/main/experiments/exp_metric.md)에 공개하였으며, Evaluation 도표를 통해 충분히 결과에 대한 유추가 가능하므로 모든 실험에 대한 Confusion Metrix 제공은 생략하며, 다음 예시를 통해 갈음함.
<br>

1. Classification report

    | Topic |   Precision |   Recall |   F1-Score |    Support |
    |:--------------:|:------------:|:---------:|:-----------:|:-----------:|
    | Culture       |    0.833333 | `0.666667` |   0.740741 |  `15`        |
    | Economy       |    0.693878 | 0.871795 |   0.772727 | 117        |
    | Entertain     |    0.868132 | 0.963415 |   0.913295 |  82        |
    | International |    0.878049 | 0.870968 |   0.874494 | 124        |
    | IT            |    0.722222 | 0.722222 |   0.722222 |  18        |
    | Politics      |    0.965278 | 0.798851 |   0.874214 | 174        |
    | Society       |    0.874101 | 0.823729 |   0.848168 | 295        |
    | Sport         |    0.906977 | `1.000000`        |   0.95122  | `117`        |```

    |  avg  |   |   |    |     |
    |:--------------:|:------------:|:---------:|:-----------:|:-----------:|
    | accuracy      |    0.860934 | 0.860934 |   0.860934 |   0.860934 |
    | macro avg     |    0.842746 | 0.839706 |   0.837135 | 942        |
    | weighted avg  |    0.86909  | 0.860934 |   0.861426 | 942        |

<br>

2. Heatmap 

    <img src="./data/imgs/cm_exp5.png" width="500"> <br>
    *Fig 4. Heatmap of Confusion Metrix in `EXP5`*



<br>
<br>

# Embedding Token Visuallization<br>

  #### **1. SBERT로부터 생성된 토큰의 유클리디안 거리를 통해 K-means 군집화를 수행할 경우, 클러스터 수 별 Ground Truth Category 분포 시각화**

  - 파란석 y-tick 밴드 사이가 각 군집을 의미
  - 군집안에 카테고리가 없을 경우 시각화되지 않으므로 밴드가 좁을수록 존재하는 Ground Truth Category의 고유값의 수가 적음을 의미
  - 전체 테이블은 [`csv`](https://github.com/DSDanielPark/fine-tuned-korean-BERT-news-article-classifier/blob/main/data/csv)폴더 안에서 확인

    <img src="./data/imgs/cluster8.png" width="220" height='200'> <br>
    <img src="./data/imgs/cluster16.png" width="450" height='200'><br>
    <img src="./data/imgs/cluster32.png" width="900" height='200'><br>
    *Fig 5. 클러스터별 Ground Truth Label의 분포. 위에서부터 아래 순서로 8, 16, 32개의 cluster로 군집화한 결과, 파란 밴드가 Cluster의 경계를 의미하며, 밴드 사이(클러스터별) Ground Truth Category의 분포정도를 표시함.*
    - SBERT로 사전에 군집에 대한 정보를 이식하여, KO-BERT의 fine-tunning을 진행하였는데, EXP8부터 EXP11까지 8개(`EXP8`)부터 64개(`EXP11`)까지 군집에 대한 수를 변경하여 봄. (64개의 경우 유의미한 차이가 없으므로 클러스터별 Ground Truth Label의 분포 시각화를 생략)
    - 실험결과는 기대와 동일하게 8개의 군집의 경우, 군집이 충분히 유사한 기사들을 묶어두지 못하였고(클러스터의 수가 너무 적음),  32개 이상으로 과하게 군집화하였을 경우, 너무 많은 군집의 수로 인해 모델 학습에 좋지 않은 영향을 미친 것으로 보임.
    - SBERT로 생성된 토큰의 유사도를 임베딩해주는 것이 KO-BERT fine-tunning에 영향을 줄 수 있는지 확인하는 목적이였으므로, 추가적인 최적 군집수를 도출은 불요.
    - 너무 적은 군집의 수(<=the number of classes)나 너무 많은 군집의 수(>=4 times the number of classes)는 모델 학습에 좋은 영향을 미치지 않을 가능성을 확인함.
    - `EXP4`,`EXP11`의 결과와 학습 양상이 비슷하므로 너무 군집의 수가 많은 EXP11에서는 KO-BERT의 미세조정에 SBERT로부터 생성된 Cluster정보를 Representation으로 사용하지 않은 것으로 추정함. 
    - `MXNET` 프레임워크에서는 인풋 데이터에 군집에 대한 정보를 삽입하는 것으로 구현하였고, `PyTorch` 프레임워크에서는 initial layer에 군집에 대한 정보를 [CLS] 토큰에 임베딩하여 initial hidden layer에 정보를 삽입하였으며, 프레임워크와 상관없이 조금 더 빠르게, 높은 성능에 도달할 수 있는 것을 확인함.
      
  <br>
  <br>
  

  
  
  
  <br>

  #### **2. SBERT로부터 생성된 토큰 시각화** <br>

  Multilingual BERT Embedding Space에 대한 연구[[10]](https://aclanthology.org/2022.findings-acl.103.pdf)와 Recommender Systems에서의 Embedding Space에 대한 연구[[11]](https://arxiv.org/abs/2105.08908)와 비슷하게 Token이 조금 더 Isotropic한 분포를 보이는 것이 좋은 성능을 나타낸 것을 확인할 수 있었다. 본 레포지토리에서 토큰 시각화에 사용한 SBERT 역시 Multilingual BERT이므로 상기[10]의 논문과 비슷한 환경에서의 실험이라고 가정하는 것에 큰 무리가 없을 것으로 판단하였다. <br> <br>제목으로부터 생성된 Embedding Space에서의 Token의 분포가 조금 더 isotropic한 것을 확인할 수 있었으며, 실제로 `EXP6`(기사의 제목만으로 학습)과 `EXP7`(기사의 본문만으로 학습)의 결과를 보면 Accuracy `0.8269`와 `0.8864`로 약 `0.595` 더 높은 정확도를 보이는 것을 확인할 수 있었다. 
  <br><br>
  결과적으로 BERT의 fine-tunning 태스크 이전에 Low Dimension(2DIM or 3DIM) Space로 시각화하여 조금 더 isotropic token 분포를 갖는 데이터셋이나 전처리 방법을 찾는 것이 조금 더 좋은 모델 학습 결과를 보일 수 있을 수 있음을 시사한다. <br><br>
  2-1 기사의 제목으로부터 생성된 토큰 시각화
  <img src="./data/imgs/title_token_vis.png" width="1000"><br>
  *Fig 6. Visualization of embedded tokens from SBERT for title of Korean articles*<br><br>
  2-2 기사 본문으로부터 생성된 토큰 시각화 <br>
  <img src="./data/imgs/body_token_vis.png" width="1000"> <br>
  *Fig 7. Visualization of embedded tokens from SBERT for body of Korean articles* <br><br>
  

  #### **3. KO-BERT fine-tunning 과정에서의 CLS Token 시각화**
  in the [HuggingFace BERT documentation](https://huggingface.co/docs/transformers/model_doc/bert#bertmodel), the returns of the BERT model are `(last_hidden_state, pooler_output, hidden_states[optional], attentions[optional])`
<br><BR>

# Discussion
- **If there is anything that needs to be corrected, productive criticism is always welcome.** <br>

  ### About total result
  - 태스크가 간단하였으므로 BERT의 fine-tuning만으로 단기간 안에 0.8이상의 분류기를 생성함.
  - 기사의 본문을 포함할 경우, `0.8641` ~ `0.8895`의 테스트 셋 Accuracy의 8개의 토픽 분류 모델을 생성함.
  - 기사의 제목만을 포함할 경우, `0.8269`의 테스트 셋 Accuracy를 확인함.
  - Sentence Embedding한 결과를 BERT 모델의 인풋으로 주는 실험에서는 Sentence Embedding한 Cluster의 수별로 약간의 차이가 있었으나, 아주 적은 정보(1자리 혹은 2자리의 정수)만으로 조금 더 빠르게 최대 Accuracy인 `0.8895`로 수렴함을 확인함.
  - 전반적으로 충분한 실험이 수행되진 않았지만, 몇 가지 조건에 대해 반복적인 학습을 진행해 본 결과, 현재 실험조건에서의 최고 성능은 `0.8895` 정도였고, 사전에 Cluster에 대한 정보를 줄 경우, 최고 성능에 더 빠르게 도달함을 확인함.
  - 기대와 동일하게, 제대로 된 약간의 초기 클러스터 정보만 있어도 BERT 모델이 조금 더 빠르게 수렴하는 것을 확인할 수 있었고, BERT 모델의 경우 Early Stop 조건을 조금 관대하게 주는 것이 좋음을 확인함.
  - 위 코드 작업이나 실험이 매우 단기간(정확도 0.88대의 base-line 도출까지 대략 3일)에 이뤄졌고, KO-BERT fine-tuning 성능을 확인하기 위한 것에 초점을 맞춰 구성되었음.
  - 본 태스크는 1) 데이터에 비해 태스크 난이도가 매우 낮고, 2) 고도의 정확도를 요구하지 않으며, 3) 몇가지 트릭 및 룰로 더 빠르게 분류가 가능하므로 BERT를 해당 태스크에 적용하기 위한 실험이라기 보다는 다른 어려운 태스크에 BERT 모델의 적용할 때 유용한 몇가지를 테스트하기 위해서 구성되었음. 
  - 추가적으로 GPT-3를 위한 리소스를 확보하였으므로, 동일한 태스크와 실험등을 통해 GPT-3와 BERT 모델의 적용 비교를 추가하고자 함.
  <br>
  <br>  

  ### Further experiments
  - `(Qualitative check)` 전처리 없이 데이터 셋의 원본만으로도 요구사항이었던 70% 이상의 정확도를 보였으므로, 추가적인 전처리를 진행하지 않았지만, 어떤 토큰이 모델 인퍼런스에 영향을 주는지 확인하고, 모델이 인풋 기사 데이터에서 이상한 토큰을 Representation으로 사용하고 있는지는 않은지 체크해 볼 필요가 있음.
  - `(eXplainable AI)`추가적으로 토큰의 영향도를 시각화해서 SBERT로 준 Cluster가 어느 정도의 영향이 있었는지 확인할 계획.
  - `(Understanding the BERT model)` 추가적으로 배치 사이즈에 대한 차이, BERT 모델이 볼 수 있는 max_length의 차이를 관찰하면 향후 fine-tuning할 때 큰 도움이 될 수 있을 것으로 보이며, GPT-3에 대한 것들은 향후 리소스가 확보되는대로 다시 비슷하게 쉬운 태스크로 체크해 볼 계획.
  - `(Exploring the Sentence Embedding Cluster)` Sentence Embedding의 경우, Cluster 수에 따라서 어느 정도 유의미한 차이가 보인다고 판단되며, 몇가지 추가 실험을 더 수행해볼 수 있음.
  - `(Hypothesis Testing)`가정하였던대로 유사도 클러스터를 초기값으로 갖고 있을경우, 더 빠르게 학습할 수 있음을 확인하였으며, 결론적으로 Sentence Embedding으로부터 얻은 아주 작은 양의 임팩트 있는 클러스터 정보(high quality and low volume of information)만으로 모델 학습 속도에 영향을 줄 수 있다는 것을 확인함.
      - 이는 multilingual-BERT인 SBERT의 토큰 임베딩 값을 기반으로 클러스터링 정보를 생성하고, KO-BERT가 fine-tuning 과정에서 SBERT의 아웃풋을 참조하므로, 일종의 지식 증류의 성격을 띈다고 볼 수 있으며, 학습 속도면에서 결과 차이가 있음을 확인함.
      - 이러한 증류는 적은 양의 데이터를 학습시키는 태스크에서 큰 실익이 없을 수 있으나, 클러스터 정보를 인풋으로 넣는 것이 더 크고 무거운 모델을 빠르게 학습시키도록 기여할 수 있다는 것을 보여줌.
      - 다른 모델로부터 전달되는 레이블 값은 모델이 참고하기에 충분히 옳은 정보만으로 구성되어 있어야한다는 것을 확인함.
        - `EXP8` ~ `EXP11`이 이러한 클러스터 정보의 질에 따른 학습속도 및 학습 결과를 확인하기 위해 구성되었으며, 8개의 클래스 분류기 생성시에 8개의 클러스터 정보보다는 16개정도 충분히 옳은 클러스터 정보만을 갖고 있도록 구성하는 것이 좋음을 확인함.로 토큰의 영향도를 시각화해서 SBERT로 준 Cluster가 어느 정도의 영향이 있었는지 확인할 계획.

<br>

### `MODE1 & MODE3` 
  - If the sentence length is sufficiently secured, the title of the article is just information that is a repetition of important keywords in the article. Thus it showed a similar accuracy whether the title was added or not. The accuracy after the first epoch can be ignored as the result of random initialization.
  - The title of the article implies the connotation of very well-refined information. Therefore, it is possible to consider giving a little more weight to the title. However, considering the complexity, it is not worth that much.


### `MODE2` 
  - The BERT model has strengths in semantic inference through a slightly wider context and self-emphasis in the context than existing natural language processing models. Therefore, it shows that it is very difficult to classify the topic with only the use of article titles(limited length).
  - Nevertheless, it was surprising that the title of the article alone could be inferred with 82% accuracy just by fine-tuning the BERT model.
  - If I could infer using GPT-3, the result would be much better in the same condition.


### `MODE4 & MODE5` 
Mode 4 and Mode 5 are designed to create and experiment with a kind of distilled BERT model.
  - This experiment is designed *to see if it can be fine-tuned a bit faster based on the clustering information generated by SBERT.*
  - Basic Architecture of BERT models is Next token prediction, hence it is performed to compare the difference between giving information about BERT-based clustering information to the initial hidden layer and just putting information about clustering in the sentence.
  - K-means clustering is performed through the euclidean distance of the latent vector generated from the multilingual-BERT model [5].
  - Since two pretrained BERT model weights were used in the entire pipeline, only information was distilled and embedded when it was determined to be significant to prevent model confusion.
<br>
<br>
  
### Distil BERT에 관련된 실험인 `EXP2 & EXP9`에 대한 고찰   
  - 위 `Embedding Token Visuallization` - 1의 연장선으로 EXP2와 EXP9의 결과를 정리하고 시각화함.
  
  <p align="center">
    <img src="./data/imgs/Precision2and9.png" width="300"> <BR> 
  </p>

  *Fig . 실험2와 실험3의 Precision 비교*


  |   exp | metric    |   culture |    economy |   entertain |   international |        it |   politics |    society |      sport |   accuracy |   macro avg |   weighted avg |
  |------:|:----------|----------:|-----------:|------------:|----------------:|----------:|-----------:|-----------:|-----------:|-----------:|------------:|---------------:|
  |     2 | f1-score  |  0.714286 |   0.801802 |    0.911243 |        0.878049 |  0.769231 |   0.8739   |   0.874791 |   0.975000    |   0.877919 |    0.849788 |       0.877036 |
  |     9 | f1-score  |  0.774194 |   0.796610  |    0.898204 |        0.904000    |  0.829268 |   0.912387 |   0.862543 |   0.951220  |   0.881104 |    0.866053 |       0.881093 |
  |     2 | precision |  0.769231 |   0.847619 |    0.885057 |        0.885246 |  0.714286 |   0.892216 |   0.861842 |   0.951220  |   0.877919 |    0.85084  |       0.877594 |
  |     9 | precision |  0.750000     |   0.789916 |    0.882353 |        0.896825 |  0.73913  |   0.961783 |   0.874564 |   0.906977 |   0.881104 |    0.850194 |       0.883224 |
  |     2 | recall    |  0.666667 |   0.760684 |    0.939024 |        0.870968 |  0.833333 |   0.856322 |   0.888136 |   1        |   0.877919 |    0.851892 |       0.877919 |
  |     9 | recall    |  0.800000      |   0.803419 |    0.914634 |        0.911290  |  0.944444 |   0.867816 |   0.850847 |   1        |   0.881104 |    0.886556 |       0.881104 |
  |   nan | support   | 15        | 117        |   82        |      124        | 18        | 174        | 295        | 117        |   0.877919 |  942        |     942        |

<br>
<br>
  
# Install Issues [Optional]
- Although my time was dead, I want to save your time.
- In `Google Colab`, some resources can handle version conflict, but the others failed to handle this issue. So you need to try 3-4 times while changing resource type(GPU or CPU).<br>
- All issues are caused by legacy `numpy` versions in each package. 
- Do not try to use mxnet acceleration because mxnet is very incomplete. The issues about mxnet acceleration in some cases have not been closed for a year.
- `**CLONE REPOSITORY AND USE THE MODULES AS IT IS - HIGHLY RECOMMENDED**`

<br>
<br>


#### * About `mxnet` and `PyTorch` Framework
- 커뮤니티와 편의성, 이슈 없는 안정성 등 거의 모든 면에서 PyTorch가 이점이 있음.

#### * About `mxnet` install (numpy version conflict)
- If you have a problem in the process of installing `mxnet`, you need to use `python=<3.7.0`.
- Any other options can not solve the numpy version conflict problem while `pip install mxnet`. 
- Except using python=3.7.xx, *EVERYTHING IS USELESS.*
  ```
  error: subprocess-exited-with-error × 

  python setup.py bdist_wheel did not run successfully. │ exit code: 1 
  ╰─> [890 lines of output] Running from numpy source directory. 

  C:\Users\user\AppData\Local\Temp\pip-install-8vepm8z2\numpy_34577a7b4e0f4f25959ef5aa089c5687\numpy\distutils\misc_util.py:476: SyntaxWarning: "is" with a literal. Did you mean "=="? return is_string(s) and ('*' in s or '?' is s) blas_opt_info: blas_mkl_info: No module named 'numpy.distutils._msvccompiler' in numpy.distutils; trying from distutils
  ```
- You may try to install `mxnet` before installing `gluonnlp`.
<br>



#### * About KO-BERT version conflict
- As a result of conflict from above note(About `mxnet` install), KO-BERT still has version conflict.
  ```
  INFO: pip is looking at multiple versions of boto3 to determine which version is compatible with other requirements. This could take a while.
  ...
  To fix this you could try to:
  1. loosen the range of package versions you've specified
  2. remove package versions to allow pip attempt to solve the dependency conflict
  ```
  **Solution of this version conflict**
  - If you DO NOT use mxnet with KO-BERT, then remove mxnet in kO-bert `requirements.txt` or adjust(loosen) version as your environments.
  - If you DO use mxnet with KO-BERT, then just clone the repository and import the module as it is.

<br>

#### * About random initialization in mxnet


  ```
  RuntimeError: Parameter 'bertclassifier2_dense0_weight' has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks.
  ```

- Even when random initialization is not normally performed due to variable management problems, you can observe indicators that seem to be successfully learned in the trainset. 
- However, it was observed that the performance of the trainset continued to drop because it could not be directed to the candidate space of other local minima in the loss space. 
- Therefore, unlike other models, in the case of the Bert model, it is recommended to perform indicator checks in the trainset every iters.
- When using the Apache mxnet framework, carefully organize which layers to lock and which layers to update and in what order. Even when refactoring, initialize and update layers carefully. => to save your time.

<br>

#### * About Tokenizer Error

  ```
  The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
  The tokenizer class you load from this checkpoint is 'XLNetTokenizer'. 
  The class this function is called from is 'KoBERTTokenizer'.
  ```

- First of all, check that num_classes are the same. And make sure classes of torch and mxnet framework are not used interchangeably. Finally, check if there is a conflict or mixed use in the wrapped class as written below. (Ignore the following if it has been resolved.)
- As mentioned above, If it is impossible to download the pretrained KOBERT weights from the aws server, you can download the wrapped weights to the hugging face interface through XLNetTokenizer. [ [3](https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf) ]

<br>

#### * About Korean NLP Models
- Almost all Korean NLP Models have not been updated often. Thus you should try to install lower versions of packages.
- recommendation: `python<=3.7.0` <br>
**Some Tips**
  - IndexError: Target 2 is out of bounds. => num_classes error (please check)
  - broken pipe error => num_workers error, if you use CPU => check your num of threads, else remove args numworkers.
  - If you can not download torch KO-BERT weight with urlib3 or boto3 library error message include 'PROTOCOL_TLS' issue, This is an error related to Amazon aws server download. Thus, => use huggingface interface https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf
  - If you have other questions, please send me an e-mail. *All is Well!! Happy Coding!!*
  - 이슈는 issue를 생성해주시거나, 메일로 문의주시기 바랍니다.

<br>

# References <Br>
[1] GPT fine-tune https://www.philschmid.de/getting-started-setfit <br>
[2] KO-GPT https://github.com/kakaobrain/kogpt <br>
[3] KO-BERT https://sktelecom.github.io/project/kobert <br>
`download` of kobert-base-v1 https://huggingface.co/gogamza/kobart-base-v1 <br>
[4] Sentence-embedding https://github.com/UKPLab/sentence-transformers <br>
[5] Multilingual SBERT https://www.sbert.net/examples/training/multilingual/README.html <br>
[6] NLP gluon BERT documentation https://nlp.gluon.ai/model_zoo/bert/index.html
<br>
참고하지 않았지만, BERT model few shot learning 관련된 논문을 찾다 mode4와 비슷한 컨셉의 논문을 확인함 <br>
[7]
  [Evaluation of Pretrained BERT Model by Using Sentence Clustering](https://aclanthology.org/2020.paclic-1.32) (Shibayama et al., PACLIC 2020)
<Br>
[8] T-SNE 관련 https://www.nature.com/articles/s41467-019-13056-x <br>
[9] DistilBERT https://arxiv.org/abs/1910.01108 <br>
[10] An Isotropy Analysis in the Multilingual BERT Embedding Space https://aclanthology.org/2022.findings-acl.103.pdf
<br>
[11] Where are we in embedding spaces? https://arxiv.org/abs/2105.08908 <br>
[12] Visuallization of BERT https://github.com/jessevig/bertviz.git <br>
 - shap https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html
 - captum https://captum.ai/ <Br>
 
[13] Hugging face model weight upload and load
- https://huggingface.co/transformers/v1.0.0/model_doc/overview.html#loading-google-ai-or-openai-pre-trained-weights-or-pytorch-dump
- https://huggingface.co/docs/transformers/main_classes/model
- https://huggingface.co/docs/huggingface_hub/how-to-downstream
 <br><br>

#### Daily Commit Summary <br>
|Date|Description|
|:---:|:---|
|23.01.16|* 자원체크: GPT inference 최소 VRAM 요구 용량(32GB) 부족으로 GPT 사용불가, KO-BERT 사용  <br> - 환경셋업 완료: 로컬 자원과 Colab 병행 <br> - 간단한 Problem Definition and EDA, Data Analysis, BERT Embbeding Visuallization&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
|23.01.17|- 베이스 코드 작성 <br> - 기간 내 수행 가능한 실험 리스트 작성|
|23.01.18|- 실험결과 정리 <br>- 파이프라인 확정|
|23.01.19~|- 최종 제출 및 리팩토링 후 Repository 정리 <br>- Documentation <br> - Recommendation project에 서브 모듈로 사용(t-sne랑 embedding부분 포함)|
|23.02.05~|- Kaggle 클릭, 장바구니, 구매 항목 예측 모델링하며 Colab Pro를 결제했으므로 2월중에 GPT를 통한 학습 실험 업데이트 예정 <br> - XAI, FrontEnd도 추가적으로 수행 예정|

<br>

#### Further Reading...
[1] PyTorch CRF https://github.com/kmkurn/pytorch-crf
