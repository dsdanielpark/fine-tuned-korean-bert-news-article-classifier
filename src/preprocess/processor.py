import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class NLPdata:
    def __init__(self) -> None:
        pass

    def get_sentence_clean(self, sentence: str, remove_list: list) -> str:
        if remove_list == None:
            remove_list = self.config['remove_list']
        for word in remove_list:
            sentence = sentence.replace(word, " ")

        return sentence

    def _get_label_encoder(self, df: pd.DataFrame, labelCol: str, reverse: bool) -> pd.DataFrame:
        # class_dict = {v: k for k, v in enumerate(list(df[labelCol].unique()))} 
        class_dict = {0: 'international', 1: 'economy', 2: 'society', 3: 'sport', 4: 'it', 5: 'politics', 6: 'entertain', 7: 'culture'}
        if reverse:
            class_dict = {'international': 0, 'economy': 1, 'society': 2, 'sport': 3, 'it': 4, 'politics': 5, 'entertain': 6, 'culture': 7}
        df[labelCol] = [class_dict[x] for x in df[labelCol]]
        
        return df

    def make_TSVD_list(self, df: pd.DataFrame, bodyCol: str, labelCol: str) -> list:
        corpus = []
        for ques, label in zip(df[bodyCol], df[labelCol]):
            data = []   
            data.append(ques)
            data.append(str(label))
            corpus.append(data)

        return corpus

    def get_corpus_cluster_df(self, corpus: list, num_clusters: int) -> list:
        embedder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
        corpus_embeddings = embedder.encode(corpus)
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        clustered_corpus = [[] for i in range(num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_corpus[cluster_id].append(corpus[sentence_id])
        clustered_list = [[str(i)+'cluster'+x for x in clustered_corpus[i]] for i in range(num_clusters)]

        return sum(clustered_list, [])

    def get_df_cluster(self, df: pd.DataFrame, num_clusters: int) -> pd.DataFrame:
        df.reset_index(inplace=True)
        df['cleanBody'] = [f'{i}index{b}' for i, b in enumerate(df['cleanBody'])]
        corpus = df['cleanBody'].to_list()
        clusted_corpus = self.get_corpus_cluster_df(corpus, num_clusters)
        
        df_cluster = pd.DataFrame(clusted_corpus, columns=['cleanBody'])
        df_cluster['cluster'] = [int(x.split('cluster')[0]) for x in df_cluster['cleanBody']]
        df_cluster['cleanBody'] = [x.split('cluster')[1] for x in df_cluster['cleanBody']]
        df_cluster['index'] = [int(x.split('index')[0]) for x in df_cluster['cleanBody']]
        df_cluster['cleanBody'] = [x.split('index')[1] for x in df_cluster['cleanBody']]
        df_cluster.drop(['cleanBody'], axis = 1, inplace=True)

        df['index'] = [int(x.split('index')[0]) for x in df['cleanBody']]
        df['cleanBody'] = [x.split('index')[1] for x in df['cleanBody']]
        df_clustered = df.join(df_cluster.set_index('index'), on='index')

        return df_clustered

    def TSVDdataset(self, file_path: str, bodyCol: str, labelCol:str, mode: str, num_cluster=None) -> list:
        df = pd.read_csv(file_path)
        df = self._get_label_encoder(df, labelCol, True)
        if mode == 'mode1':
            TVSD_dataset = self.make_TSVD_list(df, bodyCol, labelCol)
        elif mode == 'mode2':
            TVSD_dataset = self.make_TSVD_list(df, bodyCol, labelCol)
        elif mode == 'mode3':
            df['cleanBody'] = [f'기사제목:: {t}      기사본문: {b}' for t, b in zip(df['title'], df['cleanBody'])]
            TVSD_dataset = self.make_TSVD_list(df, bodyCol, labelCol)
        elif mode == 'mode4':
            df_clustered = self.get_df_cluster(df, num_cluster)
            df_clustered['cleanBody'] = [f'기사제목: {t}     기사본문: {b}' for t, b in zip(df_clustered['title'], df_clustered['cleanBody'])]
            TVSD_dataset = self.make_TSVD_list(df_clustered, bodyCol, labelCol)
        return TVSD_dataset
