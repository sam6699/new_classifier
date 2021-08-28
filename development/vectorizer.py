from pymystem3 import Mystem
from typing import List, Union
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

mapping = {'A': 'ADJ', 'ADV': 'ADV', 'ADVPRO': 'ADV', 'ANUM': 'ADJ', 'APRO': 'DET', 'COM': 'ADJ', 'CONJ': 'SCONJ',
           'INTJ': 'INTJ', 'NONLEX': 'X', 'NUM': 'NUM', 'PART': 'PART', 'PR': 'ADP', 'S': 'NOUN', 'SPRO': 'PRON', 'UNKN': 'X', 'V': 'VERB'}


class Vectorizer:
    def __init__(self,model_path) -> None:
        self.stemmer = Mystem()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            model_path, binary=True)
    def preprocess(self,text: str)-> List[str]:
        processed = self.stemmer.analyze(text)
        tagged = []
        for w in processed:
            try:
                lemma = w["analysis"][0]["lex"].lower().strip()
                pos = w["analysis"][0]["gr"].split(',')[0]
                pos = pos.split('=')[0].strip()
                if pos in mapping:
                    # здесь мы конвертируем тэги
                    tagged.append(lemma + '_' + mapping[pos])
                else:
                    # на случай, если попадется тэг, которого нет в маппинге
                    tagged.append(lemma + '_X')
            except (KeyError, IndexError):
                continue  # я здесь пропускаю знаки препинания, но вы можете поступить по-другому
        return tagged
    
    def vectorize(self, text: str) -> Union[List[float],List]:
        words = [word for word in self.preprocess(text) if word in self.model]
        if len(words) >= 1:
            return np.mean(self.model[words], axis=0)
        else:
            return np.asarray([0]*300)
    
    def compare(self, textA: List[str], textB: List, similarity=0.8) -> float:
        vecA = [self.vectorize(text) for text in textA]
        vecB = [self.vectorize(text) for text in textB]

        p = cosine_similarity(vecA, vecB)
        idx = np.where((p > similarity) & (p < 1))
        results = defaultdict(list)
        for i,j in zip(idx[0],idx[1]):
            results[textA[i]].append(textB[j])
        return results

    
    def get_similar_tags(self, textA: List[str], textB: List, similarity=0.8) -> float:
        vecA = [self.vectorize(text) for text in textA]
        vecB = [self.vectorize(text) for text in textB]

        p = cosine_similarity(vecA,vecB)
        idx = np.where((p > similarity) & (p < 1))
        result = set()
        for j in idx[1]:
            result.add(textB[j])
        return result

