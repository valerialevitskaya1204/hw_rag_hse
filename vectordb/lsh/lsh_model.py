# your code here

#сделала чтобы возвращался id text в dataset, 
#буду брать самый релеватный и передавать 
#поле Ingredients оттуда в модель

import numpy as np 
from collections import defaultdict 
import time 
import os
import pickle
from tqdm import tqdm

class SimHash: 
    def __init__(self, hash_size, input_dim): 
        self.hash_size = hash_size 
        self.random_projection = np.random.normal(0, 1, size=(hash_size, input_dim)) 
        
    def compute(self, vector): 
        proj = np.dot(self.random_projection, vector) 
        return ''.join('1' if x > 0 else '0' for x in proj)
        
class LSHash:
    def __init__(self, hash_size, input_dim, num_tables):
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_tables = num_tables

        self.hash_functions = [SimHash(hash_size, input_dim) for _ in range(num_tables)]
        self.tables = [defaultdict(list) for _ in range(num_tables)]

        self.vectors = []
        self.texts = []
        self.dataset_ids = []   

    def add(self, vector, text, dataset_id):
        idx = len(self.vectors)

        self.vectors.append(vector)
        self.texts.append(text)
        self.dataset_ids.append(dataset_id)   

        for i in range(self.num_tables):
            h = self.hash_functions[i].compute(vector)
            self.tables[i][h].append(idx)

    def query(self, vector, top_k=5):
        candidates = set()

        for i in range(self.num_tables):
            h = self.hash_functions[i].compute(vector)
            candidates.update(self.tables[i][h])

        if not candidates:
            return []

        cand = np.array(list(candidates))
        cand_vecs = np.array([self.vectors[i] for i in cand])

        dists = np.linalg.norm(cand_vecs - vector, axis=1)
        top_idx = np.argsort(dists)[:top_k]

        results = []
        for idx in top_idx:
            real_index = cand[idx]
            results.append((
                self.dataset_ids[real_index],   
                real_index,                     
                dists[idx],
                self.texts[real_index],
            ))

        return results

        #util методы
    def export_index(self, filepath, lsh):
        try:
            if os.path.isdir(filepath):
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"lsh_index_{timestamp}.pkl"
                filepath = os.path.join(filepath, filename)
    
            if not filepath.endswith('.pkl'):
                filepath += '.pkl'
            
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
       
            index_data = {
                'vectors': self.vectors,
                'texts': self.texts,
                'tables': self.tables,
                'dataset_ids': self.dataset_ids,
                'hash_size': self.hash_size,
                'input_dim': self.input_dim,
                'num_tables': self.num_tables,
                'hash_functions': self.hash_functions
            }
            
        except Exception as e:
            print(f"Ошибка: {e}")
            raise


    @classmethod
    def import_index(cls, filepath):
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            if not filepath.endswith('.pkl'):
                filepath += '.pkl'
            
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            lsh = cls(
                hash_size=index_data['hash_size'],
                input_dim=index_data['input_dim'],
                num_tables=index_data['num_tables']
            )
            lsh.vectors = index_data['vectors']
            lsh.texts = index_data['texts']
            lsh.dataset_ids = index_data['dataset_ids']
            lsh.tables = index_data['tables']
            lsh.hash_functions = index_data['hash_functions']
            
            
        except Exception as e:
            print(f"Ошибка: {e}")
            raise


def add_to_vectordb(lshash, dataset, rag_splitter, embed_model):
    chunk_index = 0
    for i in tqdm(range(len(dataset))):
    # for i in tqdm(range(100)):
        text = dataset[i]['text']

        chunks = rag_splitter.split(text)
        
        for chunk_text in chunks:
            embedding = embed_model.encode(chunk_text)
            
            lshash.add(embedding, chunk_text, i)
            chunk_index += 1

    print(f"Добавлено чанков: {chunk_index}")