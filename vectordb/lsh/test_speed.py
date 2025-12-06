import numpy as np
from vectordb.lsh.lsh_model import LSHash
import time

num_tables = 15
hash_size = 25

N = 1000
dim = 2

data = np.random.randn(N, dim)

def brute_force(query):
    d = np.linalg.norm(data - query, axis=1)
    return np.argmin(d)


results = []

lsh = LSHash(hash_size=hash_size, input_dim=dim, num_tables=num_tables)
for i in range(N):
    lsh.add(data[i], "some embed", i)

    
q = np.random.randn(dim)


t0 = time.time()
cand = lsh.query(q)
total_lsh = time.time() - t0


t1 = time.time()
cand = brute_force(q)
total_brute = time.time() - t1

speedup=total_lsh/total_brute
        
print(f"колво таблиц={num_tables}, размер хеша={hash_size}, ускорение={speedup}x")