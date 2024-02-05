#!/usr/bin/env python3

from gensim.models.doc2vec import Doc2Vec


model_path = 'Comparisons/experiments/alpha/bin/doc2vecR.200.30.20.5.1550908281.eAp.trained'

klee = Doc2Vec.load(model_path)

if 'numpy' in klee.wv.vocab: 
    vec = klee.wv.get_vector('numpy')
    print(vec)
    print(len(vec))  # 200
