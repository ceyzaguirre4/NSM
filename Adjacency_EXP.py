# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Dependencies

# + {"hidden": true}
import torch
import torch.nn as nn

# + {"hidden": true}
import easydict
from random import randint
from itertools import permutations

# + {"heading_collapsed": true, "cell_type": "markdown"}
# ## Hyperparams

# + {"hidden": true}
EMBD_DIM = 7
OUT_DIM = 4
BATCH = 10
N = 3


# -

# # Concept Vocabulary

def to_glove(token):
    return torch.rand(EMBD_DIM)


# ### Dummy data

# +
property_types = ['color', 'material']

property_concepts = {
    'color': ['red', 'green', 'blue'],
    'material': ['cloth', 'rubber']
}

state_identities = ['cat', 'shirt']

relationships = ['holding', 'behind']
# -

# ## Preparation

# +
# each property has a type
L = len(property_types)

# we add identity and relations in idx 0 and L+1 respectively (TODO: with those names?)
property_types = ['identity'] + property_types
property_types += ['relations']
property_concepts['identity'] = state_identities
property_concepts['relations'] = relationships

D = torch.stack([to_glove(property_type) for property_type in property_types])

# +
# each property has a series of concepts asociated
# ordered_C is separated by property, C includes all concepts.
ordered_C = [
    torch.stack([to_glove(concept) for concept in property_concepts[property]])
    for property in property_types
]
C = torch.cat(ordered_C, dim=0)

# we add c' for non structural words (@ idx -1)
# TODO: c' initialization?
c_prime = torch.rand(1, EMBD_DIM, requires_grad=True)
C = torch.cat([C, c_prime], dim=0)
# -

# # Scene Graph

# ### Dummy data

# +
nodes = ['kitten', 'person', 'shirt']

relations = {
    ('person', 'shirt'): 'wear',
    ('person', 'kitten'): 'holding',
    ('kitten', 'shirt'): 'bite'
}
# -

# ## Preparation

# for simplicity: random state initialization of properties (TODO)
S = torch.rand(BATCH, len(nodes), L+1, EMBD_DIM)

# build adjacency matrix (TODO: now all graphs are same)
# the edge features e' are inserted into an adjacency matrix for eficiency
adjacency_mask = torch.zeros(BATCH, len(nodes), len(nodes))
E = torch.zeros(BATCH, len(nodes), len(nodes), EMBD_DIM)
for idx_pair in permutations(range(len(nodes)), 2):
    pair = tuple(nodes[idx] for idx in idx_pair)
    if pair in relations:
        E[:,idx_pair[0],idx_pair[1]] = torch.rand(EMBD_DIM)   # (TODO)
        adjacency_mask[:,idx_pair[0],idx_pair[1]] = 1

# +
# alternatively we can use hybrid tensors to reduce memory and computation overhead
indices = []
values = []
for idx_pair in permutations(range(len(nodes)), 2):
    pair = tuple(nodes[idx] for idx in idx_pair)
    if pair in relations:
        indices.append(idx_pair)
        values.append(torch.rand(EMBD_DIM))

sparse_adj = torch.sparse.FloatTensor(
    torch.LongTensor(indices).t(), 
    torch.stack(values),
    (len(nodes), len(nodes), EMBD_DIM)
)
# -

sparse_adj.shape

sparse_adj.coalesce().indices().shape

sparse_adj.coalesce().values().shape

3*3*7 > (2*3) + (3*7)

sparse_adj.resizeAs_()


