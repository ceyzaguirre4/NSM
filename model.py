# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from random import randint
from itertools import permutations


EMBD_DIM = 7
OUT_DIM = 4
BATCH = 10
N = 3


# dummy function
def to_glove(token):
    return torch.rand(EMBD_DIM)


class NSM(nn.Module):
    def __init__(self):
        super().__init__()

        self.W = torch.eye(EMBD_DIM, requires_grad=True)
        self.property_W = torch.stack([torch.eye(EMBD_DIM, requires_grad=True) for _ in range(L + 1)], dim=0)
        self.W_L_plus_1 = torch.eye(EMBD_DIM, requires_grad=True)
        self.W_r = nn.Linear(EMBD_DIM, 1, bias=False)
        self.W_s = nn.Linear(EMBD_DIM, 1, bias=False)

        # encoder is lstm (TODO: one direction?)
        self.encoder_lstm = nn.LSTM(input_size=EMBD_DIM, hidden_size=EMBD_DIM, batch_first=True, bidirectional=False)
        # recurrent decoder (TODO: LSTM?, we know nothing of decoder)
        self.decoder_lstm = nn.LSTM(input_size=EMBD_DIM, hidden_size=EMBD_DIM, batch_first=True, bidirectional=False)

        # final classifier (TODO: hidden dims ???)
        self.classifier = nn.Sequential(
            nn.Linear(2*EMBD_DIM, 2*EMBD_DIM), 
            nn.ELU(), 
            nn.Linear(2*EMBD_DIM, OUT_DIM))

    def forward(self, questions, C, D, E, S, adjacency_mask):
        #####################################
        # Reasoning Instructions
        #####################################

        # embedded questions, shape [batch, len_question, embd_dim]
        embd_questions = torch.stack([
            torch.stack([to_glove(word) for word in question])
            for question in questions
        ])

        # Compare each question word with concept vocabulary including wildcard c'
        # TODO: check if can move transpose to definition
        P_i = torch.softmax(torch.bmm(
            torch.bmm(
                embd_questions,
                self.W.expand(BATCH, EMBD_DIM, EMBD_DIM)
            ),
            C.expand(BATCH, -1, EMBD_DIM).transpose(1,2)
        ), dim=2)

        # weighted sum over C, but using w_i (the word) instead of c' (wildcard) 
        # (if it does not match any of the concepts closely enough--> use w_i)
        V = (P_i[:, :, -1]).unsqueeze(2) * embd_questions + torch.bmm(
            P_i[:, :, :-1], C[:-1, :].expand(BATCH, -1, EMBD_DIM))

        # run encoder on normalized sequence
        _, encoder_hidden = self.encoder_lstm(V)
        (q, _) = encoder_hidden
        q = q.view(BATCH, 1, EMBD_DIM)
        
        # run decoder
        h, _ = self.decoder_lstm(q.expand(BATCH, N+1, EMBD_DIM), encoder_hidden)

        # obtain r (reasoning instructions) by expressing each h_i as a pondered sum of elements in V (projected question words)
        r = torch.bmm(torch.softmax(torch.bmm(h, V.transpose(1, 2)), dim=2), V)

        #####################################
        # Model Simulation
        #####################################

        # initial p_0 is uniform over all states
        p_i = torch.ones(BATCH, len(nodes)) / len(nodes)

        for i in range(N):
            # r_i is the appropiate reasoning instruction for the ith step
            r_i = r[:,i,:]

            R_i = F.softmax(torch.bmm(
                D.expand(BATCH, -1, EMBD_DIM),
                r_i.unsqueeze(2)
            ), dim=1).squeeze(2)

            # r_i_prime is "degree to which that reasoning instruction is concerned with semantic relations"
            r_i_prime = R_i[:,-1].unsqueeze(1)
            property_R_i = R_i[:,:-1]

            # bilinear proyecctions (one for each property) initialized to identity.
            𝛾_i_s = F.elu(torch.sum(
                torch.mul(
                    property_R_i.view(BATCH, -1, 1, 1),
                    torch.mul(
                        torch.matmul(
                            S.transpose(2,1), 
                            self.property_W
                        ), r_i.view(BATCH, 1, 1, EMBD_DIM)
                    )
                ), dim=1
            ))


            # bilinear proyecction 
            𝛾_i_e = F.elu(
                torch.mul(
                    torch.bmm(
                        E.view(BATCH, -1, EMBD_DIM), 
                        self.W_L_plus_1.expand(BATCH, EMBD_DIM, EMBD_DIM)
                    ), r_i.unsqueeze(1))
            ).view(BATCH, len(nodes), len(nodes), EMBD_DIM)


            # update state probabilities (conected to node via relevant relation)
            p_i_r = F.softmax(
                self.W_r(
                    torch.sum(
                        torch.mul(
                            𝛾_i_e,
                            p_i.view(BATCH, -1, 1, 1)
                        ), dim=1)
                ).squeeze(2), dim=1)

            # update state probabilities (property lookup)
            p_i_s = F.softmax(self.W_s(𝛾_i_s).squeeze(2), dim=1)

            p_i = r_i_prime * p_i_r + (1 - r_i_prime) * p_i_s

        #####################################
        # Final Classifier
        #####################################

        # Sumarize final NSM state
        r_N = r[:,N,:]
        property_R_N = F.softmax(
            torch.bmm(
                D.expand(BATCH, -1, EMBD_DIM),
                r_N.unsqueeze(2)
        ), dim=1).squeeze(2)[:,:-1]

        # equivalent to:torch.sum(p_i.unsqueeze(2) * torch.sum(property_R_N.view(10, 1, 3, 1) * S, dim=2), dim=1)
        m = torch.bmm(
            p_i.unsqueeze(1),
            torch.sum(property_R_N.view(BATCH, 1, L+1, 1) * S, dim=2)
        )

        pre_logits = self.classifier(torch.cat([m, q], dim=2).squeeze(1))

        return pre_logits

if __name__ == "__main__":
    #####################################
    # Dummy Data
    #####################################
    property_types = ['color', 'material']
    property_concepts = {
        'color': ['red', 'green', 'blue'],
        'material': ['cloth', 'rubber']
    }
    state_identities = ['cat', 'shirt']
    relationships = ['holding', 'behind']

    L = len(property_types)

    # we add identity and relations in idx 0 and L+1 respectively (TODO: with those names?)
    property_types = ['identity'] + property_types
    property_types += ['relations']
    property_concepts['identity'] = state_identities
    property_concepts['relations'] = relationships

    D = torch.stack([to_glove(property_type) for property_type in property_types])

    # each property has a series of concepts asociated
    # ordered_C is separated by property, C includes all concepts.
    ordered_C = [
        torch.stack([to_glove(concept) for concept in property_concepts[property]])
        for property in property_types
    ]
    C = torch.cat(ordered_C, dim=0)

    # we add c' for non structural words (@ the end)
    # TODO: c' initialization?
    c_prime = torch.rand(1, EMBD_DIM, requires_grad=True)
    C = torch.cat([C, c_prime], dim=0)

    nodes = ['kitten', 'person', 'shirt']

    relations = {
        ('person', 'shirt'): 'wear',
        ('person', 'kitten'): 'holding',
        ('kitten', 'shirt'): 'bite'
    }

    # for simplicity: random state initialization of properties s_j (TODO)
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

    # the tokenized question w/o punctuation
    questions = [['what', 'color', 'is', 'the', 'cat'] for _ in range(BATCH)]

    #####################################
    # Model
    #####################################
    nsm = NSM()
    output = nsm(questions, C, D, E, S, adjacency_mask)

    print(output.shape)
