# NSM

Neural State Machine implemented in [PyTorch](http://pytorch.org/) as presented [here](https://arxiv.org/abs/1907.03950).
This is the first code implementation of the model and is based on V1 of the paper on [arxiv](https://arxiv.org).
As can be expected, the code is incomplete and makes several assumptions where the paper wasn't clear enough.
For the time being this code is not ready to run and several steps are needed for it to train on any VQA dataset.
Principal among these is the need for a functioning [graph-rcnn](https://arxiv.org/pdf/1808.00191.pdf) to generate the scene graphs.

In the meantime, I hope the code helps readers understand better the paper, and I'm open to any colaborators who wish to help with features or efficiency.
