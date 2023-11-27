# HRViT
An unofficial Keras implementation of Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation (HRViT). HRViT leverages careful design and an efficient cross-shaped attention mechanism to minimize FLOPS while maximizing performance.

The original paper can be found here: https://arxiv.org/pdf/2111.01236v2.pdf

The official pytorch implementation: https://github.com/facebookresearch/HRViT

### Typical Usage
```
from hrvit import HRViT_b1

model = HRViT_b1()
...
model.fit(X, Y)
...
y = model.predict(x)
```

### Running Tests
From the project root, run the following command:

>`pytest ./test/.`
