# awesome-pruning

## Table of Contents

- [Type of Pruning](#type-of-pruning)

- [Pruning Before Training](#pruning-before-training)
- [Pruning During Training](#pruning-during-training)
- [Pruning After Training](#pruning-after-training)

### Type of Pruning

| Type        | `F`            | `C`             | `W`            | `P`        | `Other`     |
|:----------- |:--------------:|:---------------:|:--------------:|:----------:|:-----------:|
| Explanation | Filter pruning | Channel pruning | Weight pruning | Pioneer | other types |

### Pruning Before Training
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)| ICLR| `W`&`P` | SNIP | [TensorFLow(Author)](https://github.com/namhoonlee/snip-public) | Image Classification | 2019 |
| 02 | [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307)| ICLR **(Spotlight)** | `W` | - | [TensorFLow(Author)](https://github.com/namhoonlee/spp-public) | Image Classification | 2020 |
| 03 | [Picking Winning Tickets before Training by Preserving Gradient Flow](https://openreview.net/pdf?id=SkgsACVKPH))| ICLR | `W` | GraSP | [PyTorch(Author)](https://github.com/alecwangcq/GraSP) | Image Classification | 2020 |      
| 04 | [Pruning from Scratch](http://arxiv.org/abs/1909.12579) | AAAI | `C` | - | [PyTorch(Author)](https://github.com/frankwang345/pruning-from-scratch) | Image Classification | 2020 |
| 05 | [Pruning neural networks without any data by iteratively conserving synaptic flow](https://arxiv.org/abs/2006.05467)| NeurIPS | `W` | SynFlow | [PyTorch(Author)](https://github.com/ganguli-lab/Synaptic-Flow) | Image Classification | 2020 |
| 06 | [A Unified Paths Perspective for Pruning at Initialization](https://arxiv.org/abs/2101.10552)| arXiv | `W` | - | - | Image Classification | 2021 |
| 07 | [Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot?](https://papers.nips.cc/paper/2021/hash/6a130f1dc6f0c829f874e92e5458dced-Abstract.html) | NeurIPS | `W` | Smart-Ratios | [PyTorch(Author)](https://github.com/boone891214/sanity-check-LTH) | Image Classification | 2021 |
| 08 | [Progressive Skeletonization: Trimming More Fat from a network at initialization](https://arxiv.org/abs/2006.09081) | ICLR | `W` | FORCE | [PyTorch(Author)](https://github.com/naver/force) | Image Classification | 2021 |
| 09 | [Robust Pruning at Initialization](https://openreview.net/forum?id=vXj_ucZQ4hA) | ICLR | `W` | SPB | - | Image Classification | 2021 |
| 10 | [Prunining via Iterative Ranking of Sensitivity Statics](https://arxiv.org/abs/2006.00896) | arXiv | `WFC` | SNIP-it | [PyTorch(Author)](https://github.com/StijnVerdenius/SNIP-it) | Image Classification | 2020 |
| 11 | [Prunining Neural Networks at Initialization: Why are We Missing the Mark?](https://arxiv.org/abs/2009.08576) | ICLR | `W` | - | - | Image Classification | 2021 |
| 12 | [Why is Pruning at Initialization Immune to Reinitializating and Shuffling?](https://arxiv.org/abs/2107.01808)) | arXiv | `W` | - | - | Image Classification | 2021 |
| 13 | [Prospect Pruning: Finding Trainable Weights at Initialization using Meta-Gradients](https://openreview.net/forum?id=AIgn9uwfcD1)| ICLR | `WF`| ProsPr | [PyTorch(Author)](https://github.com/mil-ad/prospr) | Image Classification | 2022 |
| 14 | [Dual Lottery Ticket Hypothesis](https://openreview.net/forum?id=fOsN52jn25l) | ICLR | `W` | RST | [PyTorch(Author)](https://github.com/yueb17/DLTH) | Image Classification | 2022 |
| 15 | [Recent Advances on Neural Network Pruning at Initialization](https://arxiv.org/abs/2103.06460)| IJCAI | `W`| - | [PyTorch(Author)](https://github.com/mingsun-tse/smile-pruning) | Image Classification | 2022 |





### Pruning During Training
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Dynamic Sparse Training: Find Effective Sparse Network from Scratch with Trainable Masked Layers](https://arxiv.org/abs/2005.06870)| ICLR | `W`| DST | [PyTorch(Author)](https://github.com/junjieliu2910/DynamicSparseTraining) | Image Classification | 2020 |
| 02 | [Learning Structured Sparsity in Deep Neural Networks](https://proceedings.neurips.cc/paper/2016/file/41bfd20a38bb1b0bec75acf0845530a7-Paper.pdf)| NIPS | `FC`| SSL | [Caffe(Author)](https://github.com/wenwei202/caffe/tree/scnn) | Image Classification | 2016 |
| 03 | [Learning Efficient Convolutional Networks through Networks Slimming](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)| ICCV | `C`| Slimming | [Lua(Author)](https://github.com/liuzhuang13/slimming) | Image Classification | 2017 |
| 04 | [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124) | ICLR | `F` | - | [TensorFlow(Author)](https://github.com/bobye/batchnorm_prune) [PyTorch(3rd)](https://github.com/jack-willturner/batchnorm-pruning) | Image Classification&Segmentation | 2018 |
| 05 | [Data-Driven Sparse Structure Selection for Deep Neural Networks](https://arxiv.org/abs/1707.01213) | ECCV | `F` | SSS | [MXNet(Author)](https://github.com/TuSimple/sparse-structure-selection) | 2018 |




### Pruning After Training
| Title   | Venue | Type    | Code | APP | Year |
|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|

## Acknowledgements
https://github.com/he-y/awesome-Pruning/  
https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization  
https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression/blob/master/Paper/Pruning.md  



