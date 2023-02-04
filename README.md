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
| 10 | [Prunining via Iterative Ranking of Sensitivity Statics](https://arxiv.org/abs/2006.00896) | arXiv | `W` or `F`/`C` | SNIP-it | [PyTorch(Author)](https://github.com/StijnVerdenius/SNIP-it) | Image Classification | 2020 |
| 11 | [Prunining Neural Networks at Initialization: Why are We Missing the Mark?](https://arxiv.org/abs/2009.08576) | ICLR | `W` | - | - | Image Classification | 2021 |



### Pruning During Training
| Title   | Venue | Type    | Code | APP | Year |
|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|

### Pruning After Training
| Title   | Venue | Type    | Code | APP | Year |
|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|

## Acknowledgements
https://github.com/he-y/awesome-Pruning/  
https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization  
https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression/blob/master/Paper/Pruning.md  



