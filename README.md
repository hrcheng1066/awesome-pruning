# awesome-pruning

## Table of Contents

- [Type of Pruning](#type-of-pruning)

- [Pruning Before Training](#pruning-before-training)

### Type of Pruning

| Type        | `F`            | `C`             | `W`            | `R`        | `Other`     |
|:----------- |:--------------:|:---------------:|:--------------:|:----------:|:-----------:|
| Explanation | Filter pruning | Channel pruning | Weight pruning | Root level | other types |

### Pruning Before Training
| Title   | Venue | Type    | Code | Year |
|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|
| [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)| ICLR| `W`&`R` | [TensorFLow(Author)](https://github.com/namhoonlee/snip-public) | 2019 |
| [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307)| ICLR **(Spotlight)** | `W` | [TensorFLow(Author)](https://github.com/namhoonlee/spp-public) | 2020 |
| [Picking Winning Tickets before Training by Preserving Gradient Flow](https://openreview.net/pdf?id=SkgsACVKPH))| ICLR | `W` | [PyTorch(Author)](https://github.com/alecwangcq/GraSP) | 2020 |      
| [Pruning from Scratch](http://arxiv.org/abs/1909.12579) | AAAI | `C` | [PyTorch(Author)](https://github.com/frankwang345/pruning-from-scratch) | 2020 |    
