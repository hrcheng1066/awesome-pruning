# awesome-pruning
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green)[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
## Table of Contents
- [0. Overview](#0-overview)
- [1. When to prune](#1-when-to-prune)
  <!-- - [Variational Gap Optimization](#Variational-Gap-Optimization) -->
  <!-- - [Dimension Deduction](#Dimension-Deduction) -->
  - [1.1 Static Pruning](#11-static-pruning)
    - [1.1.1 Pruning Before Training](#111-pruning-before-training)
      - [1.1.1.1 Pruning CNNs](#1111-pruning-cnns)
        - [2024](#pruning-before-training-cnns-2024)  [2023](#pruning-before-training-cnns-2023)  [2022](#pruning-before-training-cnns-2022) [2021](#pruning-before-training-cnns-2021) [2020](#pruning-before-training-cnns-2020) [2019](#pruning-before-training-cnns-2019)
    - [1.1.2 Pruning During Training](#112-pruning-during-training)
      - [1.1.2.1 Pruning CNNs](#1121-pruning-cnns)
        - [2024](#pruning-during-training-cnns-2024)  [2023](#pruning-during-training-cnns-2023)  [2022](#pruning-during-training-cnns-2022)  [2021](#pruning-during-training-cnns-2021)  [2020](#pruning-during-training-cnns-2020)  [2019](#pruning-during-training-cnns-2019)   [2018 and earlier](#pruning-during-training-cnns-2018-and-earlier)
      - [1.1.2.2 Pruning Other Models](#1122-pruning-other-models)
    - [1.1.3 Pruning After Training](#113-pruning-after-training)
      - [1.1.3.1 Pruning CNNs](#1131-pruning-cnns)
        - [2023](#pruning-after-training-cnns-2023)  [2022](#pruning-after-training-cnns-2022)  [2021](#pruning-after-training-cnns-2021)  [2020](#pruning-after-training-cnns-2020)  [2019](#pruning-after-training-cnns-2019)  [2018](#pruning-after-training-cnns-2018)  [2017 and earlier](#pruning-after-training-cnns-2017-and-earlier)
      - [1.1.3.2 Pruning ViTs](#1132-pruning-vits)
        - [2024](#pruning-after-training-vits-2024)  [2023](#pruning-after-training-vits-2023)  [2022](#pruning-after-training-vits-2022)
      - [1.1.3.3 Pruning BERTs](#1133-pruning-berts)
        - [2023](#pruning-after-training-berts-2023)  [2022](#pruning-after-training-berts-2022)  [2021](#pruning-after-training-berts-2021)  [2020](#pruning-after-training-berts-2020)  [2019](#pruning-after-training-berts-2019)
      - [1.1.3.4 Pruning LLMs](#1134-pruning-llms)
        - [2024](#pruning-after-training-llms-2024)  [2023](#pruning-after-training-llms-2023)  
      - [1.1.3.5 Pruning Diffusion Models](#1135-pruning-diffusion-models)
        - [2023](#pruning-after-training-diffusion-models-2023)
      - [1.1.3.6 Pruning Vision-and-Languages](#1136-pruning-vision-and-languages)
        - [2024](#pruning-after-training-vlms-2024)  [2023](#pruning-after-training-vlms-2023)  [2022](#pruning-after-training-vlms-2022)
      - [1.1.3.7 Pruning Other Models](#1137-pruning-other-models)
      - [1.1.3.8 Post Training](#1138-post-training)
        - [2024](#post-training-2024)  [2023](#post-training-2023)  [2022](#post-training-2022)  [2021](#post-training-2021)    
    - [1.1.4 Pruning In Early Training](#114-pruning-in-early-training)
  - [1.2 Dynamic Pruning](#12-dynamic-pruning) 
- [2. Learning and Pruning](#2-learning-and-pruning)
  - [2.1 Continual learning](#21-continual-learning)
  - [2.2 Contrastive learning](#22-contrastive-learning)
  - [2.3 Federated learning](#23-federated-learning)
- [3. Application](#3-application) 
  - [3.1 Computer Vision](#31-computer-vision)
  - [3.2 Natural Language Processing](#32-natural-language-processing)
  - [3.3 Audio Signal Processing](#33-audio-signal-processing)
- [4. Combination](#4-combination) 
  - [4.1 pruning and quantization](#41-pruning-and-quantization)
  - [4.2 pruning and decomposition](#42-pruning-and-decomposition)
- [5. Survey of Pruning](#5-survey-of-pruning)
  - [2024](#survey-of-pruning-2024)  [2023](#survey-of-pruning-2023)  [2022](#survey-of-pruning-2022)  [2021](#survey-of-pruning-2021)  [2020](#survey-of-pruning-2020)  [2019 and earlier](#survey-of-pruning-2019-and-earlier)   
- [6. Other Works](#6-other-works)
  - [Papers](#papers)
  - [Useful Links](#useful-links)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation) 

## 0. Overview
The repo includes the ongoing updates of representative neural network pruning papers and open-source codes.  
Our paper [_**A Survey on Deep Neural Network Pruning-Taxonomy, Comparison, Analysis, and Recommendations**_] ([Paper Link](https://arxiv.org/abs/2308.06767)), accepted by TPAMI 2024.

**Taxonomy**: In our survey, we provide a comprehensive review of the state-of-the-art in deep neural network pruning, which we categorize along five orthogonal axes: Universal/Specific Speedup, When to Prune, Pruning Criteria, Learn to Prune, and Fusion of Pruning and Other Techniques. 

<p align="center">
  <img src=taxonomy.png width="500">
</p>


## 1. When to Prune
### Type Explanation
| Type        |`L`             | `F`            | `C`             | `N`             | `H`            | `B`             | `M`            | `E`            | `W`          | `P`        | `Other`     |
|:----------- |:--------------:|:--------------:|:---------------:|:---------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:----------:|:-----------:|
| Explanation | Layer pruning  | Filter pruning | Channel pruning |  Neuron pruning |  Head pruning  | Block pruning | Matrix pruning | Embedding pruning | Weight pruning |  Pioneer work   | other types |

### 1.1 Static Pruning

#### 1.1.1 Pruning Before Training
##### 1.1.1.1 Pruning CNNs
###### Pruning Before Training CNNs 2024
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [No Free Prune: Information-Theoretic Barriers to Pruning at Initialization](https://openreview.net/pdf?id=Uzb45nolTb) | ICML | `W` | - | - | Image Classification | 2024 |

###### Pruning Before Training CNNs 2023
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Data-Free Model Pruning at Initialization via Expanders](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Stewart_Data-Free_Model_Pruning_at_Initialization_via_Expanders_CVPRW_2023_paper.pdf) | CVPRW | `W` | RReg | [PyTorch(Author)](https://github.com/SamsungLabs/ExpandersPruning) | Image Classification | 2023 |
| 02 | [Revisiting Pruning as Initialization through the Lens of Ramanujan Graph](https://openreview.net/forum?id=uVcDssQff) | ICLR (TOP 5%) | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/ramanujan-on-pai) | Image Classification | 2023 |
| 03 | [Pruning at Initialization - A Sketching Perspective](https://arxiv.org/abs/2305.17559) | arXiv | `W` | - | - | Image Classification | 2023 |
| 04 | [NTK-SAP: Improving neural network pruning by aligning training dynamics](https://arxiv.org/abs/2304.02840) | ICLR | `W` | NTK-SAP | [PyTorch(Author)](https://github.com/yitewang/ntk-sap) | Image Classification | 2023 |


###### Pruning Before Training CNNs 2022
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Prospect Pruning: Finding Trainable Weights at Initialization using Meta-Gradients](https://openreview.net/forum?id=AIgn9uwfcD1)| ICLR | `WF`| ProsPr | [PyTorch(Author)](https://github.com/mil-ad/prospr) | Image Classification | 2022 |
| 02 | [Dual Lottery Ticket Hypothesis](https://openreview.net/forum?id=fOsN52jn25l) | ICLR | `W` | RST | [PyTorch(Author)](https://github.com/yueb17/DLTH) | Image Classification | 2022 |
| 03 | [Recent Advances on Neural Network Pruning at Initialization](https://arxiv.org/abs/2103.06460)| IJCAI | `W`| - | [PyTorch(Author)](https://github.com/mingsun-tse/smile-pruning) | Image Classification | 2022 |
| 04 | [The Unreasonable Effectiveness of Random Pruning: Return of the Most Naive Baseline for Sparse Training](https://openreview.net/forum?id=VBZJ_3tz-t) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/Random_Pruning) | Image Classification | 2022 |
| 05 | [Structured Pruning is All You Need for Pruning CNNs at Initialization](https://arxiv.org/abs/2203.02549) | arXiv | `C` | PreCropping | - | Image Classification | 2022 |

###### Pruning Before Training CNNs 2021
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Progressive Skeletonization: Trimming More Fat from a network at initialization](https://arxiv.org/abs/2006.09081) | ICLR | `W` | FORCE | [PyTorch(Author)](https://github.com/naver/force) | Image Classification | 2021 |
| 02 | [Robust Pruning at Initialization](https://openreview.net/forum?id=vXj_ucZQ4hA) | ICLR | `W` | SPB | - | Image Classification | 2021 |
| 03 | [A Unified Paths Perspective for Pruning at Initialization](https://arxiv.org/abs/2101.10552)| arXiv | `W` | - | - | Image Classification | 2021 |
| 04 | [Prunining Neural Networks at Initialization: Why are We Missing the Mark?](https://arxiv.org/abs/2009.08576) | ICLR | `W` | - | - | Image Classification | 2021 |
| 05 | [Why is Pruning at Initialization Immune to Reinitializating and Shuffling?](https://arxiv.org/abs/2107.01808)) | arXiv | `W` | - | - | Image Classification | 2021 |
| 06 | [Dense for the Price of Sparse: Improved Performance of Sparsely Initialized Networks via a Subspace Offset](https://arxiv.org/abs/2102.07655)) | ICML | `W` | DCTpS | [PyTorch(Author)](https://github.com/IlanPrice/DCTpS) | Image Classification | 2021 |

###### Pruning Before Training CNNs 2020
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Single Shot Structured Pruning Before Training](https://arxiv.org/abs/2007.00389)| arXiv | `C`| 3SP  | [PyTorch(Author)](https://github.com/fmi-basel/neural-tangent-transfer) | Image Classification | 2020 |
| 02 | [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307)| ICLR **(Spotlight)** | `W` | - | [TensorFLow(Author)](https://github.com/namhoonlee/spp-public) | Image Classification | 2020 |
| 03 | [Picking Winning Tickets before Training by Preserving Gradient Flow](https://openreview.net/pdf?id=SkgsACVKPH))| ICLR | `W` | GraSP | [PyTorch(Author)](https://github.com/alecwangcq/GraSP) | Image Classification | 2020 |      
| 04 | [Pruning from Scratch](http://arxiv.org/abs/1909.12579) | AAAI | `C` | - | [PyTorch(Author)](https://github.com/frankwang345/pruning-from-scratch) | Image Classification | 2020 |
| 05 | [Pruning neural networks without any data by iteratively conserving synaptic flow](https://arxiv.org/abs/2006.05467)| NeurIPS | `W` | SynFlow | [PyTorch(Author)](https://github.com/ganguli-lab/Synaptic-Flow) | Image Classification | 2020 |
| 06 | [Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot](https://proceedings.neurips.cc/paper/2020/file/eae27d77ca20db309e056e3d2dcd7d69-Paper.pdf) | NeurIPS | `W` | Smart-Ratios | [PyTorch(Author)](https://github.com/JingtongSu/sanity-checking-pruning) | Image Classification | 2020 |
| 07 | [Prunining via Iterative Ranking of Sensitivity Statics](https://arxiv.org/abs/2006.00896) | arXiv | `WFC` | SNIP-it | [PyTorch(Author)](https://github.com/StijnVerdenius/SNIP-it) | Image Classification | 2020 |
| 08 | [What’s Hidden in a Randomly Weighted Neural Network?](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ramanujan_Whats_Hidden_in_a_Randomly_Weighted_Neural_Network_CVPR_2020_paper.pdf)| CVPR | `W`| - | [PyTorch(Author)](https://github.com/allenai/hidden-networks) | Image Classification | 2020 |
| 09 | [Finding trainable sparse networks through Neural Tangent Transfer](https://arxiv.org/abs/2006.08228)| ICML | `W`| - | [PyTorch(Author)](https://github.com/fmi-basel/neural-tangent-transfer) | Image Classification | 2020 |

###### Pruning Before Training CNNs 2019
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)| ICLR| `WP` | SNIP | [TensorFLow(Author)](https://github.com/namhoonlee/snip-public) | Image Classification | 2019 |


#### 1.1.2 Pruning During Training
##### 1.1.2.1 Pruning CNNs
###### Pruning During Training CNNs 2024
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Auto-Train-Once: Controller Network Guided Automatic Network Pruning from Scratch](https://arxiv.org/abs/2403.14729) | CVPR | `W` | ATO | [PyTorch(Author)](https://github.com/xidongwu/AutoTrainOnce) | Image Classification | 2024 |

###### Pruning During Training CNNs 2023
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [PDP: Parameter-free Differentiable Pruning is All You Need](https://arxiv.org/pdf/2305.11203) | NeurIPS | `WC` | - | - | Vision&NLP | 2023 |
| 02 | [LAPP: Layer Adaptive Progressive Pruning for Compressing CNNs from Scratch](https://arxiv.org/pdf/2309.14157) | arXiv | `F` | LAPP | - | Image Classification | 2023 |

###### Pruning During Training CNNs 2022
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [SuperTickets: Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning](https://arxiv.org/abs/2207.03677) | ECCV | `W` | SuperTickets | [PyTorch(Author)](https://github.com/GATECH-EIC/SuperTickets) | Image Classification&Object Detection&Human Pose Estimation | 2022 |
| 02 | [Deep ensembling with no overhead for either training or testing: The all-round blessings of dynamic sparsity](https://openreview.net/pdf?id=RLtqs6pzj1-) | ICLR | `W` | FreeTickets |[PyTorch(Anthor)](https://github.com/VITA-Group/FreeTickets) | Image Classification | 2022 |
| 03 | [Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win](https://arxiv.org/abs/2010.03533) | AAAI | `W` | - |[PyTorch(Anthor)](https://github.com/google-research/rigl/tree/master/rigl/rigl_tf2) | Image Classification | 2022 |

###### Pruning During Training CNNs 2021
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Training Adversarially Robust Sparse Networks via Bayesian Connectivity Sampling](http://proceedings.mlr.press/v139/ozdenizci21a/ozdenizci21a.pdf) | ICML | `W` | - |[PyTorch(Anthor)](https://github.com/IGITUGraz/SparseAdversarialTraining) | Adversarial Robustness | 2021 |
| 02 | [Training Neural Networks with Fixed Sparse Masks](https://arxiv.org/abs/2111.09839) | NeurIPS | `W` | - | [PyTorch(Author)]( https://github.com/varunnair18/FISH) | Image Classification | 2021 |
| 03 | [DPFPS: Dynamic and Progressive Filter Pruning for Compressing Convolutional Neural Networks from Scratch](https://ojs.aaai.org/index.php/AAAI/article/view/16351) | AAAI | `C` | DPFPS | [PyTorch(Author)]( https://github.com/taoxvzi/DPFPS) | Image Classification | 2021 |
| 04 | [Sparse Training via Boosting Pruning Plasticity with Neuroregeneration](https://papers.nips.cc/paper/2021/hash/5227b6aaf294f5f027273aebf16015f2-Abstract.html)  | NeurIPS | `WF` | GraNet | [PyTorch(Author)](https://github.com/VITA-Group/GraNet) | Image Classification | 2021 |
| 05 | [Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training](http://proceedings.mlr.press/v139/liu21y/liu21y.pdf) | ICML | `W` | ITOP | [PyTorch(Anthor)](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization) | Image Classification | 2021 |  
| 06 | [Dense for the Price of Sparse: Improved Performance of Sparsely Initialized Networks via a Subspace Offset](http://proceedings.mlr.press/v139/price21a/price21a.pdf) | ICML | `W` | DCTpS | [PyTorch(Anthor)](https://github.com/IlanPrice/DCTpS) | Image Classification | 2021 | 

###### Pruning During Training CNNs 2020
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Pruning Filter in Filter](https://arxiv.org/abs/2009.14410) | NeurIPS | `Other` | SWP | [PyTorch(Author)](https://github.com/fxmeng/Pruning-Filter-in-Filter)    | Image Classification | 2020 |
| 02 | [Dynamic Sparse Training: Find Effective Sparse Network from Scratch with Trainable Masked Layers](https://arxiv.org/abs/2005.06870)| ICLR | `NF`| DST | [PyTorch(Author)](https://github.com/junjieliu2910/DynamicSparseTraining) | Image Classification | 2020 |
| 03 | [DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation](https://arxiv.org/abs/2004.02164) | ECCV | `F` | DSA | [PyTorch(Author)](https://github.com/walkerning/differentiable-sparsity-allocation) | Image Classification | 2020 | 
| 04 | [Dynamic Model Pruning with Feedback](https://openreview.net/forum?id=SJem8lSFwB) | ICLR | `WF` | DPF | [PyTorch(3rd)](https://github.com/INCHEON-CHO/Dynamic_Model_Pruning_with_Feedback) | Image Classification | 2020 |  

###### Pruning During Training CNNs 2019
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250) | CVPR | `F` | FPGM | [PyTorch(Author)](https://github.com/he-y/filter-pruning-geometric-median) | Image Classification | 2019 | 
| 02 | [Compressing Convolutional Neural Networks via Factorized Convolutional Filters](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Compressing_Convolutional_Neural_Networks_via_Factorized_Convolutional_Filters_CVPR_2019_paper.pdf) | CVPR | `F` | FCF | [PyTorch(Author)](https://github.com/IIGROUP/CNN-FCF) | Image Classification | 2019 |
| 03 | [Rigging the Lottery: Making All Tickets Winners](https://arxiv.org/abs/1911.11134) | ICML | `W` | RigL | [PyTorch(Author)](https://github.com/google-research/rigl) | Image Classification | 2019 | 
| 04 | [NeST: A Neural Network Synthesis Tool Based on a Grow-and-Prune Paradigm](https://arxiv.org/abs/1711.02017) | arXiv | `N` | NeST | - | Image Classification | 2019 |
| 05 | [Variational Convolutional Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.html) | CVPR | `F` | VCP | - | Image Classification | 2019 | 
| 06 | [Sparse Networks from Scratch: Faster Training without Losing Performance](https://arxiv.org/abs/1907.04840) | arXiv | `W` | SM | [PyTorch(Author)](https://github.com/TimDettmers/sparse_learning) | Image Classification | 2019 | 
| 07 | [Online Filter Clustering and Pruning for Efficient Convets](https://arxiv.org/abs/1905.11787) | arXiv | `W` | - | - | Image Classification | 2019 | 
| 08 | [Parameter Efficient Training of Deep Convolutional Neural Networks by Dynamic Sparse Reparameterization](https://arxiv.org/abs/1902.05967) | ICML | `W` | DSR | [PyTorch(Not Available)](https://github.com/IntelAI/dynamic-reparameterization) | Image Classification | 2019 | 
| 09 | [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717) | NeurIPS | `F` | TAS | [PyTorch(Author)](https://github.com/D-X-Y/NAS-Projects) | Image Classification | 2019 |
| 10 | [MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/abs/1903.10258) | ICCV | `F` | MetaPruning | [PyTorch(Author)](https://github.com/liuzechun/MetaPruning) | Image Classification | 2019 |
| 11 | [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/abs/2003.13683) | ECCV | `F` | DHP | [PyTorch(Author)](https://github.com/ofsoundof/dhp) | Image Classification&Super-resolution&Denoising | 2019 |
| 12 | [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://arxiv.org/abs/1909.12778) | NeurIPS  | `W` | GSM | [PyTorch(Author)](https://github.com/DingXiaoH/GSM-SGD)  | Image Classification | 2019 |

###### Pruning During Training CNNs 2018 and earlier
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Learning the Number of Neurons in Deep Networks](https://arxiv.org/abs/1611.06321) | NIPS | `N` | - | - | Image Classification | 2016 |
| 02 | [Learning Structured Sparsity in Deep Neural Networks](https://proceedings.neurips.cc/paper/2016/file/41bfd20a38bb1b0bec75acf0845530a7-Paper.pdf)| NIPS | `FC`| SSL | [Caffe(Author)](https://github.com/wenwei202/caffe/tree/scnn) | Image Classification | 2016 |
| 03 | [Learning Efficient Convolutional Networks through Networks Slimming](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)| ICCV | `C`| Slimming | [Lua(Author)](https://github.com/liuzhuang13/slimming) | Image Classification | 2017 |
| 04 | [Deep Rewiring: Training very Sparse Deep Networks](https://arxiv.org/pdf/1711.05136.pdf) | ICLR | `W` | - | - | Image Classification&Audio | 2018 |
| 05 | [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124) | ICLR | `F` | - | [TensorFlow(Author)](https://github.com/bobye/batchnorm_prune) [PyTorch(3rd)](https://github.com/jack-willturner/batchnorm-pruning) | Image Classification&Segmentation | 2018 |
| 06 | [Data-Driven Sparse Structure Selection for Deep Neural Networks](https://arxiv.org/abs/1707.01213) | ECCV | `F` | SSS | [MXNet(Author)](https://github.com/TuSimple/sparse-structure-selection) | Image Classification | 2018 |
| 07 | [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gordon_MorphNet_Fast__CVPR_2018_paper.pdf) | CVPR | `L` | MorphNet | [PyTorch(Author)](https://github.com/google-research/morph-net) | Image Classification | 2018 |
| 08 | [Learning Sparse Neural Networks Through $L_0$ Regularization](https://openreview.net/pdf?id=H1Y8hhg0b) | ICLR | `FN` | - | [PyTorch(Author)](https://github.com/AMLab-Amsterdam/L0_regularization) | Image Classification | 2018 |
| 09 | [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866)  | IJCAI   | `F`  | SFP | [PyTorch(Author)](https://github.com/he-y/soft-filter-pruning) | Image Classification | 2018 |
| 10 | [Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science](https://arxiv.org/pdf/1707.04780.pdf) | Nature Communication | `W&P` | SET | - | Image Classification | 2018 | 



##### 1.1.2.2 Pruning Other Models
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Exploring Sparsity in Recurrent Neural Networks](https://arxiv.org/abs/1704.05119) | ICLR | `W` | - | [PyTorch](https://github.com/puhsu/pruning) | Speech Recognition | 2017 |
| 02 | [Sparse Training via Boosting Pruning Plasticity with Neuroregeneration](https://arxiv.org/abs/2106.10404) | NeurIPS | `H` | GraNet | [PyTorch](https://github.com/VITA-Group/GraNet) | Image Classification | 2021 |
| 03 | [Selfish Sparse RNN Training](http://proceedings.mlr.press/v139/liu21p/liu21p.pdf) | ICML | `W` | SNT-ASGD |[PyTorch(Anthor)](https://github.com/Shiweiliuiiiiiii/Selfish-RNN) | Language Modeling | 2021 |
| 04 | [Dynamic Sparse Training for Deep Reinforcement Learning](https://arxiv.org/pdf/2106.04217.pdf) | IJCAI | `W` | - |[PyTorch(Anthor)](https://github.com/GhadaSokar/Dynamic-Sparse-Training-for-Deep-Reinforcement-Learning) | Continuous Control | 2022 |
| 05 | [The State of Sparse Training in Deep Reinforcement Learning.](https://proceedings.mlr.press/v162/graesser22a/graesser22a.pdf) | ICML | `W` | - |[Tensorflow(Anthor)](github.com/google-research/rigl/tree/master/rigl/rl) | Continuous Control | 2022 |

#### 1.1.3 Pruning After Training
##### 1.1.3.1 Pruning CNNs
###### Pruning After Training CNNs 2024
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Fast and Controllable Post-training Sparsity: Learning Optimal Sparsity Allocation with Global Constraint in Minutes](https://arxiv.org/abs/2203.04570) | AAAI | `W` | FCPTS | - | Image Classification&Object Detection | 2024 |
| 02 | [UPDP: A Unified Progressive Depth Pruner for CNN and Vision Transformer](https://arxiv.org/pdf/2401.06426v1#page=3.05) | AAAI | `L` | UPDP | - | Image Classification&Object Detection | 2024 |

###### Pruning After Training CNNs 2023
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://arxiv.org/abs/2308.07209) | ICCV |  `C` | UDFC | -  | Image Classification | 2023|
| 02 | [Unmasking the Lottery Ticket Hypothesis: What’s Encoded in a Winning Ticket’s Mask?](https://arxiv.org/abs/2210.03044) | ICLR(TOP-25%) | `W` | - | - | Image Classification | 2023 |
| 03 | [DepGraph: Towards Any Structural Pruning](https://www.ijcai.org/proceedings/2018/0336.pdf) | CVPR | `C` | DepGraph | [PyTorch(Author)](https://github.com/VainF/Torch-Pruning)  | CV/NLP | 2023 |
| 04 | [DFPC: Data flow driven pruning of coupled channels without data](https://openreview.net/forum?id=mhnHqRqcjYU) | ICLR | `C` | DFPC | [PyTorch(Author)](https://github.com/TanayNarshana/DFPC-Pruning)  | Image Classification | 2023 |
| 05 | [Memory-Oriented Structural Pruning for Efficient Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/25319) | AAAI | `C` | MOSP | -  | Image Restoration | 2023 |
| 06 | [Trainability Preserving Nueral Structured Pruning](https://openreview.net/pdf?id=AZFvpnnewr) | ICLR | `F` | TPP | [Pytorch(Author)](https://github.com/MingSun-Tse/TPP) | Image Classification | 2023 |

###### Pruning After Training CNNs 2022
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win](https://arxiv.org/pdf/2010.03533.pdf) | AAAI | `W` | - | [PyTorch(Author)](https://github.com/google-research/rigl/tree/master/rigl/rigl_tf2) | Image Classification | 2022 |
| 02 | [How Well Do Sparse ImageNet Models Transfer?](https://arxiv.org/abs/2111.13445) | CVPR | `W` | - |  [PyTorch(Author)](https://github.com/ISTDASLab/sparse-imagenet-transfer) | Image Classification&Object Detection | 2022 |
| 03 | [Lottery Jackpots Exist in Pre-trained Models](https://arxiv.org/abs/2104.08700) | arXiv |  `W` | oBERT | [PyTorch(Author)](https://github.com/zyxxmu/lottery-jackpots/tree/main)| Image Classification | 2022|
| 04 | [Graph Pruning for Model Compression](https://arxiv.org/abs/1911.09817) | Applied Intelligence | `W` | GraphPruning | - | Image Classification | 2022 |
| 05 | [Advancing Model Pruning via Bi-level Optimization](https://arxiv.org/pdf/2210.04092.pdf) | NeurIPS |  `WC` | BiP | [PyTorch(Author)](https://github.com/OPTML-Group/BiP)  | Image Classification | 2022|
| 06 | [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/abs/2208.11580) | NeurIPS | `W` | ExactOBS  | [PyTorch(Author)](https://github.com/IST-DASLab/OBC) | Image Classification&Object Detection&Question Answering | 2022 |
| 07 | [Prune Your Model Before Distill It](https://arxiv.org/abs/2109.14960) | ECCV | `F` | - | [PyTorch(Author)](https://github.com/ososos888/prune-then-distill) | Image Classification | 2022 |
| 08 | [SOSP: Efficiently Capturing Global Correlations by Second-Order Structured Pruning](https://openreview.net/forum?id=t5EmXZ3ZLR) | ICLR **(Spotlight)** | `F` | SOSP | [PyTorch(Author)](https://github.com/boschresearch/sosp)(Releasing)  | Image Classification | 2022 |
| 09 | [Dreaming to Prune Image Deraining Networks](https://arxiv.org/abs/2105.14713) | TPAMI | 1XN | - | [PyTorch(Author)](https://github.com/lmbxmu/1xN) | Image Classification | 2022 |
| 10 | [1xN Pattern for Pruning Convolutional Neural Networks](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.pdf) | CVPR | `F` | - | - | Image Deraining | 2022 |
| 11 | [Prior Gradient Mask Guided Pruning-Aware Fine-Tuning](https://ojs.aaai.org/index.php/AAAI/article/view/19888) | AAAI | `C` | PGMPF | [PyTorch(Author)](https://github.com/cailinhang/PGMPF) | Image Classification | 2022 |

###### Pruning After Training CNNs 2021
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations](https://openreview.net/forum?id=xCxXwTzx4L1) | ICLR | `F` | ChipNet | [PyTorch(Author)](https://github.com/transmuteAI/ChipNet) |Image Classification | 2021 | 
| 02 | [Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot?](https://openreview.net/pdf?id=WL7pr00_fnJ) | NeurIPS | `W` | - |  [PyTorch(Author)](https://github.com/boone891214/sanity-check-LTH) |Image Classification | 2021 |
| 03 | [Multi-Prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning A Randomly Weighted Network](https://openreview.net/forum?id=U_mat0b9iv) | ICLR | `W` | MPTs | [PyTorch(Author)](https://github.com/chrundle/biprop) | Image Classification | 2021 |
| 04 | [Long live the lottery: the existence of winning tickets in lifelong learning](https://arxiv.org/abs/1906.02773) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/Lifelong-Learning-LTH) | Image Classification | 2021 |
| 05 | [Enabling Retrain-free Deep Neural Network Pruning Using Surrogate Lagrangian Relaxation](https://arxiv.org/abs/2012.10079) | IJCAI | `W` | - | - | Image Classification & Object Detection | 2021 |
| 06 | [Joint-DetNAS: Upgrade Your Detector with NAS, Pruning and Dynamic Distillation](https://arxiv.org/abs/2105.12971)| CVPR | `F`| Joint-DetNAS | - | Image Classification & Object Detection | 2021 |
| 07 | [Validating the Lottery Ticket Hypothesis with Inertial Manifold Theory](https://papers.nips.cc/paper/2021/hash/fdc42b6b0ee16a2f866281508ef56730-Abstract.html)  | NeurIPS | `W`     | - | - | Image Classification | 2021 |
| 08 | [Towards Compact CNNs via Collaborative Compression](https://arxiv.org/abs/2105.11228) | CVPR | `F` | CC | [PyTorch(Author)](https://github.com/liuguoyou/Towards-Compact-CNNs-via-Collaborative-Compression) | Image Classification | 2021 |
| 09 | [NPAS: A Compiler-aware Framework of Unified Network Pruning andArchitecture Search for Beyond Real-Time Mobile Acceleration](https://arxiv.org/abs/2012.00596) | CVPR | `F` | NPAS | - | Image Classification | 2021 | 
| 10 | [Neural Pruning via Growing Regularization](https://openreview.net/pdf?id=o966_Is_nPA) | ICLR | `WF` | Greg | - | Image Classification | 2021 |
| 11 | [Towards Adversarial Robustness Via Compact Feature Representations](https://ieeexplore.ieee.org/document/9414696) | ICASSP| `N` | -  | [PyTorch(Author)](https://github.com/lena-voita/the-story-of-heads)| Adversarial Robustness | 2021 |
| 12 | [On the Predictability of Pruning Across Scales](https://arxiv.org/abs/2006.10621) | ICML | `W` | -| - | Image Classification | 2021 |
| 13 | [How much pre-training is enough to discover a good subnetwork?](https://arxiv.org/abs/2108.00259) | arXiv | `W` | -| - | Image Classification | 2021 |
| 14 | [The Lottery Tickets Hypothesis for Supervised and Self-supervised Pre-training in Computer Vision Models](https://arxiv.org/pdf/2005.05232.pdf) | CVPR | `W` | -| [PyTorch(Author)](https://github.com/VITA-Group/CV_LTH_Pre-training) | Image Classification | 2021 |
| 15 | [The Elastic Lottery Ticket Hypothesis](https://papers.nips.cc/paper/2021/hash/dfccdb8b1cc7e4dab6d33db0fef12b88-Abstract.html)  | NeurIPS | `W` | E-LTH | [PyTorch(Author)](https://github.com/VITA-Group/ElasticLTH) | Image Classification | 2021 | 
| 16 | [Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks](https://arxiv.org/abs/2102.08124)  | NeurIPS | N:M | AdaPrune | [PyTorch(Author)](https://github.com/papers-submission/structured_transposable_masks) | Image Classification | 2021 | 
| 17 | [Why Lottery Ticket Wins? A Theoretical Perspective of Sample Complexity on Sparse Neural Networks](https://papers.nips.cc/paper/2021/hash/15f99f2165aa8c86c9dface16fefd281-Abstract.html) | NeurIPS | `W` | - | - | Image Classification | 2021 |
| 18 | [Group Fisher Pruning for Practical Network Compression](https://arxiv.org/abs/2108.00708) | ICML | `F` | GFP | [PyTorch(Author)](https://github.com/jshilong/FisherPruning) | Image Classification&Object Detection | 2021 |
| 19 | [TransTailor: Pruning the Pre-trained Model for Improved Transfer Learning](https://arxiv.org/abs/2103.01542) | AAAI | `F` | TransTailor | - | Image Classification | 2021 |
| 20 | [Network Pruning That Matters: A Case Study on Retraining Variants](https://openreview.net/forum?id=Cb54AMqHQFP) | ICLR | `F` | - | [PyTorch(Author)](https://github.com/lehduong/NPTM) | Image Classification | 2021 | 
| 21 | [The Lottery Ticket Hypothesis for Object Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Girish_The_Lottery_Ticket_Hypothesis_for_Object_Recognition_CVPR_2021_paper.pdf) | CVPR | `W` | - | [PyTorch(Author)](https://github.com/Sharath-girish/LTH-ObjectRecognition) | Object Detection | 2021 |
| 22 | [Lottery Jackpot Exist in Pre-trained Models](https://arxiv.org/pdf/2104.08700.pdf) | TPAMI | `W` | Jackpot | [PyTorch(Author)](https://github.com/zyxxmu/lottery-jackpots) | Image Classification |2021 |
| 23 | [Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework](https://arxiv.org/abs/2010.04879) | ICML | `F` | - | - | Image Classification | 2021 | 
| 24 | [Network Pruning via Performance Maximization](https://openaccess.thecvf.com/content/CVPR2021/papers/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.pdf) | CVPR | `F` | NPPM | [Pytorch(Author)](https://github.com/gaosh/NPPM) | Image Classification | 2021 |
| 25 | [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378) | arXiv | `W` | - | - | Image Classification&Image Segmentation and Detection&Language Modeling&Language Translation | 2021 |
 
###### Pruning After Training CNNs 2020
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [SCOP: Scientific Control for Reliable Neural Network Pruning](https://arxiv.org/abs/2001.08565) | NeurIPS | `F` | SCOP | [PyTorch(Author)](https://github.com/yehuitang/Pruning/tree/master/SCOP_NeurIPS2020) | Image Classification | 2020 |
| 02 | [Discrete Model Compression With Resource Constraint for Deep Neural Networks](http://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Discrete_Model_Compression_With_Resource_Constraint_for_Deep_Neural_Networks_CVPR_2020_paper.html) | CVPR | `F` | - | - | Image Classification | 2020 |
| 03 | [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/abs/2002.10179) | CVPR | `F` | HRank | [Pytorch(Author)](https://github.com/lmbxmu/HRank) | Image Classification | 2020 |
| 04 | [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2020/html/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.html) | CVPR | `F` | LFPC | - | Image Classification | 2020 | 
| 05 | [Towards Efficient Model Compression via Learned Global Ranking](https://arxiv.org/abs/1904.12368)| CVPR | `F` | LeGR | [Pytorch(Author)](https://github.com/cmu-enyac/LeGR) | Image Classification | 2020 |
| 06 | [Reborn filters: Pruning convolutional neural networks with limited data](https://ojs.aaai.org/index.php/AAAI/article/view/6058) | AAAI | `F` | - | - | Image Classification | 2020 |
| 07 | [Operation-Aware Soft Channel Pruning using Differentiable Masks](https://arxiv.org/abs/2007.03938) | ICML| `F` | SCP | - | Image Classification | 2020 |
| 08 | [Neural Network Pruning with Residual-Connections and Limited-Data](https://arxiv.org/abs/1911.08114) | CVPR | `C` | CURL | [PyTorch(Author)](https://github.com/Roll920/CURL) | Image Classification | 2020 |
| 09 | [On the Transferability of Winning Tickets in Non-Natural Image Datasets](https://arxiv.org/pdf/2005.05232.pdf) | arXiv | `W` | -| - | Image Classification | 2020 |
| 10 | [Towards Compact and Robust Deep Networks](https://arxiv.org/abs/1906.06110) | arXiv | `W` | - | - | Image Classification | 2020 |
| 11 | [HYDRA: Pruning Adversarially Robust Neural Networks](https://arxiv.org/abs/2002.10509) | NeurIPS | `W` | HYDRA | [PyTorch(Author)](https://github.com/inspire-group/hydra) | Adversarial Robustness | 2020 |
| 12 | [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683) | NeurIPS | `W` | - | [PyTorch(Author)](https://github.com/huggingface/block_movement_pruning) | NLP | 2020 | 
| 13 | [DMCP: Differentiable Markov Channel Pruning for Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_DMCP_Differentiable_Markov_Channel_Pruning_for_Neural_Networks_CVPR_2020_paper.pdf) | CVPR | `C`  | DMCP | -  | Image Classification | 2020 |
| 14 | [How many winning tickets are there in one DNN?](https://arxiv.org/abs/2006.07014) | arXiv | `W` | - | - | Image Classification | 2020 |
| 15 | [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://arxiv.org/abs/2003.08935) | CVPR | `F` | Hinge | [PyTorch(Author)](https://github.com/ofsoundof/group_sparsity) | Image Classification | 2020 |
| 16 | [Proving the Lottery Ticket Hypothesis for Convolutional Neural Networks](https://openreview.net/forum?id=Vjki79-619-) | ICML | `N` | - | - | - | 2020 |
| 17 | [Logarithmic Pruning is All You Need](https://arxiv.org/abs/2006.12156) | NeurIPS | `N` | - | - | - | 2020 |
| 18 | [Optimal Lottery Tickets via SUBSETSUM:Logarithmic Over-Parameterization is Sufficient](https://arxiv.org/abs/2006.07990) | NeurIPS | `N` | - |  [PyTorch(Author)](https://github.com/acnagle/optimal-lottery-tickets) |Image Classification | 2020 |
| 19 | [EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning](https://arxiv.org/abs/2007.02491) | ECCV | `F` | EagleEye |  [PyTorch(Author)](https://github.com/anonymous47823493/EagleEye) |Image Classification | 2020 |
| 20 | [Channel Pruning via Automatic Structure Search](https://arxiv.org/abs/2001.08565) | IJCAI | `F` | ABC | [PyTorch(Author)](https://github.com/lmbxmu/ABCPruner) | Image Classification | 2020 |

###### Pruning After Training CNNs 2019
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Auto-Balanced Filter Pruning for Efficient Convolutional Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/12262) | AAAI | `F`  | - | - | Image Classification | 2019 |
| 02 | [Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1909.08174) | NeurIPS | `F` | Gate Decorator | [PyTorch(Author)](https://github.com/youzhonghui/gate-decorator-pruning) | Image Classification&Semantic Segmentation | 2019 |
| 03 | [EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis](https://arxiv.org/abs/1905.05934) | ICML | `C`  | EigenDamage | [PyTorch(Author)](https://github.com/alecwangcq/EigenDamage-Pytorch) | Image Classification | 2019 |
| 04 | [Importance Estimation for Neural Network Pruning](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf) | CVPR | `F` | Taylor-FO-BN |[PyTorch(Author)](https://github.com/NVlabs/Taylor_pruning) | Image Classification | 2019 |  
| 05 | [The State of Sparsity in Deep Neural Networks](https://arxiv.org/abs/1902.09574) | arXiv | `w`  | - |[TensorFlow(Author)](https://github.com/google-research/google-research/blob/master/state_of_sparsity/README.md) | Image Classification&machine translation | 2019 |
| 06 | [Collaborative Channel Pruning for Deep Networks](http://proceedings.mlr.press/v97/peng19c.html) | ICML | `F` | CCP | - | Image Classification | 2019 |
| 07 | [One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers](https://arxiv.org/abs/1906.02773) | NeurIPS | `W` | - | - | Image Classification | 2019 |
| 08 | [ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_ECC_Platform-Independent_Energy-Constrained_Deep_Neural_Network_Compression_via_a_Bilinear_CVPR_2019_paper.pdf) | CVPR | `F` | ECC | [Pytorch(Author)](https://github.com/hyang1990/energy_constrained_compression) | Image Classification&Semantic Segmentation | 2019 |
| 09 | [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization github](https://arxiv.org/abs/1905.04748) | ICML | `F` | AOFP | [Pytorch(Author)](https://github.com/DingXiaoH/AOFP) | Image Classification | 2019 |
| 10 | [Sparse Transfer Learning via Winning Lottery Tickets](https://arxiv.org/abs/1905.07785) | arXiv | `W` | - | [PyTorch(Author)](https://github.com/rahulsmehta/sparsity-experiments) | Image Classification | 2019 |
| 11 | [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2019/file/f34185c4ca5d58e781d4f14173d41e5d-Paper.pdf) | NeurIPS | `W` | - | [PyTorch(Author)](https://github.com/DingXiaoH/GSM-SGD) | Image Classification | 2019 |
| 12 | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) | ICLR **(Best)** | `W` | LTH | [TensorFlow(Author)](https://github.com/google-research/lottery-ticket-hypothesis) | Image Classification | 2019 | 
| 13 | [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://arxiv.org/abs/1905.01067) | NeurIPS | `W` | - | [TensorFlow(Author)](https://github.com/uber-research/deconstructing-lottery-tickets) | Image Classification | 2019 |
| 14 | [Winning the Lottery with Continuous Sparsification](https://arxiv.org/abs/1912.04427) | NeurIPS | `F` | CS | [PyTorch(Author)](https://github.com/lolemacs/continuous-sparsification) | Image Classification | 2019 |
| 15 | [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](https://arxiv.org/abs/1904.03837) | CVPR | `F` | C-SGD | [Tensorflow(Author)](https://github.com/ShawnDing1994/Centripetal-SGD) |Image Classification | 2019 |
| 16 | [Exploiting Kernel Sparsity and Entropy for Interpretable CNN Compression](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Exploiting_Kernel_Sparsity_and_Entropy_for_Interpretable_CNN_Compression_CVPR_2019_paper.pdf) | CVPR | `W` | KSE | [PyTorch(Author)](https://github.com/yuchaoli/KSE) | Image Classification |2019 |
| 17 | [Towards Compact ConvNets via Structure-Sparsity Regularized Filter Pruning](https://arxiv.org/abs/1901.07827) | TNNLS | `F` | SSR | [Caffe(Author)](https://github.com/ShaohuiLin/SSR) | Image Classification | 2019 |
| 18 | [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://arxiv.org/abs/1903.09291) | CVPR | `F` | GAL | [PyTorch(Author)](https://github.com/ShaohuiLin/GAL) | Image Classification | 2019 |
| 18 | [Efficient Neural Network Compression](https://arxiv.org/abs/1811.12781) | CVPR | `C` | ENC | [PyTorch(Author)](https://github.com/Hyeji-Kim/ENC) | Image Classification | 2019 |

###### Pruning After Training CNNs 2018
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Accelerating Convolutional Networks via Global & Dynamic Filter Pruning](https://www.ijcai.org/proceedings/2018/0336.pdf) | IJCAI | `F` | GDP | -  | Image Classification | 2018 |
| 02 | [AMC: Automl for model compression and acceleration on mobile devices](https://arxiv.org/abs/1802.03494) | ECCV | `F` | AMC | [TensorFlow(3rd)](https://github.com/Tencent/PocketFlow#channel-pruning) |  Image Classification | 2018 |
| 03 | [Exploring Linear Relationship in Feature Map Subspace for ConvNets Compression](https://arxiv.org/abs/1803.05729)| arXiv | `F`| - | - | Object Detection&Human Pose Estimation | 2018 |
| 04 | [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878) | ICLRW| `W` | - | [TensorFlow(Author)](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning) | NLP | 2018 |
| 05 | [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html)  | CVPR | `W` | CLIP-Q| - | Image Classification | 2018 |
| 06 | [Discrimination-aware Channel Pruning for Deep Neural Networks](https://arxiv.org/abs/1810.11809) | NeurIPS | `C` | DCP | [TensorFlow(Author)](https://github.com/SCUT-AILab/DCP)  | Image Classification | 2018 |
| 07 | [NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/abs/1711.05908) | CVPR | `NC` | NISP | - | Image Classification | 2018 |
| 08 | [2PFPCE: Two-Phase Filter Pruning Based on Conditional Entropy](https://arxiv.org/pdf/1809.02220.pdf) | AAAI | `W` | 2PFPCE | - | Image Classification | 2018 |


###### Pruning After Training CNNs 2017 and earlier
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) | NIPS | `W` | OBD | - | Image Classification | 1989 |
| 02 | [Second Order Derivatives for Network Pruning: Optimal Brain Surgeon](https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf) | NIPS | `W` | OBS | - | Image Classification | 1992 |
| 03 | [Structured Pruning of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1512.08571) | arXiv | `C` | - | - | Image Classification | 2015 |
| 04 | [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) | ICLR **(Best)** | `W`  | - |[Caffe(Author)](https://github.com/songhan/Deep-Compression-AlexNet) | Image Classification | 2016 |
| 05 | [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) | ICCV&TPAMI | `F` | ThiNet | [Caffe(Author)](https://github.com/Roll920/ThiNet), [PyTorch(3rd)](https://github.com/tranorrepository/reprod-thinet) | Image Classification | 2017&2019 |
| 06 | [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) | ICLR | `F` | - | [PyTorch](https://github.com/jacobgil/pytorch-pruning/tree/master) | Image Classification | 2017 |
| 07 | [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) | ICLR    | `F`  | PFEC | [PyTorch(3rd)](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet/l1-norm-pruning) | Image Classification | 2017 |
| 08 | [Channel pruning for accelerating very deep neural networks](https://arxiv.org/abs/1707.06168) | ICCV | `C` | - | [Caffe(Author)](https://github.com/yihui-he/channel-pruning) |Image Classification&Object Detection | 2017 |



##### 1.1.3.2 Pruning ViTs
###### Pruning After Training ViTs 2024
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Fast and Controllable Post-training Sparsity: Learning Optimal Sparsity Allocation with Global Constraint in Minutes](https://arxiv.org/abs/2203.04570) | AAAI | `W` | FCPTS | - | Image Classification&Object Detection | 2024 |
| 02 | [UPDP: A Unified Progressive Depth Pruner for CNN and Vision Transformer](https://arxiv.org/pdf/2401.06426v1#page=3.05) | AAAI | `L` | UPDP | - | Image Classification&Object Detection | 2024 |
| 03 | [Pruning Self-attentions into Convolutional Layers in Single Path](https://arxiv.org/abs/2111.11802) | TPAMI | `H` | SPViT |  [PyTorch](https://github.com/ziplab/SPViT) | Image Classification&Object Detection | 2024 |

###### Pruning After Training ViTs 2023
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [X-Pruner: eXplainable Pruning for Vision Transformers](https://arxiv.org/abs/2303.04935) | CVPR | `CH` | X-Pruner | [Pytorch(Author)](https://github.com/vickyyu90/XPruner) | Image Classification | 2023 |
| 02 | [Global Vision Transformer Pruning with Hessian-Aware Saliency](https://arxiv.org/abs/2110.04869) | CVPR | `CH` | NViT | - | Image Classification | 2023 |
| 03 | [Pruning Parameterization with Bi-level Optimization for Efficient Semantic Segmentation on the Edge](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Pruning_Parameterization_With_Bi-Level_Optimization_for_Efficient_Semantic_Segmentation_on_CVPR_2023_paper.pdf) | CVPR | `W` | STE | - | semantic Segmentation | 2023 |
| 04 | [Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models](https://arxiv.org/abs/2306.10460) | ICML | `W` | ISP | [Pytorch(Author)](https://github.com/VITA-Group/instant_soup) | Image Classification&NLP | 2023 |

###### Pruning After Training ViTs 2022
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Width & Depth Pruning for Vision Transformers](https://cdn.aaai.org/ojs/20222/20222-13-24235-1-2-20220628.pdf) | AAAI | `C` | WDPruning | [Pytorch(Author)](https://github.com/andyrull/width-and-Depth-pruning-for-Vision-Transformer) | Image Classification | 2022 |
| 02 | [SAViT: Structure-Aware Vision Transformer Pruning via Collaborative Optimization](https://cdn.aaai.org/ojs/20222/20222-13-24235-1-2-20220628.pdf) | NeurIPS | `CHE` | SAViT | [Pytorch(Author)](https://github.com/hikvision-research/SAViT) | Image Classification&object detection | 2022 |
| 03 | [VTC-LFC: Vision Transformer Compression with Low-Frequency Components](https://papers.neurips.cc/paper_files/paper/2022/file/5a8177df23bdcc15a02a6739f5b9dd4a-Paper-Conference.pdf) | NeurIPS | `C` | VTC-LFC | [Pytorch(Author)](https://github.com/Daner-Wang/VTC-LFC) | Image Classification | 2022 |
| 04 | [CP-ViT: Cascade Vision Transformer Pruning via Progressive Sparsity Prediction](https://arxiv.org/abs/2203.04570) | arXiv | `H` | CP-ViT  | - | Image Classification | 2022 |
| 05 | [Unified Visual Transformer Compression](https://arxiv.org/abs/2203.08243) | ICLR | `H` | UVC  | [Pytorch(Author)](https://github.com/VITA-Group/UVC) | Image Classification | 2022 |

##### 1.1.3.3 Pruning BERTs
###### Pruning After Training BERTs 2023
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://proceedings.mlr.press/v202/li23ap/li23ap.pdf) | ICML |  `H` | LoSparse | [PyTorch(Author)](https://github.com/yxli2123/LoSparse) | NLP | 2023|
| 02 | [Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models](https://arxiv.org/abs/2306.10460) | ICML | `W` | ISP | [Pytorch(Author)](https://github.com/VITA-Group/instant_soup) | Image Classification&NLP | 2023 |
| 03 | [Gradient-Free Structured Pruning with Unlabeled Data](https://arxiv.org/pdf/2204.00408.pdf) | ICML |  `F` | KCM | - | NLP | 2023|
| 04 | [The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter](https://arxiv.org/abs/2306.03805) | arXiv |  `W`&N:M | - | [Pytorch(Author)](https://github.com/VITA-Group/essential_sparsity?tab=readme-ov-file) | NLP | 2023|

###### Pruning After Training BERTs 2022
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Structured Pruning Learns Compact and Accurate Models](https://arxiv.org/pdf/2204.00408.pdf) | ACL |  `LH` | CoFi | [PyTorch(Author)](https://github.com/princeton-nlp/CoFiPruning)  | Natural Language Understanding | 2022|
| 02 | [From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression](https://arxiv.org/abs/2112.07198) | AAAI |  `WH` | CAP | [PyTorch(Author)](https://github.com/alibaba/AliceMind/tree/main/ContrastivePruning)  | NLP | 2022|
| 03 | [PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance](https://arxiv.org/abs/2206.12562) | ICML |  `WC` | PLATON | [PyTorch(Author)](https://github.com/QingruZhang/PLATON)  | Natural Language Understanding&Question Answering&Image Classification | 2022|
| 04 | [Parameter-Efficient Sparsity for Large Language Models Fine-Tuning](https://arxiv.org/pdf/2205.11005.pdf) | IJCAI | `W` | PST |  [PyTorch(Author)](https://github.com/yuchaoli/pst) | Language Modeling | 2022|
| 05 | [The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models](https://arxiv.org/pdf/2203.07259.pdf) | EMNLP |  `W` | oBERT | [PyTorch(Author)](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)| Natural Language Understanding | 2022|
| 06 | [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/abs/2208.11580) | NeurIPS | `W` | ExactOBS  | [PyTorch(Author)](https://github.com/IST-DASLab/OBC) | Image Classification&Object Detection&Question Answering | 2022 |

###### Pruning After Training BERTs 2021
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Super Tickets in Pre-Trained Language Models: From Model Compression to Improving Generalization](https://arxiv.org/abs/2105.12002) | ACL | `W` | super tickets | [PyTorch(Author)](https://github.com/cliang1453/super-structured-lottery-tickets) | Language Understanding | 2021 | 
| 02 | [Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks](https://arxiv.org/abs/2102.08124)  | NeurIPS | N:M | AdaPrune | [PyTorch(Author)](https://github.com/papers-submission/structured_transposable_masks) | Image Classification | 2021 |
| 03 | [Prune Once for All: Sparse Pre-Trained Language Models](https://arxiv.org/abs/2111.05754)  | NeurIPS Workshop | `W` | OFA | [PyTorch(Author)](https://github.com/IntelLabs/Model-Compression-Research-Package) | NLP | 2021 | 
| 04 | [BERT Busters: Outlier Dimensions that Disrupt Transformers](https://arxiv.org/abs/2105.06990) | ACL | `W` | - | - | NLP | 2021 | 
| 05 | [PARP: Prune, Adjust and Re-Prune for Self-Supervised Speech Recognition](https://arxiv.org/abs/2106.05933) | NeurIPS | `W` | PARP | -| Speach Recognition | 2021 | 
| 06 | [Parameter-Efficient Transfer Learning with Diff Pruning](https://arxiv.org/abs/2012.07463) | ACL | `M` | Diff Pruning | [PyTorch(Author)](https://github.com/dguo98/DiffPruning) | NLP | 2021 | 
| 07 | [EarlyBERT: Efficient BERT training via early-bird lottery tickets](https://arxiv.org/abs/2101.00063) | ACL-IJCNLP| `H` | EarlyBERT | [PyTorch(Author)](https://github.com/VITA-Group/EarlyBERT) | NLP | 2021 |
| 08 | [The Lottery Ticket Hypothesis for Pre-trained BERT Networks](https://arxiv.org/abs/2007.12223) | ICML | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/BERT-Tickets) | Language Modeling | 2021 |
| 09 | [Structured Pruning of Large Language Models](https://arxiv.org/abs/1910.04732) | arXiv | `W` | FLOP | [PyTorch(Author)](https://github.com/asappresearch/flop) | NLP classification | 2021 | 
| 10 | [Accelerating Sparse Deep Neural Networks](https://arxiv.org/abs/2104.08378) | arXiv | `W` | - | - | Image Classification&Image Segmentation and Detection&Language Modeling&Language Translation | 2021 |
| 11 | [Differentiable Subset Pruning of Transformer Heads](https://arxiv.org/abs/2108.04657) | TACL | `H` | - | [PyTorch(Author)](https://github.com/rycolab/differentiable-subset-pruning) | NLP | 2021 |

###### Pruning After Training BERTs 2020
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 03 | [Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](https://arxiv.org/abs/2002.11794)| ICML | `W`| - | - | NLP | 2020 |
| 04 | [When BERT Plays the Lottery, All Tickets Are Winning](https://arxiv.org/abs/2005.00561) | EMNLP | `W` | - | [PyTorch(Author)](https://github.com/sai-prasanna/bert-experiments) | Language Modeling | 2020 |
| 05 | [LadaBERT: Lightweight Adaptation of BERT through Hybrid Model Compression](https://arxiv.org/abs/2004.04124) | COLING | `W` | - | - | NLP(Sentiment Classification,Natural Language Inference,Pairwise Semantic Equivalence) | 2020 |
| 06 | [Pruning Redundant Mappings in Transformer Models via Spectral-Normalized Identity Prior](https://arxiv.org/abs/2010.01791) | EMNLP| `H` | - | - | NLP | 2020 |
| 07 | [Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning](https://arxiv.org/abs/2002.08307) | Rep4NLP| `W` | - | - | NLP | 2020 |

###### Pruning After Training BERTs 2019
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Reweighted Proximal Pruning for Large-Scale Language Representation](http://arxiv.org/abs/1909.12486) | arXiv| `Other`  | -  | - | NLP | 2019 |
| 02 | [Efficient Transformer-based Large Scale Language Representations using Hardware-friendly Block Structured Pruning](https://arxiv.org/abs/2009.08065) | EMNLP| `Other`| - | - | NLP | 2019 |

##### 1.1.3.4 Pruning LLMs
###### Pruning After Training LLMs 2024
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2305.18403) | ACL | `CH` | LoRAPrune | [PyTorch(Author)](https://github.com/aim-uofa/LoRAPrune)  |Language Modeling&Classification | 2024|
| 02 | [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695) | ICLR | `W` |  Wanda | [PyTorch(Author)](https://github.com/locuslab/wanda)  | Language Modeling&Classification | 2024|
| 03 | [SliceGPT: Compress Large Language Models by Deleting Rows and Columns](https://arxiv.org/abs/2401.15024) | ICLR | `CH` | SliceGPT | [PyTorch(Author)](https://github.com/microsoft/TransformerCompression)  | Language Modeling&Classification | 2024|
| 04 | [Fluctuation-based Adaptive Structured Pruning for Large Language Models](https://arxiv.org/abs/2312.11983) | AAAI | `CH` | FLAP | [PyTorch(Author)](https://github.com/CASIA-IVA-Lab/FLAP)  | Language Modeling&Classification | 2024|
| 05 | [BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation](https://arxiv.org/abs/2402.16880) | arXiv | `B` | BESA | [PyTorch(Author)](https://github.com/OpenGVLab/LLMPrune-BESA) |Language Modeling&Classification | 2024|
| 06 | [APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference](https://arxiv.org/abs/2401.12200) | ICML | `HC` | APT | [PyTorch(Author)](https://github.com/roim1998/apt) |Classification | 2024|
| 07 | [Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://arxiv.org/abs/2310.06694) | ICLR | `CH` | Sheared LLaMA | [PyTorch(Author)](https://github.com/princeton-nlp/LLM-Shearing)  | Language Modeling&Classification | 2024|
| 08 | [Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes](https://arxiv.org/abs/2402.05406) | arXiv | `CH` | Bonsai | [PyTorch(Author)](https://github.com/ldery/Bonsai)  | Language Modeling&Classification | 2024|
| 09 | [LaCo: Large Language Model Pruning via Layer Collapse](https://arxiv.org/abs/2402.11187) | arXiv | `L` | LaCo | - | Language Modeling&Classification | 2024|
| 10 | [ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853) | arXiv | `L` | ShortGPT | - | Language Modeling&Classification | 2024|
| 11 | [SparseLLM: Towards Global Pruning for Pre-trained Language Models](https://arxiv.org/abs/2402.17946) | arXiv | `B` | SparseLLM | [PyTorch(Author)](https://github.com/baithebest/sparsellm) | Language Modeling&Classification | 2024|
| 12 | [SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025) | arXiv | `N` | SLEB | [PyTorch(Author)](https://github.com/jiwonsong-dev/sleb) | Language Modeling&Classification | 2024|
| 13 | [Streamlining Redundant Layers to Compress Large Language Models](https://arxiv.org/abs/2403.19135) | arXiv | `L` | LLMStreamline | - | Language Modeling&Classification | 2024|
| 14 | [Why Lift so Heavy? Slimming Large Language Models by Cutting Off the Layers](https://arxiv.org/pdf/2402.11700) | arXiv | `L` | - | - |Classification | 2024|
| 15 | [Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods](https://arxiv.org/abs/2402.02834) |ICLRW | `HC` | - | [PyTorch(Author)](https://github.com/nota-netspresso/shortened-llm) |Classification | 2024|
| 16 | [Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity](https://arxiv.org/abs/2309.10285) | VLDB | `W` | Flash-LLM | [PyTorch(Author)](https://github.com/alibabaresearch/flash-llm) | Recognizing Textual Entailment | 2024|
| 17 | [The LLM Surgeon](https://arxiv.org/abs/2312.17244) | arXiv | `WC` | LLM Surgeon | [PyTorch(Author)](https://github.com/Qualcomm-AI-research/llm-surgeon) | Language Modeling | 2024|
| 18 | [Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity](https://arxiv.org/abs/2310.05175) | ICML | `W` | OWL | [PyTorch(Author)](https://github.com/luuyin/owl) | Language Modeling&Classification | 2024|
| 19 | [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/abs/2403.17887) | arXiv | `B` | - | - | Classification | 2024|
| 20 | [Enhancing One-Shot Pruned Generative Pre-training Language Models through Sparse-Dense-Sparse Mechanism](https://openreview.net/forum?id=TjXjkxhSdE&referrer=%5Bthe%20profile%20of%20Dong%20Li%5D(%2Fprofile%3Fid%3D~Dong_Li13)) | OpenReview | `W` | SDS | - | Classification | 2024|
| 21 | [KS-Lottery: Finding Certified Lottery Tickets for Multilingual Language Models](https://arxiv.org/abs/2402.02801) | arXiv | `W` | - | - | Language Translation | 2024|


###### Pruning After Training LLMs 2023
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/pdf/2301.00774.pdf) | NeurIPS |  `WP` | - | [PyTorch(Author)](https://github.com/IST-DASLab/sparsegpt)  | Language Modeling&Classification | 2023|
| 02 | [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627) | arXiv | `CHP` | LLM-Pruner |  [PyTorch(Author)](https://github.com/horseee/LLM-Pruner) | Language Modeling&Language Generation&Classification | 2023|
| 03 | [LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery](https://arxiv.org/abs/2310.18356) | arXiv | `CH` | LoRAShear | - | Language Modeling&Language Generation&Classification | 2023|
| 04 | [Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models](https://arxiv.org/abs/2310.05015) | arXiv | `CH` | Compresso | [PyTorch(Author)](https://github.com/microsoft/Moonlit/tree/main/Compresso) | Classification | 2023|
| 05 | [Mini-GPTs: Efficient Large Language Models through Contextual Pruning](https://arxiv.org/abs/2312.12682) | arXiv | `WC` | - | - |Language Modeling& Classification | 2023|
| 06 | [The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter](https://arxiv.org/abs/2306.03805) | arXiv |  `W`&N:M | - | [Pytorch(Author)](https://github.com/VITA-Group/essential_sparsity?tab=readme-ov-file) | NLP | 2023|


##### 1.1.3.5 Pruning Diffusion Models
###### Pruning After Training Diffusion Models 2023
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Structural Pruning for Diffusion Models](https://arxiv.org/abs/2305.10924) |NeurIPS | `C` | Diff-Pruning | [PyTorch(Author)](https://github.com/VainF/Diff-Pruning) | Image Generation | 2023 |

##### 1.1.3.6 Pruning Vision-and-Languages
###### Pruning After Training VLMs 2024
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models](https://arxiv.org/abs/2310.02998) | ICLR | `L` | ECoFLaP | [Pytorch(Author)](https://ecoflap.github.io/) | VQA&Image Captioning&Image-text Retrieval&Image Classification | 2024 |

###### Pruning After Training VLMs 2023
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Large Multimodal Model Compression via Efficient Pruning and Distillation at AntGroup](https://arxiv.org/abs/2312.05795) | arXiv | `B` | - | - | Multimodal Advertisement Audition | 2023 |
| 02 | [UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers](https://arxiv.org/abs/2301.13741) | ICML | `H` | UPop | [Pytorch(Author)](https://github.com/sdc17/UPop) | Image Classification&Image Caption&Image Retrieval&VQA | 2023 |
| 03 | [Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models](https://arxiv.org/abs/2306.10460) | ICML | `W` | ISP | [Pytorch(Author)](https://github.com/VITA-Group/instant_soup) | Image Classification&NLP | 2023 |

###### Pruning After Training VLMs 2022
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Playing Lottery Tickets with Vision and Language](https://arxiv.org/abs/2104.11832) | AAAI | `W` | - | - | Vision-and-Language | 2022 |


##### 1.1.3.7 Pruning Other Models
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418) | ACL| `W` | - | [PyTorch(Author)](https://github.com/lena-voita/the-story-of-heads)| NLP | 2019 |
| 02 | [Playing the Lottery with Rewards and Multiple Languages: Lottery Tickets in RL and NLP](https://arxiv.org/abs/1906.02768) | ICLR | `W` | - | - | Classic Control&Atari Game | 2020 |
| 03 | [Dynamic Sparsity Neural Networks for Automatic Speech Recognition](https://arxiv.org/abs/2005.10627) | ICASSP | `W` |- | -| Speach Recognition | 2021 |
| 04 | [GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/pdf/2003.08936.pdf) | arXiv | `C` | - | - | Image-to-Image Translation | 2021 |
| 05 | [Content-Aware GAN Compression](https://arxiv.org/abs/2104.02244) | CVPR | `F` | - | [PyTorch(Author)](https://github.com/lychenyoko/content-aware-gan-compression) | Image Generation, Image Projection, Image Editing | 
| 06 | [A Unified Lottery Ticket Hypothesis for Graph Neural Networks](https://arxiv.org/abs/2102.06790) | ICML | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/Unified-LTH-GNN) | Node Classification&Link Prediction | 2021 |
| 07 | [Winning Lottery Tickets in Deep Generative Models](https://arxiv.org/abs/2010.02350) | AAAI | `W` | - | - | Image generative | 2021 |
| 08 | [GANs Can Play Lottery Tickets Too](https://arxiv.org/abs/2106.00134) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/GAN-LTH) | Image generative | 2021 |
| 09 | [Layer-wise Pruning of Transformer Attention Heads for Efficient Language Modeling](https://arxiv.org/abs/2110.03252) | arXiv | `H` | - | [PyTorch(Author)](https://github.com/aiha-lab/Attention-Head-Pruning) | Lanugage Modeling | 2021 |
| 10 | [Can We Find Strong Lottery Tickets in Generative Models?](https://arxiv.org/abs/2212.08311) | arXiv | `W` | - | - | Image generative | 2022 |
| 11 | [Exploring Lottery Ticket Hypothesis in Spiking Neural Networks](https://arxiv.org/abs/2207.01382) | ECCV | `W` | ET | [PyTorch(Author)](https://github.com/intelligent-computing-lab-yale/exploring-lottery-ticket-hypothesis-in-snns) | Image Classification | 2022 |
| 12 | [Structured Pruning for Efficient Generative Pre-trained Language Models](https://aclanthology.org/2023.findings-acl.692/) | ACL | `C` | CP3 | - | Language Modeling&Machine Translation&Abstractive Summarization | 2023 |
| 13 | [Rethinking Graph Lottery Tickets: Graph Sparsity Matters](https://arxiv.org/abs/2305.02190) | ICLR | `W` | - | - | Node Classification | 2023 |
| 14 | [CP3: Channel Pruning Plug-in for Point-based Networks](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_CP3_Channel_Pruning_Plug-In_for_Point-Based_Networks_CVPR_2023_paper.pdf) | CVPR | `C` | CP3 | - | 3D Image Classification and Object Detection | 2023 |



##### 1.1.3.8 Post Training
###### Post Training 2024
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Fast and Controllable Post-training Sparsity: Learning Optimal Sparsity Allocation with Global Constraint in Minutes](https://arxiv.org/abs/2203.04570) | AAAI | `W` | FCPTS | - | Image Classification | 2024 |

###### Post Training 2023
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://arxiv.org/pdf/2301.00774.pdf) | NeurIPS |  `WP` | - | [PyTorch(Author)](https://github.com/IST-DASLab/sparsegpt)  | Language Modeling | 2023|
| 02 | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://arxiv.org/abs/2308.07209) | ICCV |  `C` | UDFC | -  | Image Classification | 2023|
| 03 | [OTOv3: Automatic Architecture-Agnostic Neural Network Training and Compression from Structured Pruning to Erasing Operators](https://arxiv.org/abs/2312.09411) | arXiv | `WFC`  | - | -  | Image Classification | 2023|

###### Post Training 2022
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [CP-ViT: Cascade Vision Transformer Pruning via Progressive Sparsity Prediction](https://arxiv.org/abs/2203.04570) | arXiv | `H` | CP-ViT  | - | Image Classification | 2022 |
| 02 | [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://arxiv.org/abs/2208.11580) | NeurIPS | `W` | ExactOBS  | [PyTorch(Author)](https://github.com/IST-DASLab/OBC) | Image Classification&Object Detection&Question Answering | 2022 |
| 03 | [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/pdf/2210.04092.pdf) | NeurIPS |  `HF` | - | [PyTorch(Author)](https://github.com/WoosukKwon/retraining-free-pruning)  | Natural Language Understanding | 2022|

###### Post Training 2021
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Enabling Retrain-free Deep Neural Network Pruning Using Surrogate Lagrangian Relaxation](https://arxiv.org/abs/2012.10079) | IJCAI | `W` | - | - | Image Classification & Object Detection | 2021 |
| 02 | [Accelerated Sparse Neural Training: A Provable and Efficient Method to Find N:M Transposable Masks](https://arxiv.org/abs/2102.08124)  | NeurIPS | N:M | AdaPrune | [PyTorch(Author)](https://github.com/papers-submission/structured_transposable_masks) | Image Classification | 2021 | 


#### 1.1.4 Pruning in Early Training
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://arxiv.org/abs/1912.05671) | ICML | `W` | - | - | Image Classification | 2020 |
| 02 | [When To Prune? A Policy Towards Early Structural Pruning](https://openaccess.thecvf.com/content/CVPR2022/html/Shen_When_To_Prune_A_Policy_Towards_Early_Structural_Pruning_CVPR_2022_paper.html) | CVPR | `F` | PaT | - | Image Classification | 2022 |
| 03 | [Drawing Early-Bird Tickets: Towards More Efficient Training of Deep Networks](https://arxiv.org/abs/1909.11957) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/GATECH-EIC/Early-Bird-Tickets) | Image Classification | 2020 |
| 04 | [A Gradient Flow Framework For Analyzing Network Pruning](https://arxiv.org/abs/2009.11839) | ICLR | `F` | - | [PyTorch(Author)](https://github.com/ModelTC/FCPTS) | Image Classification | 2021 |

### 1.2 Dynamic Pruning
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Channel Gating Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2017/file/a51fb975227d6640e4fe47854476d133-Paper.pdf) | NeurIPS | `F` | RNP | - | Image Classification | 2017 |
| 02 | [Channel Gating Neural Networks](https://arxiv.org/abs/1805.12549) | NeurIPS | `C` | CGNet | [PyTorch(Author)](https://github.com/cornell-zhang/dnn-gating) | Image Classification | 2019 |
| 03 | [Dynamic Channel Pruning: Feature Boosting and Suppression](https://arxiv.org/pdf/1810.05331.pdf) | ICLR | `C` | FBS | [PyTorch(Author)](https://github.com/YOUSIKI/PyTorch-FBS) | Image Classification | 2019 |
| 04 | [Frequency-Domain Dynamic Pruning for Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2018/file/a9a6653e48976138166de32772b1bf40-Paper.pdf) | NeurIPS | `F` | FDNP | - | Image Classification | 2019 |
| 05 | [Fire Together Wire Together: A Dynamic Pruning Approach With Self-Supervised Mask Prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Elkerdawy_Fire_Together_Wire_Together_A_Dynamic_Pruning_Approach_With_Self-Supervised_CVPR_2022_paper.html) | CVPR| `F` | - | - | Image Classification | 2019 |
| 06 | [Dynamic Dual Gating Neural Networks](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Dynamic_Dual_Gating_Neural_Networks_ICCV_2021_paper.pdf) | ICCV | `C` | DGNet | [PyTorch(Author)](https://github.com/lfr-0531/DGNet) | Image Classification | 2021 |
| 07 | [Manifold Regularized Dynamic Network Pruning](https://arxiv.org/abs/2103.05861) | CVPR | `F` | ManiDP |  [PyTorch(Author)](https://github.com/huaweinoah/Pruning/tree/master/ManiDP) | Image Classification | 2021 |
| 08 | [Contrastive Dual Gating: Learning Sparse Features With Contrastive Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Meng_Contrastive_Dual_Gating_Learning_Sparse_Features_With_Contrastive_Learning_CVPR_2022_paper.html) | CVPR | `WF` | CDG | - | Image Classification | 2022 |


## 2. Learning and Pruning

### 2.1 Continual learning
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| 01 | [Continual Learning via Neural Pruning](https://arxiv.org/abs/1903.04476)| arXiv | CLNP | - | Image Classification | 2019 |
| 02 | [Learning Bayesian Sparse Networks With Full Experience Replay for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Yan_Learning_Bayesian_Sparse_Networks_With_Full_Experience_Replay_for_Continual_CVPR_2022_paper.html)| CVPR | SNCL | - | Image Classification | 2022 |  
| 03 | [Continual Prune-and-Select: Class-Incremental Learning with SPecialized Subnetworks](https://arxiv.org/pdf/2208.04952.pdf)| Applied Intelligence | - | [PyTorch(Author)]( https://github.com/adekhovich/continual_prune_and_select) | Image Classification | 2023 |
| 04 | [Continual Domain Adaptation through Pruning-aided Domain-specific Weight Modulation](https://arxiv.org/abs/2304.07560)| CVPRW | PaCDA | [PyTorch(Author)]( https://github.com/prasannab29/pacda) | Image Classification | 2023 |

### 2.2 Contrastive learning
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| 01 | [Studying the impact of magnitude pruning on contrastive learning methods](https://arxiv.org/pdf/2207.00200.pdf) | ICML | - | [PyTorch(Author)](https://github.com/FraCorti/Studying-the-impact-of-magnitude-pruning-on-contrastive-learning-methods) | Image Classification | 2020 |
| 02 | [Training Debiased Subnetworks with Contrastive Weight Pruning](https://openaccess.thecvf.com/content/CVPR2023/papers/Park_Training_Debiased_Subnetworks_With_Contrastive_Weight_Pruning_CVPR_2023_paper.pdf) | CVPR | DCWP | - | Image Classification | 2023 |

### 2.3 Federated learning
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| 01 | [FedDUAP: Federated Learning with Dynamic Update and Adaptive Pruning Using Shared Data on the Server](https://arxiv.org/pdf/2204.11536.pdf) | IJCAI | FedDUAP | - | Image Classification | 2020 |
| 02 | [Model Pruning Enables Efficient Federated Learning on Edge Devices](https://arxiv.org/pdf/1909.12326.pdf) | TNNLS | - | [PyTorch(Author)](https://github.com/jiangyuang/PruneFL) | Image Classification | 2022 |


## 3. Application

### 3.1 Computer Vision
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Deep Rewiring: Training very Sparse Deep Networks](https://arxiv.org/pdf/1711.05136.pdf) | ICLR | - | Image Classification&Audio | 2018 |
| 02 | [Co-Evolutionary Compression for Unpaired Image Translation](https://arxiv.org/pdf/1907.10804.pdf) | ICCV | [PyTorch(Author)](https://github.com/yehuitang/Pruning) | Image Style Translation | 2019 |
| 03 | [Content-Aware GAN Compression](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Content-Aware_GAN_Compression_CVPR_2021_paper.pdf) | CVPR |  [PyTorch(Author)](https://github.com/lychenyoko/content-aware-gan-compression) | Image Style Translation | 2021 |
| 04 | [Training Neural Networks with Fixed Sparse Masks](https://arxiv.org/abs/2111.09839) | NeurIPS | [PyTorch(Author)]( https://github.com/varunnair18/FISH) | Image Classification | 2021 |
| 05 | [Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space](https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Vision_Transformer_Slimming_Multi-Dimension_Searching_in_Continuous_Optimization_Space_CVPR_2022_paper.pdf) | CVPR | [PyTorch(Author)](https://github.com/Arnav0400/ViT-Slim) | Image Classification&Audio | 2022 |
| 06 | [SuperTickets: Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning](https://arxiv.org/abs/2207.03677) | ECCV | [PyTorch(Author)](https://github.com/GATECH-EIC/SuperTickets) | Image Classification&Object Detection&Human Pose Estimation | 2022 |


### 3.2 Natural Language Processing

| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [When BERT Plays the Lottery, All Tickets Are Winning](https://arxiv.org/abs/2005.00561) | EMNLP | [PyTorch(Author)](https://github.com/sai-prasanna/bert-experiments) | Language Modeling | 2020 |
| 02 | [The Lottery Ticket Hypothesis for Pre-trained BERT Networks](https://arxiv.org/abs/2007.12223) | ICML | [PyTorch(Author)](https://github.com/VITA-Group/BERT-Tickets) | Language Modeling | 2021 |
| 03 | [Structured Pruning Learns Compact and Accurate Models](https://arxiv.org/pdf/2204.00408.pdf) | ACL | [PyTorch(Author)](https://github.com/OPTML-Group/BiP)  | Natural Language Understanding | 2022|
| 04 | [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/pdf/2204.09656.pdf) | NeurIPS | [PyTorch(Author)](https://github.com/WoosukKwon/retraining-free-pruning) | Natural Language Understanding | 2022 |
| 05 | [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/pdf/2210.04092.pdf) | NeurIPS | [PyTorch(Author)](https://github.com/WoosukKwon/retraining-free-pruning)  | Natural Language Understanding | 2022|
| 06 | [The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models](https://arxiv.org/pdf/2203.07259.pdf) | EMNLP | [PyTorch(Author)](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)| Natural Language Understanding | 2022|
| 07 | [Pruning Meets Low-Rank Parameter-efficient](https://arxiv.org/abs/2305.18403) | arXiv |  -  | Image Classification&Language Modeling | 2023|
| 08 | [LLM-Pruner: On the Structural Pruning of Large Language Models](https://arxiv.org/abs/2305.11627) | arXiv |  -  | Language Modeling | 2023|

### 3.3 Audio Signal Processing
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Exploring Sparsity in recurrent neural networks](https://arxiv.org/abs/1704.05119) | ICLR | [PyTorch](https://github.com/puhsu/pruning) | Speech Recognition | 2017 |
| 02 | [Deep Rewiring: Training very Sparse Deep Networks](https://arxiv.org/pdf/1711.05136.pdf) | ICLR | - | Image Classification&Audio | 2018 |


## 4. Combination

### 4.1 Pruning and Quantization
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html)  | CVPR | - | Image Classification | 2018 |
| 02 | [Accelerating Sparse Deep Neural Networks](https://arxiv.org/pdf/2104.08378.pdf) | arXiv | - | Image Classification&Object Detection&Language Translation&Language Modeling&Image Synthesis&Domain Translation&Style Transfer&Image-Image Translation&Super Resolution | 2021 |
| 03 | [OPQ: Compressing Deep Neural Networks with One-shot Pruning-Quantization](https://arxiv.org/pdf/2205.11141.pdf) | AAAI | - | Image Classification | 2021 |
| 04 | [Deep Model Compression Based on the Training History](https://arxiv.org/pdf/2102.00160.pdf) | arXiv | - | Image Classification | 2022 |
| 05 | [LLM-Pruner: On the Structural Pruning of Large Language Models](arxiv.org/abs/2305.11627) | arXiv | [PyTorch](https://github.com/horseee/LLM-Pruner) | Causal Language Modeling | 2023 |
| 06 | [Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning](https://openaccess.thecvf.com/content/ICCV2023/papers/Bai_Unified_Data-Free_Compression_Pruning_and_Quantization_without_Fine-Tuning_ICCV_2023_paper.pdf) | ICCV | - | Image Classification | 2023 |


### 4.2 Pruning and Decomposition
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Enhanced Network Compression Through Tensor Decompositions and Pruning](https://ieeexplore.ieee.org/document/10463116) | TNNLS | [PyTorch(Author)](https://github.com/pvti/NORTON) | Image Classification&Object Detection&Human Pose Estimation | 2024 |
| 02 | [Efficient tensor decomposition-based filter pruning](https://doi.org/10.1016/j.neunet.2024.106393) | Neural Networks | [PyTorch(Author)](https://github.com/pvti/CORING) | Image Classification&Object Detection&Human Pose Estimation | 2024 |


## 5. Survey of Pruning
### Survey of Pruning 2024
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Structured Pruning for Deep Convolutional Neural Networks: A survey](https://arxiv.org/pdf/2303.00566.pdf) | TPAMI | - | CV&NLP | 2024 |
| 02 | [A survey on efficient vision transformers: algorithms, techniques, and performance benchmarking](https://arxiv.org/abs/2309.02031) | arXiv | - | CV | 2024 |
| 03 | [A Survey of Lottery Ticket Hypothesis](https://arxiv.org/abs/2403.04861) | arXiv | - | CV&NLP | 2024 |
| 04 | [Model Compression and Efficient Inference for Large Language Models: A Survey](https://arxiv.org/abs/2402.09748) | arXiv | - | NLP | 2024 |

### Survey of Pruning 2023
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning](https://arxiv.org/pdf/2301.05219.pdf) | arXiv | [PyTorch(Author)](https://github.com/MingSun-Tse/Why-the-State-of-Pruning-So-Confusing) | Image Classification | 2023 |
| 02 | [Transforming Large-Size to Lightweight Deep Neural Networks for IoT Applications](https://dl.acm.org/doi/10.1145/3570955) | ACM Computing Surveys | - | CV&NLP&Audio | 2023 |
| 03 | [A Survey on Model Compression for Large Language Models](https://arxiv.org/abs/2308.07633) | TACL | - | NLP&Unseen Instructions | 2023 |
| 04 | [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234) | arXiv | - | - | 2023 |
| 05 | [A Survey on Dynamic Neural Networks for Natural Language Processing](https://arxiv.org/pdf/2202.07101.pdf) | arXiv | - | NLP | 2023 |
| 06 | [Dimensionality Reduced Training by Pruning and Freezing Parts of a Deep Neural Network, a Survey](https://arxiv.org/abs/2205.08099) | arXiv | - | CV&NLP | 2023 |


### Survey of Pruning 2022
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [A Survey on Efficient Convolutional Neural Networks and Hardware Acceleration](https://arxiv.org/pdf/2103.06460.pdf) | Electronics | - | - | 2022 |
| 02 | [Dimensionality Reduced Training by Pruning and Freezing Parts of a Deep Neural Network, a Survey](https://arxiv.org/pdf/2205.08099.pdf) | arXiv | - | Image Classification | 2022 |
| 03 | [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732) | arXiv | - | CV&NLP | 2022 |
| 04 | [Recent Advances on Neural Network Pruning at Initialization](https://arxiv.org/pdf/2103.06460.pdf) | IJCAI | - | CV&NLP | 2022 |

### Survey of Pruning 2021
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://arxiv.org/abs/2102.00554) | JMLR | - | Image Classification | 2021 |
| 02 | [Dynamic Neural Networks: A Survey](https://arxiv.org/pdf/2102.04906.pdf) | arXiv | - | - | 2021 |
| 03 | [Pruning and Quantization for Deep Neural Network Acceleration: A Survey](https://arxiv.org/pdf/2101.09671.pdf) | Neurocomputing | - | Image Classification | 2021 |
| 04 | [Compressing Large-Scale Transformer-Based Models: A Case Study on BERT](https://arxiv.org/abs/2002.11985) | TACL | - | NLP | 2021 |

### Survey of Pruning 2020
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey](https://ieeexplore.ieee.org/document/9043731) | IEEE | - | - | 2020 |
| 02 | [Pruning Algorithms to Accelerate Convolutional Neural Networks for Edge Applications: A Survey](https://arxiv.org/pdf/2005.04275.pdf) | arXiv | - | Image Classification | 2020 |
| 03 | [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/pdf/1710.09282.pdf) | arXiv | - | - | 2020 |
| 04 | [An Survey of Neural Network Compression](https://arxiv.org/pdf/2006.03669.pdf) | arXiv | - | - | 2020 |
| 05 | [Convolutional Neural Network Pruning: A Survey](https://ieeexplore.ieee.org/document/9189610) | CCC | - | - | 2020 |
| 06 | [What is the State of Neural Network Pruning?](https://arxiv.org/pdf/2003.03033.pdf) | MLSys | - | - | 2020 |
| 07 | [A comprehensive survey on model compression and acceleration](https://link.springer.com/article/10.1007/s10462-020-09816-7) | Artificial Intelligence Review | - | - | 2020 |
| 08 | [A Survey on Deep Neural Network Compression: Challenges, Overview, and Solutions](https://arxiv.org/pdf/2010.03954.pdf) | arXiv | - | - | 2020 |

### Survey of Pruning 2019 and earlier
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Pruning Algorithms-A Survey](https://ieeexplore.ieee.org/document/248452) | IEEE Transactions on Neural Networks | - | Image Classification | 1993 |
| 02 | [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://arxiv.org/abs/1703.09039) | arXiv | - | Image Classification | 2017 |
| 03 | [Recent advances in efficient computation of deep convolutional neural networks](https://arxiv.org/pdf/1802.00939.pdf) | arXiv | - | - | 2018 |
| 04 | [The State of Sparsity in Deep Neural Networks](https://arxiv.org/abs/1902.09574) | arXiv | [PyTorch(Author)](https://github.com/google-research/google-research/blob/master/state_of_sparsity/README.md) | Image Classification&machine translation | 2019 |


## 6. Other Works
### Papers
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:-----:|:-------:|:----:|:----:|:----:|
| 01 | [Is Pruning Compression?: Investigating Pruning Via Network Layer Similarity](https://openaccess.thecvf.com/content_WACV_2020/papers/Blakeney_Is_Pruning_Compression_Investigating_Pruning_Via_Network_Layer_Similarity_WACV_2020_paper.pdf) | WACV | - | - | Image Classification | 2020 |
| 02 | [A Gradient Flow Framework For Analyzing Network Pruning](https://openreview.net/forum?id=rumv7QmLUue) | ICLR | - | [PyTorch(Author)](https://github.com/EkdeepSLubana/flowandprune) | Image Classification | 2021 |
| 03 | [Data Level Lottery Ticket Hypothesis for Vision Transformers](https://arxiv.org/abs/2211.01484) | IJCAI | - | [PyTorch(Author)](https://github.com/shawnricecake/vit-lottery-ticket-input) | Image Classification | 2021 |
| 04 | [Are All Layers Created Equal?](https://arxiv.org/abs/1902.01996) | JMLR | - | - | Image Classification | 2022 |



### Useful Links
https://github.com/airaria/TextPruner  

## Acknowledgements
We would like to express our gratitude to the authors of the articles cited in our survey and the authors of the following repositories.

https://github.com/he-y/awesome-Pruning/  
https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization  
https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression/blob/master/Paper/Pruning.md  

## Citation
If you find this project useful, please cite
```bibtex
@article{cheng2023survey,
  title={A Survey on Deep Neural Network Pruning:Taxonomy, Comparison, Analysis, and Recommendations},
  author={Hongrong Cheng and Miao Zhang and Javen Qinfeng Shi},
  journal={arXiv preprint arXiv:2308.06767},
  year={2023}
}
```


