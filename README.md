# awesome-pruning
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
## Table of Contents
- [0. Overview](#0-overview)
- [1. When to prune](#1-when-to-prune)
  <!-- - [Variational Gap Optimization](#Variational-Gap-Optimization) -->
  <!-- - [Dimension Deduction](#Dimension-Deduction) -->
  - [1.1 Static Pruning](#11-static-pruning)
    - [1.1.1 Pruning Before Training](#111-pruning-before-training)
    - [1.1.2 Pruning During Training](#112-pruning-during-training)
    - [1.1.3 Pruning After Training](#113-pruning-after-training)
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
- [4. Survey of Pruning](#4-survey-of-pruning)  
- [5. Other Works](#5-other-works)
- [Acknowledgements](#acknowledgements)  

## 0. Overview
A curated list for neural network pruning introduced by the paper [_**A Comprehensive Survey on Deep Neural Network Pruning**_](under review)


## 1. When to Prune
### 1.1 Static Pruning

| Type        |`L`             | `F`            | `C`             | `N`             | `W`            | `P`        | `Other`     |
|:----------- |:--------------:|:--------------:|:---------------:|:---------------:|:--------------:|:----------:|:-----------:|
| Explanation | Layer pruning  | Filter pruning | Channel pruning |  Neuron pruning | Weight pruning |  Pioneer   | other types |

#### 1.1.1 Pruning Before Training
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:-----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)| ICLR| `W`&`P` | SNIP | [TensorFLow(Author)](https://github.com/namhoonlee/snip-public) | Image Classification | 2019 |
| 02 | [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://arxiv.org/abs/1906.06307)| ICLR **(Spotlight)** | `W` | - | [TensorFLow(Author)](https://github.com/namhoonlee/spp-public) | Image Classification | 2020 |
| 03 | [Picking Winning Tickets before Training by Preserving Gradient Flow](https://openreview.net/pdf?id=SkgsACVKPH))| ICLR | `W` | GraSP | [PyTorch(Author)](https://github.com/alecwangcq/GraSP) | Image Classification | 2020 |      
| 04 | [Pruning from Scratch](http://arxiv.org/abs/1909.12579) | AAAI | `C` | - | [PyTorch(Author)](https://github.com/frankwang345/pruning-from-scratch) | Image Classification | 2020 |
| 05 | [Pruning neural networks without any data by iteratively conserving synaptic flow](https://arxiv.org/abs/2006.05467)| NeurIPS | `W` | SynFlow | [PyTorch(Author)](https://github.com/ganguli-lab/Synaptic-Flow) | Image Classification | 2020 |
| 06 | [A Unified Paths Perspective for Pruning at Initialization](https://arxiv.org/abs/2101.10552)| arXiv | `W` | - | - | Image Classification | 2021 |
| 07 | [Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot](https://proceedings.neurips.cc/paper/2020/file/eae27d77ca20db309e056e3d2dcd7d69-Paper.pdf) | NeurIPS | `W` | Smart-Ratios | [PyTorch(Author)](https://github.com/JingtongSu/sanity-checking-pruning) | Image Classification | 2020 |
| 08 | [Progressive Skeletonization: Trimming More Fat from a network at initialization](https://arxiv.org/abs/2006.09081) | ICLR | `W` | FORCE | [PyTorch(Author)](https://github.com/naver/force) | Image Classification | 2021 |
| 09 | [Robust Pruning at Initialization](https://openreview.net/forum?id=vXj_ucZQ4hA) | ICLR | `W` | SPB | - | Image Classification | 2021 |
| 10 | [Prunining via Iterative Ranking of Sensitivity Statics](https://arxiv.org/abs/2006.00896) | arXiv | `WFC` | SNIP-it | [PyTorch(Author)](https://github.com/StijnVerdenius/SNIP-it) | Image Classification | 2020 |
| 11 | [Prunining Neural Networks at Initialization: Why are We Missing the Mark?](https://arxiv.org/abs/2009.08576) | ICLR | `W` | - | - | Image Classification | 2021 |
| 12 | [Why is Pruning at Initialization Immune to Reinitializating and Shuffling?](https://arxiv.org/abs/2107.01808)) | arXiv | `W` | - | - | Image Classification | 2021 |
| 13 | [Prospect Pruning: Finding Trainable Weights at Initialization using Meta-Gradients](https://openreview.net/forum?id=AIgn9uwfcD1)| ICLR | `WF`| ProsPr | [PyTorch(Author)](https://github.com/mil-ad/prospr) | Image Classification | 2022 |
| 14 | [Dual Lottery Ticket Hypothesis](https://openreview.net/forum?id=fOsN52jn25l) | ICLR | `W` | RST | [PyTorch(Author)](https://github.com/yueb17/DLTH) | Image Classification | 2022 |
| 15 | [Recent Advances on Neural Network Pruning at Initialization](https://arxiv.org/abs/2103.06460)| IJCAI | `W`| - | [PyTorch(Author)](https://github.com/mingsun-tse/smile-pruning) | Image Classification | 2022 |
| 16 | [What’s Hidden in a Randomly Weighted Neural Network?](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ramanujan_Whats_Hidden_in_a_Randomly_Weighted_Neural_Network_CVPR_2020_paper.pdf)| CVPR | `W`| - | [PyTorch(Author)](https://github.com/allenai/hidden-networks) | Image Classification | 2020 |
| 17 | [Finding trainable sparse networks through Neural Tangent Transfer](https://arxiv.org/abs/2006.08228)| ICML | `W`| - | [PyTorch(Author)](https://github.com/fmi-basel/neural-tangent-transfer) | Image Classification | 2020 |
| 18 | [The Unreasonable Effectiveness of Random Pruning: Return of the Most Naive Baseline for Sparse Training](https://openreview.net/forum?id=VBZJ_3tz-t) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/Random_Pruning) | Image Classification | 2022 |







#### 1.1.2 Pruning During Training
| No. | Title | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Dynamic Sparse Training: Find Effective Sparse Network from Scratch with Trainable Masked Layers](https://arxiv.org/abs/2005.06870)| ICLR | `NF`| DST | [PyTorch(Author)](https://github.com/junjieliu2910/DynamicSparseTraining) | Image Classification | 2020 |
| 02 | [Learning Structured Sparsity in Deep Neural Networks](https://proceedings.neurips.cc/paper/2016/file/41bfd20a38bb1b0bec75acf0845530a7-Paper.pdf)| NIPS | `FC`| SSL | [Caffe(Author)](https://github.com/wenwei202/caffe/tree/scnn) | Image Classification | 2016 |
| 03 | [Learning Efficient Convolutional Networks through Networks Slimming](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)| ICCV | `C`| Slimming | [Lua(Author)](https://github.com/liuzhuang13/slimming) | Image Classification | 2017 |
| 04 | [Rethinking the Smaller-Norm-Less-Informative Assumption in Channel Pruning of Convolution Layers](https://arxiv.org/abs/1802.00124) | ICLR | `F` | - | [TensorFlow(Author)](https://github.com/bobye/batchnorm_prune) [PyTorch(3rd)](https://github.com/jack-willturner/batchnorm-pruning) | Image Classification&Segmentation | 2018 |
| 05 | [Data-Driven Sparse Structure Selection for Deep Neural Networks](https://arxiv.org/abs/1707.01213) | ECCV | `F` | SSS | [MXNet(Author)](https://github.com/TuSimple/sparse-structure-selection) | Image Classification | 2018 |
| 06 | [Compressing Convolutional Neural Networks via Factorized Convolutional Filters](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Compressing_Convolutional_Neural_Networks_via_Factorized_Convolutional_Filters_CVPR_2019_paper.pdf) | CVPR | `F` | FCF | [PyTorch(Author)](https://github.com/IIGROUP/CNN-FCF) | Image Classification | 2019 |
| 07 | [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gordon_MorphNet_Fast__CVPR_2018_paper.pdf) | CVPR | `L` | MorphNet | [PyTorch(Author)](https://github.com/google-research/morph-net) | Image Classification | 2018 |
| 08 | [Learning the Number of Neurons in Deep Networks](https://arxiv.org/abs/1611.06321) | NIPS | `N` | - | - | Image Classification | 2016 |
| 09 | [Learning Sparse Neural Networks Through $L_0$ Regularization](https://openreview.net/pdf?id=H1Y8hhg0b) | ICLR | `FN` | - | [PyTorch(Author)](https://github.com/AMLab-Amsterdam/L0_regularization) | Image Classification | 2018 |
|10 | [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866)  | IJCAI   | `F`  | SFP | [PyTorch(Author)](https://github.com/he-y/soft-filter-pruning) | Image Classification | 2018 |
| 11 | [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250) | CVPR | `F` | FPGM | [PyTorch(Author)](https://github.com/he-y/filter-pruning-geometric-median) | Image Classification | 2019 | 
| 12 | [Variational Convolutional Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.html) | CVPR | `F` | VCP | - | Image Classification | 2019 | 
| 13 | [Rigging the Lottery:Making All Tickets Winners](https://arxiv.org/abs/1911.11134) | ICML | `W` | RigL | [PyTorch(Author)](https://github.com/google-research/rigl) | Image Classification | 2019 | 
| 14 | [NeST: A Neural Network Synthesis Tool Based on a Grow-and-Prune Paradigm](https://arxiv.org/abs/1711.02017) | arXiv | `N` | NeST | - | Image Classification | 2019 |
| 15 | [Sparse Training via Boosting Pruning Plasticity with Neuroregeneration](https://papers.nips.cc/paper/2021/hash/5227b6aaf294f5f027273aebf16015f2-Abstract.html)  | NeurIPS | `WF` | GraNet | [PyTorch(Author)](https://github.com/VITA-Group/GraNet) | Image Classification | 2021 |
| 16 | [DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation](https://arxiv.org/abs/2004.02164) | ECCV | `F` | DSA | [PyTorch(Author)](https://github.com/walkerning/differentiable-sparsity-allocation) | Image Classification | 2020 | 
| 17 | [Parameter Efficient Training of Deep Convolutional Neural Networks by Dynamic Sparse Reparameterization](https://arxiv.org/abs/1902.05967) | ICML | `W` | DSR | [PyTorch(Not Available)](https://github.com/IntelAI/dynamic-reparameterization) | Image Classification | 2019 | 
| 18 | [Sparse Networks from Scratch: Faster Training without Losing Performance](https://arxiv.org/abs/1907.04840) | arXiv | `W` | SM | [PyTorch(Author)](https://github.com/TimDettmers/sparse_learning) | Image Classification | 2019 | 
| 19 | [Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science](https://arxiv.org/pdf/1707.04780.pdf) | Nature Communication | `W&P` | SET | - | Image Classification | 2018 | 
| 20 | [Online Filter Clustering and Pruning for Efficient Convets](https://arxiv.org/abs/1905.11787) | arXiv | `W` | - | - | Image Classification | 2019 | 
| 21 | [Dynamic Model Pruning with Feedback](https://openreview.net/forum?id=SJem8lSFwB) | ICLR | `WF` | DPF | [PyTorch(3rd)](https://github.com/INCHEON-CHO/Dynamic_Model_Pruning_with_Feedback) | Image Classification | 2020 |  
| 22 | [Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training](http://proceedings.mlr.press/v139/liu21y/liu21y.pdf) | ICML | `W` | ITOP | [PyTorch(Anthor)](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization) | Image Classification | 2021 |  
| 23 | [Dense for the Price of Sparse: Improved Performance of Sparsely Initialized Networks via a Subspace Offset](http://proceedings.mlr.press/v139/price21a/price21a.pdf) | ICML | `W` | DCTpS | [PyTorch(Anthor)](https://github.com/IlanPrice/DCTpS) | Image Classification | 2021 | 
| 24 | [Selfish Sparse RNN Training](http://proceedings.mlr.press/v139/liu21p/liu21p.pdf) | ICML | `W` | SNT-ASGD |[PyTorch(Anthor)](https://github.com/Shiweiliuiiiiiii/Selfish-RNN) | Language Modeling | 2021 |
| 25 | [Deep ensembling with no overhead for either training or testing: The all-round blessings of dynamic sparsity](https://openreview.net/pdf?id=RLtqs6pzj1-) | ICLR | `W` | FreeTickets |[PyTorch(Anthor)](https://github.com/VITA-Group/FreeTickets) | Image Classification | 2022 |
| 26 | [Training Adversarially Robust Sparse Networks via Bayesian Connectivity Sampling](http://proceedings.mlr.press/v139/ozdenizci21a/ozdenizci21a.pdf) | ICML | `W` | - |[PyTorch(Anthor)](https://github.com/IGITUGraz/SparseAdversarialTraining) | Adversarial Robustness | 2021 |
| 27 | [Dynamic Sparse Training for Deep Reinforcement Learning](https://arxiv.org/pdf/2106.04217.pdf) | IJCAI | `W` | - |[PyTorch(Anthor)](https://github.com/GhadaSokar/Dynamic-Sparse-Training-for-Deep-Reinforcement-Learning) | Continuous Control | 2022 |
| 28 | [The State of Sparse Training in Deep Reinforcement Learning.](https://proceedings.mlr.press/v162/graesser22a/graesser22a.pdf) | ICML | `W` | - |[Tensorflow(Anthor)](github.com/google-research/rigl/tree/master/rigl/rl) | Continuous Control | 2022 |
| 29 | [MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning](https://arxiv.org/abs/1903.10258) | ICCV | `F` | MetaPruning | [PyTorch(Author)](https://github.com/liuzechun/MetaPruning) | Image Classification | 2019 |
| 30 | [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/abs/2003.13683) | ECCV | `F` | DHP | [PyTorch(Author)](https://github.com/ofsoundof/dhp) | Image Classification&Super-resolution&Denoising | 2019 |
| 31 | [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](https://arxiv.org/abs/1909.12778) | NeurIPS  | `W` | GSM | [PyTorch(Author)](https://github.com/DingXiaoH/GSM-SGD)  | Image Classification | 2019 |
| 32 | [Pruning Filter in Filter](https://arxiv.org/abs/2009.14410) | NeurIPS | `Other` | SWP | [PyTorch(Author)](https://github.com/fxmeng/Pruning-Filter-in-Filter)    | Image Classification | 2020 |
| 33 | [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717) | NeurIPS | `F` | TAS | [PyTorch(Author)](https://github.com/D-X-Y/NAS-Projects) | Image Classification | 2019 |
| 34 | [SuperTickets: Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning](https://arxiv.org/abs/2207.03677) | ECCV | `W` | SuperTickets | [PyTorch(Author)](https://github.com/GATECH-EIC/SuperTickets) | Image Classification&Object Detection&Human Pose Estimation | 2022 |
| 35 | [Exploring Sparsity in recurrent neural networks](https://arxiv.org/abs/1704.05119) | ICLR | `W` | - | [PyTorch](https://github.com/puhsu/pruning) | Speech Recognition | 2017 |
| 36 | [Training Neural Networks with Fixed Sparse Masks](https://arxiv.org/abs/2111.09839) | NeurIPS | `W` | - | [PyTorch(Author)]( https://github.com/varunnair18/FISH) | Image Classification | 2021 |
| 37 | [Deep Rewiring: Training very Sparse Deep Networks](https://arxiv.org/pdf/1711.05136.pdf) | ICLR | `W` | - | - | Image Classification&Audio | 2018 |










#### 1.1.3 Pruning After Training
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://arxiv.org/abs/1903.09291) | CVPR | `F` | GAL | [PyTorch(Author)](https://github.com/ShaohuiLin/GAL) | Image Classification | 2019 |
| 02 | [Winning the Lottery with Continuous Sparsification](https://arxiv.org/abs/1912.04427) | NeurIPS | `F` | CS | [PyTorch(Author)](https://github.com/lolemacs/continuous-sparsification) | Image Classification | 2019 |
| 03 | [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](https://arxiv.org/abs/1904.03837) | CVPR | `F` | C-SGD | [Tensorflow(Author)](https://github.com/ShawnDing1994/Centripetal-SGD) |Image Classification | 2019 |
| 04 | [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) | ICCV&TPAMI | `F` | ThiNet | [Caffe(Author)](https://github.com/Roll920/ThiNet), [PyTorch(3rd)](https://github.com/tranorrepository/reprod-thinet) | Image Classification | 2017&2019 |
| 05 | [Channel pruning for accelerating very deep neural networks](https://arxiv.org/abs/1707.06168) | ICCV | `C` | - | [Caffe(Author)](https://github.com/yihui-he/channel-pruning) |Image Classification&Object Detection | 2017 |
| 06 | [NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/abs/1711.05908) | CVPR | `NC` | NISP | - | Image Classification | 2018 |
| 07 | [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) | ICLR | `F` | - | [TensorFlow(3rd)](https://github.com/Tencent/PocketFlow#channel-pruning) | Image Classification | 2017 |
| 08 | [Discrimination-aware Channel Pruning for Deep Neural Networks](https://arxiv.org/abs/1810.11809) | NeurIPS | `C` | DCP | [TensorFlow(Author)](https://github.com/SCUT-AILab/DCP)  | Image Classification | 2018 |
| 09 | [Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1909.08174) | NeurIPS | `F` | Gate Decorator | [PyTorch(Author)](https://github.com/youzhonghui/gate-decorator-pruning) | Image Classification&Semantic Segmentation | 2019 |
| 10 | [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) | ICLR    | `F`  | PFEC | [PyTorch(3rd)](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet/l1-norm-pruning) | Image Classification | 2017 |
| 11 | [Neural Network Pruning with Residual-Connections and Limited-Data](https://arxiv.org/abs/1911.08114) | CVPR | `C` | CURL | [PyTorch(Author)](https://github.com/Roll920/CURL) | Image Classification | 2020 |
| 12 | [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/abs/2002.10179) | CVPR | `F` | HRank | [Pytorch(Author)](https://github.com/lmbxmu/HRank) | Image Classification | 2020 |
| 13 | [Importance Estimation for Neural Network Pruning](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf) | CVPR | `F` | Taylor-FO-BN |[PyTorch(Author)](https://github.com/NVlabs/Taylor_pruning) | Image Classification | 2019 |  
| 14 | [Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework](https://arxiv.org/abs/2010.04879) | ICML | `F` | - | - | Image Classification | 2021 | 
| 15 | [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2020/html/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.html) | CVPR | `F` | LFPC | - | Image Classification | 2020 | 
| 16 | [Neural Pruning via Growing Regularization](https://openreview.net/pdf?id=o966_Is_nPA) | ICLR | `WF` | Greg | - | Image Classification | 2021 |
| 17 | [Trainability Preserving Nueral Structured Pruning](https://openreview.net/pdf?id=AZFvpnnewr) | ECCV | `F` | TPP | [Pytorch(Author)](https://github.com/MingSun-Tse/TPP) | Image Classification | 2022 |
| 18 | [Optimal Brain Damage](https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) | NIPS | `W` | OBD | - | Image Classification | 1989 |
| 19 | [Second Order Derivatives for Network Pruning: Optimal Brain Surgeon](https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf) | NIPS | `W` | OBS | - | Image Classification | 1992 |
| 20 | [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) | ICLR **(Best)** | `W`  | - |[Caffe(Author)](https://github.com/songhan/Deep-Compression-AlexNet) | Image Classification | 2016 |
| 21 | [The State of Sparsity in Deep Neural Networks]() | arXiv | `w`  | - |[TensorFlow(Author)](https://github.com/google-research/google-research/blob/master/state_of_sparsity/README.md) | Image Classification&machine translation | 2019 |
| 22 | [Auto-Balanced Filter Pruning for Efficient Convolutional Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/12262) | AAAI | `F`  | - | - | Image Classification | 2019 |
| 23 | [Reborn filters: Pruning convolutional neural networks with limited data](https://ojs.aaai.org/index.php/AAAI/article/view/6058) | AAAI | `F` | - | - | Image Classification | 2020 |
| 24 | | ICLR | `F` | - | [PyTorch(Author)](https://github.com/EkdeepSLubana/flowandprune) | Image Classification | 2021 |
| 25 | [Lottery Jackpot Exist in Pre-trained Models](https://arxiv.org/pdf/2104.08700.pdf) | arXiv | `W` | Jackpot | [PyTorch(Author)]（https://github.com/zyxxmu/lottery-jackpots） | Image Classification |2021 |
| 26 | [2PFPCE: Two-Phase Filter Pruning Based on Conditional Entropy](https://arxiv.org/pdf/1809.02220.pdf) | AAAI | `W` | 2PFPCE | - | Image Classification | 2018 |
| 27 | [Exploiting Kernel Sparsity and Entropy for Interpretable CNN Compression](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Exploiting_Kernel_Sparsity_and_Entropy_for_Interpretable_CNN_Compression_CVPR_2019_paper.pdf) | CVPR | `W` | KSE | [PyTorch(Author)](https://github.com/yuchaoli/KSE) | Image Classification |2019 |
| 28 | [AMC: Automl for model compression and acceleration on mobile devices](https://arxiv.org/abs/1802.03494) | ECCV | `F` | AMC | [TensorFlow(3rd)](https://github.com/Tencent/PocketFlow#channel-pruning) |  Image Classification | 2018 |
| 29 | [Towards Efficient Model Compression via Learned Global Ranking](https://arxiv.org/abs/1904.12368)| CVPR | `F` | LeGR | [Pytorch(Author)](https://github.com/cmu-enyac/LeGR) | Image Classification | 2020 |
| 30 | [Collaborative Channel Pruning for Deep Networks](http://proceedings.mlr.press/v97/peng19c.html) | ICML | `F` | CCP | - | Image Classification | 2019 |
| 31 | [ECC: Platform-Independent Energy-Constrained Deep Neural Network Compression via a Bilinear Regression Model](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_ECC_Platform-Independent_Energy-Constrained_Deep_Neural_Network_Compression_via_a_Bilinear_CVPR_2019_paper.pdf) | CVPR | `F` | ECC | [Pytorch(Author)](https://github.com/hyang1990/energy_constrained_compression) | Image Classification&Semantic Segmentation | 2019 |
| 32 | [Discrete Model Compression With Resource Constraint for Deep Neural Networks](http://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Discrete_Model_Compression_With_Resource_Constraint_for_Deep_Neural_Networks_CVPR_2020_paper.html) | CVPR | `F` | - | - | Image Classification | 2020 |
| 33 | [Network Pruning via Performance Maximization](https://openaccess.thecvf.com/content/CVPR2021/papers/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.pdf) | CVPR | `F` | NPPM | [Pytorch(Author)](https://github.com/gaosh/NPPM) | Image Classification | 2021 |
| 34 | [Operation-Aware Soft Channel Pruning using Differentiable Masks](https://arxiv.org/abs/2007.03938) | ICML| `F` | SCP | - | Image Classification | 2020 |
| 35 | [Towards Compact and Robust Deep Networks](https://arxiv.org/abs/1906.06110) | arXiv | `W` | - | - | Image Classification | 2020 |
| 36 | [HYDRA: Pruning Adversarially Robust Neural Networks](https://arxiv.org/abs/2002.10509) | NeurIPS | `W` | HYDRA | [PyTorch(Author)](https://github.com/inspire-group/hydra) | Adversarial Robustness | 2020 |
| 37 | [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization github](https://arxiv.org/abs/1905.04748) | ICML | `F` | AOFP | [Pytorch(Author)](https://github.com/DingXiaoH/AOFP) | Image Classification | 2019 |
| 38 | [Channel Pruning via Automatic Structure Search](https://arxiv.org/abs/2001.08565) | IJCAI | `F` | ABC | [PyTorch(Author)](https://github.com/lmbxmu/ABCPruner) | Image Classification | 2020 |
| 39 | [Group Fisher Pruning for Practical Network Compression](https://arxiv.org/abs/2108.00708) | ICML | `F` | GFP | [PyTorch(Author)](https://github.com/jshilong/FisherPruning) | Image Classification&Object Detection | 2021 |
| 40 | [TransTailor: Pruning the Pre-trained Model for Improved Transfer Learning](https://arxiv.org/abs/2103.01542) | AAAI | `F` | TransTailor | - | Image Classification | 2021 |
| 41 | [Towards Compact ConvNets via Structure-Sparsity Regularized Filter Pruning](https://arxiv.org/abs/1901.07827) | TNNLS | `F` | SSR | [Caffe(Author)](https://github.com/ShaohuiLin/SSR) | Image Classification | 2019 |
| 42 | [Network Pruning That Matters: A Case Study on Retraining Variants](https://openreview.net/forum?id=Cb54AMqHQFP) | ICLR | `F` | - | [PyTorch(Author)](https://github.com/lehduong/NPTM) | Image Classification | 2021 | 
| 43 | [ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations](https://openreview.net/forum?id=xCxXwTzx4L1) | ICLR | `F` | ChipNet | [PyTorch(Author)](https://github.com/transmuteAI/ChipNet) |Image Classification | 2021 | 
| 44 | [SOSP: Efficiently Capturing Global Correlations by Second-Order Structured Pruning](https://openreview.net/forum?id=t5EmXZ3ZLR) | ICLR **(Spotlight)** | `F`     | SOSP | [PyTorch(Author)](https://github.com/boschresearch/sosp)(Releasing)  | Image Classification | 2022 | 
| 45 | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) | ICLR **(Best)** | `W` | LTH | [TensorFlow(Author)](https://github.com/google-research/lottery-ticket-hypothesis) | Image Classification | 2019 | 
| 46 | [Proving the Lottery Ticket Hypothesis for Convolutional Neural Networks](https://openreview.net/forum?id=Vjki79-619-) | ICML | `N` | - | - | - | 2020 |
| 47 | [Logarithmic Pruning is All You Need](https://arxiv.org/abs/2006.12156) | NeurIPS | `N` | - | - | - | 2020 |
| 48 | [Optimal Lottery Tickets via SUBSETSUM:Logarithmic Over-Parameterization is Sufficient](https://arxiv.org/abs/2006.07990) | NeurIPS | `N` | - |  [PyTorch(Author)](https://github.com/acnagle/optimal-lottery-tickets) |Image Classification | 2020 |
| 49 | [Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot?](https://openreview.net/pdf?id=WL7pr00_fnJ) | NeurIPS | `W` | - |  [PyTorch(Author)](https://github.com/boone891214/sanity-check-LTH) |Image Classification | 2021 |
| 50 | [Multi-Prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning A Randomly Weighted Network](https://openreview.net/forum?id=U_mat0b9iv) | ICLR | `W` | MPTs | [PyTorch(Author)](https://github.com/chrundle/biprop) | Image Classification | 2021 |
| 51 | [One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers](https://arxiv.org/abs/1906.02773) | NeurIPS | `W` | - | - | Image Classification | 2019 |
| 52 | [Long live the lottery: the existence of winning tickets in lifelong learning](https://arxiv.org/abs/1906.02773) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/Lifelong-Learning-LTH) | Image Classification | 2021 |
| 53 | [A Unified Lottery Ticket Hypothesis for Graph Neural Networks](https://arxiv.org/abs/2102.06790) | ICML | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/Unified-LTH-GNN) | Node Classification&Link Prediction | 2021 |
| 54 | [The Lottery Ticket Hypothesis for Pre-trained BERT Networks](https://arxiv.org/abs/2007.12223) | ICML | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/BERT-Tickets) | Language Modeling | 2021 |
| 55 | [When BERT Plays the Lottery, All Tickets Are Winning](https://arxiv.org/abs/2005.00561) | EMNLP | `W` | - | [PyTorch(Author)](https://github.com/sai-prasanna/bert-experiments) | Language Modeling | 2020 |
| 56 | [Playing the Lottery with Rewards and Multiple Languages: Lottery Tickets in RL and NLP](https://arxiv.org/abs/1906.02768) | ICLR | `W` | - | - | Classic Control&Atari Game | 2020 |
| 57 | [Playing Lottery Tickets with Vision and Language](https://arxiv.org/abs/2104.11832) | AAAI | `W` | - | - | Vision-and-Language | 2022 |
| 58 | [Validating the Lottery Ticket Hypothesis with Inertial Manifold Theory](https://papers.nips.cc/paper/2021/hash/fdc42b6b0ee16a2f866281508ef56730-Abstract.html)  | NeurIPS | `W`     | - | - | Image Classification | 2021 |
| 59 | [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://arxiv.org/abs/1905.01067) | NeurIPS | `W` | - | [TensorFlow(Author)](https://github.com/uber-research/deconstructing-lottery-tickets) | Image Classification | 2019 |
| 60 | [Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win](https://arxiv.org/pdf/2010.03533.pdf) | AAAI | `W` | - | [PyTorch(Author)](https://github.com/google-research/rigl/tree/master/rigl/rigl_tf2) | Image Classification | 2022 |
| 61 | [Sparse Transfer Learning via Winning Lottery Tickets](https://arxiv.org/abs/1905.07785) | arXiv | `W` | - | [PyTorch(Author)](https://github.com/rahulsmehta/sparsity-experiments) | Image Classification | 2019 |
| 62 | [How many winning tickets are there in one DNN?](https://arxiv.org/abs/2006.07014) | arXiv | `W` | - | - | Image Classification | 2020 |
| 63 | [CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization](https://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html)  | CVPR | `W` | CLIP-Q| - | Image Classification | 2018 |
| 64 | [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://arxiv.org/abs/2003.08935) | CVPR | `F` | Hinge | [PyTorch(Author)](https://github.com/ofsoundof/group_sparsity) | Image Classification | 2020 |
| 65 | [Towards Compact CNNs via Collaborative Compression](https://arxiv.org/abs/2105.11228) | CVPR | `F` | CC | [PyTorch(Author)](https://github.com/liuguoyou/Towards-Compact-CNNs-via-Collaborative-Compression) | Image Classification | 2021 |
| 66 | [NPAS: A Compiler-aware Framework of Unified Network Pruning andArchitecture Search for Beyond Real-Time Mobile Acceleration](https://arxiv.org/abs/2012.00596) | CVPR | `F` | NPAS | - | Image Classification | 2021 | 
| 67 | [GAN Compression: Efficient Architectures for Interactive Conditional GANs](https://arxiv.org/pdf/2003.08936.pdf) | arXiv | `C` | - | - | Image-to-Image Translation | 2021 |
| 68 | [Content-Aware GAN Compression](https://arxiv.org/abs/2104.02244) | CVPR | `F` | - | [PyTorch(Author)](https://github.com/lychenyoko/content-aware-gan-compression) | Image Generation, Image Projection, Image Editing | 
| 69 | [Dreaming to Prune Image Deraining Networks](https://openaccess.thecvf.com/content/CVPR2022/papers/Zou_Dreaming_To_Prune_Image_Deraining_Networks_CVPR_2022_paper.pdf) | CVPR | `F` | - | - | Image Deraining | 2022 |
| 70 | [Prune Your Model Before Distill It](https://arxiv.org/abs/2109.14960) | ECCV | `F` | - | [PyTorch(Author)](https://github.com/ososos888/prune-then-distill) | Image Classification | 2022 |
| 71 | [LadaBERT: Lightweight Adaptation of BERT through Hybrid Model Compression](https://arxiv.org/abs/2004.04124) | COLING | `W` | - | - | NLP(Sentiment Classification,Natural Language Inference,Pairwise Semantic Equivalence) | 2020 |
| 72 | [The Lottery Ticket Hypothesis for Object Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Girish_The_Lottery_Ticket_Hypothesis_for_Object_Recognition_CVPR_2021_paper.pdf) | CVPR | `W` | - | [PyTorch(Author)](https://github.com/Sharath-girish/LTH-ObjectRecognition) | Object Detection | 2021 |
| 73 | [Enabling Retrain-free Deep Neural Network Pruning Using Surrogate Lagrangian Relaxation](https://arxiv.org/abs/2012.10079) | IJCAI | `W` | - | - | Image Classification & Object Detection | 2021 |
| 74 | [Joint-DetNAS: Upgrade Your Detector with NAS, Pruning and Dynamic Distillation](https://arxiv.org/abs/2105.12971)| CVPR | `F`| Joint-DetNAS | - | Image Classification & Object Detection | 2021 |
| 75 | [Exploring Linear Relationship in Feature Map Subspace for ConvNets Compression](https://arxiv.org/abs/1803.05729)| arXiv | `F`| - | - | Object Detection&Human Pose Estimation | 2018 |
| 76 | [Train Large, Then Compress: Rethinking Model Size for Efficient Training and Inference of Transformers](https://arxiv.org/abs/2002.11794)| ICML | `W`| - | - | NLP | 2020 |
| 77 | [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878) | ICLRW| `W` | - | [TensorFlow(Author)](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/model_pruning) | NLP | 2018 |
| 78 | [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418) | ACL| `W` | - | [PyTorch(Author)](https://github.com/lena-voita/the-story-of-heads)| NLP | 2019 |
| 79 | [Towards Adversarial Robustness Via Compact Feature Representations](https://ieeexplore.ieee.org/document/9414696) | ICASSP| `N` | -  | [PyTorch(Author)](https://github.com/lena-voita/the-story-of-heads)| Adversarial Robustness | 2021 |
| 80 | [Pruning Redundant Mappings in Transformer Models via Spectral-Normalized Identity Prior](https://arxiv.org/abs/2010.01791) | EMNLP| `Other` | - | - | NLP | 2020 |
| 81 | [Reweighted Proximal Pruning for Large-Scale Language Representation](http://arxiv.org/abs/1909.12486) | arXiv| `Other`  | -  | - | NLP | 2019 |
| 82 | [Efficient Transformer-based Large Scale Language Representations using Hardware-friendly Block Structured Pruning](https://arxiv.org/abs/2009.08065) | EMNLP| `Other`| - | - | NLP | 2019 |
| 83 | [EarlyBERT: Efficient BERT training via early-bird lottery tickets](https://arxiv.org/abs/2101.00063) | ACL-IJCNLP| `Other` | EarlyBERT | [PyTorch(Author)](https://github.com/VITA-Group/EarlyBERT) | NLP | 2021 |
| 84 | [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683) | NeurIPS | `W` | - | [PyTorch(Author)](https://github.com/huggingface/block_movement_pruning) | NLP | 2020 | 
| 85 | [Audio Lottery: Speech Recognition made ultra-lightweight, transferable, and noise-robust](https://openreview.net/pdf?id=9Nk6AJkVYB) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/VITA-Group/Audio-Lottery) | Speach Recognition | 2022 | 
| 86 | [PARP: Prune, Adjust and Re-Prune for Self-Supervised Speech Recognition](https://arxiv.org/abs/2106.05933) | NeurIPS | `W` | PARP | -| Speach Recognition | 2021 | 
| 87 | [Dynamic Sparsity Neural Networks for Automatic Speech Recognition](https://arxiv.org/abs/2005.10627) | ICASSP | `W` |- | -| Speach Recognition | 2021 |
| 88 | [On the Predictability of Pruning Across Scales](https://arxiv.org/abs/2006.10621) | ICML | `W` | -| - | Image Classification | 2021 |
| 89 | [How much pre-training is enough to discover a good subnetwork?](https://arxiv.org/abs/2108.00259) | arXiv | `W` | -| - | Image Classification | 2021 |
| 90 | [On the Transferability of Winning Tickets in Non-Natural Image Datasets](https://arxiv.org/pdf/2005.05232.pdf) | arXiv | `W` | -| - | Image Classification | 2020 |
| 91 | [The Lottery Tickets Hypothesis for Supervised and Self-supervised Pre-training in Computer Vision Models](https://arxiv.org/pdf/2005.05232.pdf) | CVPR | `W` | -| [PyTorch(Author)](https://github.com/VITA-Group/CV_LTH_Pre-training) | Image Classification | 2021 |
| 92 | [DepGraph: Towards Any Structural Pruning](https://www.ijcai.org/proceedings/2018/0336.pdf) | CVPR | DepGraph | - | [PyTorch(Author)](https://github.com/VainF/Torch-Pruning)  | CV/NLP | 2023 |
| 93 | [How Well Do Sparse ImageNet Models Transfer?](https://arxiv.org/abs/2111.13445) | CVPR | `W` | - |  [PyTorch(Author)](https://github.com/ISTDASLab/sparse-imagenet-transfer) | Image Classification&Object Detection | 2022 |
| 94 | [The Elastic Lottery Ticket Hypothesis](https://papers.nips.cc/paper/2021/hash/dfccdb8b1cc7e4dab6d33db0fef12b88-Abstract.html)  | NeurIPS | `W` | E-LTH | [PyTorch(Author)](https://github.com/VITA-Group/ElasticLTH) | Image Classification | 2021 | 
| 95 | [Why Lottery Ticket Wins? A Theoretical Perspective of Sample Complexity on Sparse Neural Networks](https://papers.nips.cc/paper/2021/hash/15f99f2165aa8c86c9dface16fefd281-Abstract.html) | NeurIPS | `W` | - | - | Image Classification | 2021 | 
| 96 | [Exploring Lottery Ticket Hypothesis in Spiking Neural Networks](https://arxiv.org/abs/2207.01382) | ECCV | `W` | ET | [PyTorch(Author)](https://github.com/intelligent-computing-lab-yale/exploring-lottery-ticket-hypothesis-in-snns) | Image Classification | 2022 |
| 97 | [Graph Pruning for Model Compression](https://arxiv.org/abs/1911.09817) | Applied Intelligence | `W` | GraphPruning | - | Image Classification | 2022 |
| 98 | [Accelerating Convolutional Networks via Global & Dynamic Filter Pruning](https://www.ijcai.org/proceedings/2018/0336.pdf) | IJCAI | GDP | `F` | -  | Image Classification | 2018 |
| 99 | [DMCP: Differentiable Markov Channel Pruning for Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_DMCP_Differentiable_Markov_Channel_Pruning_for_Neural_Networks_CVPR_2020_paper.pdf) | CVPR | DMCP | `C` | -  | Image Classification | 2020 |






#### 1.1.4 Pruning in Early Training
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://arxiv.org/abs/1912.05671) | ICML | `W` | - | - | Image Classification | 2020 |
| 02 | [When To Prune? A Policy Towards Early Structural Pruning](https://openaccess.thecvf.com/content/CVPR2022/html/Shen_When_To_Prune_A_Policy_Towards_Early_Structural_Pruning_CVPR_2022_paper.html) | CVPR | `F` | PaT | - | Image Classification | 2022 |
| 03 | [Drawing Early-Bird Tickets: Towards More Efficient Training of Deep Networks](https://arxiv.org/abs/1909.11957) | ICLR | `W` | - | [PyTorch(Author)](https://github.com/GATECH-EIC/Early-Bird-Tickets) | Image Classification | 2020 |

### 1.2 Dynamic Pruning
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Channel Gating Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2017/file/a51fb975227d6640e4fe47854476d133-Paper.pdf) | NeurIPS | `F` | RNP | - | Image Classification | 2017 |
| 02 | [Channel Gating Neural Networks](https://arxiv.org/abs/1805.12549) | NeurIPS | `C` | CGNet | [PyTorch(Author)](https://github.com/cornell-zhang/dnn-gating) | Image Classification | 2019 |
| 03 | [Dynamic Dual Gating Neural Networks](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Dynamic_Dual_Gating_Neural_Networks_ICCV_2021_paper.pdf) | ICCV | `C` | DGNet | [PyTorch(Author)](https://github.com/lfr-0531/DGNet) | Image Classification | 2021 |
| 04 | [Manifold Regularized Dynamic Network Pruning](https://arxiv.org/abs/2103.05861) | CVPR | `F` | ManiDP |  [PyTorch(Author)](https://github.com/huaweinoah/Pruning/tree/master/ManiDP) | Image Classification | 2021 |
| 05 | [Dynamic Channel Pruning: Feature Boosting and Suppression](https://arxiv.org/pdf/1810.05331.pdf) | ICLR | `C` | FBS | [PyTorch(Author)](https://github.com/YOUSIKI/PyTorch-FBS) | Image Classification | 2019 |
| 06 | [Frequency-Domain Dynamic Pruning for Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2018/file/a9a6653e48976138166de32772b1bf40-Paper.pdf) | NeurIPS | `F` | FDNP | - | Image Classification | 2019 |
| 07 | [Fire Together Wire Together: A Dynamic Pruning Approach With Self-Supervised Mask PredictionFire Together Wire Together: A Dynamic Pruning Approach With Self-Supervised Mask Prediction](https://openaccess.thecvf.com/content/CVPR2022/html/Elkerdawy_Fire_Together_Wire_Together_A_Dynamic_Pruning_Approach_With_Self-Supervised_CVPR_2022_paper.html) | CVPR| `F` | - | - | Image Classification | 2019 |


## 2. Learning and Pruning

### 2.1 Continual learning
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| 01 | [Learning Bayesian Sparse Networks With Full Experience Replay for Continual Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Yan_Learning_Bayesian_Sparse_Networks_With_Full_Experience_Replay_for_Continual_CVPR_2022_paper.html)| CVPR | SNCL | - | Image Classification | 2022 |  
| 02 | [Continual Prune-and-Select: Class-Incremental Learning with SPecialized Subnetworks](https://arxiv.org/pdf/2208.04952.pdf)| Applied Intelligence | - | [PyTorch(Author)]( https://github.com/adekhovich/continual_prune_and_select) | Image Classification | 2023 |

### 2.2 Contrastive learning
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| 01 | [Studying the impact of magnitude pruning on contrastive learning methods](https://arxiv.org/pdf/2207.00200.pdf) | ICML | - | [PyTorch(Author)](https://github.com/FraCorti/Studying-the-impact-of-magnitude-pruning-on-contrastive-learning-methods) | Image Classification | 2020 |

### 2.3 Federated learning
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|
| 01 | [FedDUAP: Federated Learning with Dynamic Update and Adaptive Pruning Using Shared Data on the Server](https://arxiv.org/pdf/2204.11536.pdf) | IJCAI | FedDUAP | - | Image Classification | 2020 |
| 02 | [Model Pruning Enables Efficient Federated Learning on Edge Devices](https://arxiv.org/pdf/1909.12326.pdf) | TNNLS | - | [PyTorch(Author)](https://github.com/jiangyuang/PruneFL) | Image Classification | 2022 |


## 3. Application

### 3.1 Computer Vision
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [SuperTickets: Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning](https://arxiv.org/abs/2207.03677) | ECCV | [PyTorch(Author)](https://github.com/GATECH-EIC/SuperTickets) | Image Classification&Object Detection&Human Pose Estimation | 2022 |
| 02 | [Training Neural Networks with Fixed Sparse Masks](https://arxiv.org/abs/2111.09839) | NeurIPS | [PyTorch(Author)]( https://github.com/varunnair18/FISH) | Image Classification | 2021 |
| 03 | [Deep Rewiring: Training very Sparse Deep Networks](https://arxiv.org/pdf/1711.05136.pdf) | ICLR | - | Image Classification&Audio | 2018 |
| 04 | [Co-Evolutionary Compression for Unpaired Image Translation](https://arxiv.org/pdf/1907.10804.pdf) | ICCV | [PyTorch(Author)](https://github.com/yehuitang/Pruning) | Image Style Translation | 2019 |
| 05 | [Content-Aware GAN Compression](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Content-Aware_GAN_Compression_CVPR_2021_paper.pdf) | CVPR |  [PyTorch(Author)](https://github.com/lychenyoko/content-aware-gan-compression) | Image Style Translation | 2021 |
| 06 | [Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space](https://openaccess.thecvf.com/content/CVPR2022/papers/Chavan_Vision_Transformer_Slimming_Multi-Dimension_Searching_in_Continuous_Optimization_Space_CVPR_2022_paper.pdf) | CVPR | [PyTorch(Author)](https://github.com/Arnav0400/ViT-Slim) | Image Classification&Audio | 2022 |


### 3.2 Natural Language Processing

| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/pdf/2204.09656.pdf) | NeurIPS | [PyTorch(Author)](https://github.com/WoosukKwon/retraining-free-pruning) | Natural Language Understanding | 2022 |
| 02 | [The Lottery Ticket Hypothesis for Pre-trained BERT Networks](https://arxiv.org/abs/2007.12223) | ICML | [PyTorch(Author)](https://github.com/VITA-Group/BERT-Tickets) | Language Modeling | 2021 |
| 03 | [When BERT Plays the Lottery, All Tickets Are Winning](https://arxiv.org/abs/2005.00561) | EMNLP | [PyTorch(Author)](https://github.com/sai-prasanna/bert-experiments) | Language Modeling | 2020 |

### 3.3 Audio Signal Processing
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Exploring Sparsity in recurrent neural networks](https://arxiv.org/abs/1704.05119) | ICLR | [PyTorch](https://github.com/puhsu/pruning) | Speech Recognition | 2017 |
| 02 | [Deep Rewiring: Training very Sparse Deep Networks](https://arxiv.org/pdf/1711.05136.pdf) | ICLR | - | Image Classification&Audio | 2018 |


## 4. Survey of Pruning
| No. | Title   | Venue | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:----:|:----:|:----:|:----:|
| 01 | [Pruning Algorithms-A Survey](https://ieeexplore.ieee.org/document/248452) | IEEE Transactions on Neural Networks | - | Image Classification | 1993 |
| 02 | [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://arxiv.org/abs/1703.09039) | arXiv | - | Image Classification | 2017 |
| 03 | [Recent advances in efficient computation of deep convolutional neural networks](https://arxiv.org/pdf/1802.00939.pdf) | arXiv | - | - | 2018 |
| 04 | [The State of Sparsity in Deep Neural Networks](https://arxiv.org/abs/1902.09574) | arXiv | [PyTorch(Author)](https://github.com/google-research/google-research/blob/master/state_of_sparsity/README.md) | Image Classification&machine translation | 2019 |
| 05 | [Convolutional Neural Network Pruning: A Survey](https://ieeexplore.ieee.org/document/9189610) | CCC | - | - | 2020 |
| 06 | [What is the State of Neural Network Pruning?](https://arxiv.org/pdf/2003.03033.pdf) | MLSys | - | - | 2020 |
| 07 | [A comprehensive survey on model compression and acceleration](https://link.springer.com/article/10.1007/s10462-020-09816-7) | Artificial Intelligence Review | - | - | 2020 |
| 08 | [A Survey on Deep Neural Network Compression: Challenges, Overview, and Solutions](https://arxiv.org/pdf/2010.03954.pdf) | arXiv | - | - | 2020 |
| 09 | [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/pdf/1710.09282.pdf) | arXiv | - | - | 2020 |
| 10 | [An Survey of Neural Network Compression](https://arxiv.org/pdf/2006.03669.pdf) | arXiv | - | - | 2020 |
| 11 | [Model Compression and Hardware Acceleration for Neural Networks: A Comprehensive Survey](https://ieeexplore.ieee.org/document/9043731) | IEEE | - | - | 2020 |
| 12 | [Pruning Algorithms to Accelerate Convolutional Neural Networks for Edge Applications: A Survey](https://arxiv.org/pdf/2005.04275.pdf) | arXiv | - | Image Classification | 2020 |
| 13 | [Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](https://arxiv.org/abs/2102.00554) | JMLR | - | Image Classification | 2021 |
| 14 | [Dynamic Neural Networks: A Survey](https://arxiv.org/pdf/2102.04906.pdf) | arXiv | - | - | 2021 |
| 15 | [Pruning and Quantization for Deep Neural Network Acceleration: A Survey](https://arxiv.org/pdf/2101.09671.pdf) | Neurocomputing | - | Image Classification | 2021 |
| 16 | [Recent Advances on Neural Network Pruning at Initialization](https://arxiv.org/pdf/2103.06460.pdf) | IJCAI | - | CV&NLP | 2022 |
| 17 | [A Survey on Efficient Convolutional Neural Networks and Hardware Acceleration](https://arxiv.org/pdf/2103.06460.pdf) | Electronics | - | - | 2022 |
| 18 | [Dimensionality Reduced Training by Pruning and Freezing Parts of a Deep Neural Network, a Survey](https://arxiv.org/pdf/2205.08099.pdf) | arXiv | - | Image Classification | 2022 |
| 19 | [A Survey on Dynamic Neural Networks for Natural Language Processing](https://arxiv.org/pdf/2202.07101.pdf) | arXiv | - | NLP | 2023 |
| 20 | [Why is the State of Neural Network Pruning so Confusing? On the Fairness, Comparison Setup, and Trainability in Network Pruning](https://arxiv.org/pdf/2301.05219.pdf) | arXive | [PyTorch(Author)](https://github.com/MingSun-Tse/Why-the-State-of-Pruning-So-Confusing) | Image Classification | 2023 |
| 21 | [Structured Pruning for Deep Convolutional Neural Networks: A survey](https://arxiv.org/pdf/2303.00566.pdf) | arXive | - | CV&NLP | 2023 |


## 5. Other Works
| No. | Title   | Venue | Algorithm Name | Code | APP | Year |
|:----:|:--------------------------------------------------------------------------------------------------------------------------------:|:-----:|:-------:|:----:|:----:|:----:|
| 01 | [Are All Layers Created Equal?](https://arxiv.org/abs/1902.01996) | JMLR | - | - | Image Classification | 2022 |
| 02 | [Is Pruning Compression?: Investigating Pruning Via Network Layer Similarity](https://openaccess.thecvf.com/content_WACV_2020/papers/Blakeney_Is_Pruning_Compression_Investigating_Pruning_Via_Network_Layer_Similarity_WACV_2020_paper.pdf) | WACV | - | - | Image Classification | 2020 |
| 03 | [A Gradient Flow Framework For Analyzing Network Pruning](https://openreview.net/forum?id=rumv7QmLUue) | ICLR | - | [PyTorch(Author)](https://github.com/EkdeepSLubana/flowandprune) | Image Classification | 2021 |


## Other Useful Links
https://github.com/airaria/TextPruner  
https://zhuanlan.zhihu.com/p/130645948  

## Acknowledgements
We not only thank for the articles cited in our survey, but also thank the following repos' authors.   
https://github.com/he-y/awesome-Pruning/  
https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization  
https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression/blob/master/Paper/Pruning.md  



