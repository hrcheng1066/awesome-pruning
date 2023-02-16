# awesome-pruning

## Table of Contents

- [Type of Pruning](#type-of-pruning)

- [Pruning Before Training](#pruning-before-training)
- [Pruning During Training](#pruning-during-training)
- [Pruning After Training](#pruning-after-training)

### Type of Pruning

| Type        |`L`             | `F`            | `C`             | `N`             | `W`            | `P`        | `Other`     |
|:----------- |:--------------:|:--------------:|:---------------:|:---------------:|:--------------:|:----------:|:-----------:|
| Explanation | Layer pruning  | Filter pruning | Channel pruning |  Neuron pruning | Weight pruning |  Pioneer   | other types |

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
| 16 | [What’s Hidden in a Randomly Weighted Neural Network?](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ramanujan_Whats_Hidden_in_a_Randomly_Weighted_Neural_Network_CVPR_2020_paper.pdf)| CVPR | `W`| - | [PyTorch(Author)](https://github.com/allenai/hidden-networks) | Image Classification | 2020 |






### Pruning During Training
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
| 17 | [Parameter Efficient Training of Deep Convolutional Neural Networks by Dynamic Sparse Reparameterization](https://arxiv.org/abs/1902.05967) | ICML | `W` | DSR | [PyTorch(Author&Not Available)](https://github.com/IntelAI/dynamic-reparameterization) | Image Classification | 2019 | 
| 18 | [Sparse Networks from Scratch: Faster Training without Losing Performance](https://arxiv.org/abs/1907.04840) | arXiv | `W` | SM | [PyTorch(Author)](https://github.com/TimDettmers/sparse_learning) | Image Classification | 2019 | 
| 19 | [Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science](https://arxiv.org/pdf/1707.04780.pdf) | Nature Communication | `W&P` | SET | - | Image Classification | 2018 | 
| 20 | [Online Filter Clustering and Pruning for Efficient Convets](https://arxiv.org/abs/1905.11787) | arXiv | `W` | - | - | Image Classification | 2019 | 
| 21 | [Dynamic Model Pruning with Feedback](https://openreview.net/forum?id=SJem8lSFwB) | ICLR | `WF` | DPF | [PyTorch(3rd)](https://github.com/INCHEON-CHO/Dynamic_Model_Pruning_with_Feedback) | Image Classification | 2020 |  
| 22 | [Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training](http://proceedings.mlr.press/v139/liu21y/liu21y.pdf) | ICML | `W` | ITOP | [PyTorch(Anthor)](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization) | Image Classification | 2021 |  
| 23 | [Dense for the Price of Sparse: Improved Performance of Sparsely Initialized Networks via a Subspace Offset](http://proceedings.mlr.press/v139/price21a/price21a.pdf) | ICML | `W` | DCTpS | [PyTorch(Anthor)](https://github.com/IlanPrice/DCTpS) | Image Classification | 2021 | 
| 24 | [Selfish Sparse RNN Training](http://proceedings.mlr.press/v139/liu21p/liu21p.pdf) | ICML | `W` | SNT-ASGD |[PyTorch(Anthor)](https://github.com/Shiweiliuiiiiiii/Selfish-RNN) | Language Modeling | 2021 |
| 25 | [Deep ensembling with no overhead for either training or testing: The all-round blessings of dynamic sparsity](https://openreview.net/pdf?id=RLtqs6pzj1-) | ICLR | `W` | FreeTickets |[PyTorch(Anthor)](https://github.com/VITA-Group/FreeTickets) | Image Classification | 2022 |
| 26 | [Training Adversarially Robust Sparse Networks via Bayesian Connectivity Sampling](http://proceedings.mlr.press/v139/ozdenizci21a/ozdenizci21a.pdf) | ICML | `W` | - |[PyTorch(Anthor)](https://github.com/IGITUGraz/SparseAdversarialTraining) | Image Classification | 2021 |
| 27 | [Dynamic Sparse Training for Deep Reinforcement Learning](https://arxiv.org/pdf/2106.04217.pdf) | IJCAI | `W` | - |[PyTorch(Anthor)](https://github.com/GhadaSokar/Dynamic-Sparse-Training-for-Deep-Reinforcement-Learning) | Continuous Control | 2022 |
| 28 | [The State of Sparse Training in Deep Reinforcement Learning.](https://proceedings.mlr.press/v162/graesser22a/graesser22a.pdf) | ICML | `W` | - |[Tensorflow(Anthor)](github.com/google-research/rigl/tree/master/rigl/rl) | Continuous Control | 2022 |






### Pruning After Training
| No. | Title   | Venue | Type | Algorithm Name | Code | APP | Year |
|:----:|:-------------------------------------------------------------------------------------------------------------------------------- |:-----:|:-------:|:----:|:----:|:----:|:----:|
| 01 | [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://arxiv.org/abs/1903.09291) | CVPR | `F` | GAL | [PyTorch(Author)](https://github.com/ShaohuiLin/GAL) | Image Classification | 2019 |
| 02 | [Winning the Lottery with Continuous Sparsification](https://arxiv.org/abs/1912.04427) | NeurIPS | `F` | CS | [PyTorch(Author)](https://github.com/lolemacs/continuous-sparsification) | Image Classification | 2019 |
| 03 | [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](https://arxiv.org/abs/1904.03837) | CVPR | `F` | C-SGD | [Tensorflow(Author)](https://github.com/ShawnDing1994/Centripetal-SGD) |Image Classification | 2019 |
| 04 | [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) | ICCV | `F` | ThiNet | [Caffe(Author)](https://github.com/Roll920/ThiNet), [PyTorch(3rd)](https://github.com/tranorrepository/reprod-thinet) | Image Classification | 2019 |
| 05 | [Channel pruning for accelerating very deep neural networks](https://arxiv.org/abs/1707.06168) | ICCV | `C` | - | [Caffe(Author)](https://github.com/yihui-he/channel-pruning) |Image Classification&Object Detection | 2017 |
| 06 | [NISP: Pruning Networks using Neuron Importance Score Propagation](https://arxiv.org/abs/1711.05908) | CVPR | `NC` | NISP | - | Image Classification | 2018 |
| 07 | [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) | ICLR | `F` | - | [TensorFlow(3rd)](https://github.com/Tencent/PocketFlow#channel-pruning) | Image Classification | 2017 |
| 08 | [Discrimination-aware Channel Pruning for Deep Neural Networks](https://arxiv.org/abs/1810.11809) | NeurIPS | `C` | DCP | [TensorFlow(Author)](https://github.com/SCUT-AILab/DCP)  | Image Classification | 2018 |
| 09 | [Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1909.08174) | NeurIPS | `F` | Gate Decorator | [PyTorch(Author)](https://github.com/youzhonghui/gate-decorator-pruning) | Image Classification&Segmentation | 2019 |
| 10 | [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) | ICLR    | `F`  | PFEC | [PyTorch(3rd)](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet/l1-norm-pruning) | Image Classification | 2017 |
| 11 | [Neural Network Pruning with Residual-Connections and Limited-Data](https://arxiv.org/abs/1911.08114) | CVPR | CURL | `C` | [PyTorch(Author)](https://github.com/Roll920/CURL) | Image Classification | 2020 |
| 12 | [HRank: Filter Pruning using High-Rank Feature Map](https://arxiv.org/abs/2002.10179) | CVPR | `F` | HRank | [Pytorch(Author)](https://github.com/lmbxmu/HRank) | Image Classification | 2020 |
| 13 | [Importance Estimation for Neural Network Pruning](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf) | CVPR | `F` | Taylor-FO-BN |[PyTorch(Author)](https://github.com/NVlabs/Taylor_pruning) | Image Classification | 2019 |  
| 14 | [Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework](https://arxiv.org/abs/2010.04879) | ICML | `F` | - | - | Image Classification | 2021 | 
| 15 | [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2020/html/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.html) | CVPR | `F` | LFPC | - | Image Classification | 2020 |  










## Acknowledgements
https://github.com/he-y/awesome-Pruning/  
https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization  
https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression/blob/master/Paper/Pruning.md  



