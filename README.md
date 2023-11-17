# Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination

This is the code for the [paper](https://arxiv.org/abs/2311.02960) "Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination".

## Introduction

Over the past decade, deep learning has proven to be a highly effective tool for learning meaningful features from raw data. However, it remains an open question how deep networks perform hierarchical feature learning across layers. In this work, we attempt to unveil this mystery by investigating the structures of intermediate features. Motivated by our empirical findings that linear layers mimic the roles of deep layers in nonlinear networks for feature learning, we explore how deep linear networks transform input data into output by investigating the output (i.e., features) of each layer after training in the context of multi-class classification problems. Toward this goal, we first define metrics to measure within-class compression and between-class discrimination of intermediate features, respectively. Through theoretical analysis of these two metrics, we show that the evolution of features follows a simple and quantitative pattern from shallow to deep layers when the input data is nearly orthogonal and the network weights are minimum-norm, balanced, and approximate low-rank: Each layer of the linear network progressively compresses within-class features at a geometric rate and discriminates between-class features at a linear rate with respect to the number of layers that data have passed through. To the best of our knowledge, this is the first quantitative characterization of feature evolution in hierarchical representations of deep linear networks. Empirically, our extensive experiments not only validate our theoretical results numerically but also reveal a similar pattern in deep nonlinear networks which aligns well with recent empirical studies. Moreover, we demonstrate the practical implications of our results in transfer learning.

## Environment

Please check the requirement.sh

## Check Assumption 2 in the paper (Figure 6)

~~~python
$ cd assumption
$ sh submit_job_all.sh
~~~

Then please refer to `assumption.ipynb` for the assumption validation demo.

## Verify our theorem (Figure 4)

Please refer to `assumption/theory.ipynb` for more details and setups.

## Reproduce Figure 1, 5

For the Hybrid network, run
~~~python
$ sh submit_job.sh 5000 7 3 1024 cifar10 100.0
~~~

For the MLP network, run
~~~python
$ sh submit_job.sh 5000 7 7 1024 cifar10 1.0
~~~

After training, please refer to `hybrid_mlp.ipynb` for the postprocessing and visualization.

## Citation and reference 
For technical details and full experimental results, please check [our paper](https://arxiv.org/abs/2311.02960).
```
@article{wang2023understanding,
  title={Understanding Deep Representation Learning via Layerwise Feature Compression and Discrimination},
  author={Wang, Peng and Li, Xiao and Yaras, Can and Zhu, Zhihui and Balzano, Laura and Hu, Wei and Qu, Qing},
  journal={arXiv preprint arXiv:2311.02960},
  year={2023}
}
```
