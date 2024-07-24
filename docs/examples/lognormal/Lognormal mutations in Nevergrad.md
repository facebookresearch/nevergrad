We present a family of black-box optimization algorithms recently added in Nevergrad. We explain how they can be used for improving fake detection algorithms.
## Lognormal algorithm

Lognormal mutations have been added in Nevergrad.

Papers:
- https://dl.acm.org/doi/10.1145/1389095.1389298
- https://dl.acm.org/doi/abs/10.1145/2001576.2001699

## Extension to the discrete setting

Nevergrad contains an automatic adaptation of continuous algorithms to discrete settings and of discrete algorithms to continuous settings. It turns out that the lognormal algorithm performs quite well in some continuous cases:
- when the problem is very difficult (in particular, highly multimodal);
- when there is a good prior (a probability distribution) on the local of the optimum.
## Application to fake images

We observe the followings:
- No-box attacks are excellent for some fake detectors
- Square-Attacks are excellent for many fake detectors

However, both attacks can be detected by a ResNet trained on a few thousands images.

We observe that:
- Lognormal is not necessarily better than square-attacks, but it succeeds on many detectors.
- Lognormal attacks are not detected by detectors specialized on no-box or square-attacks
 
Therefore they must be included in a good defense mechanism.
We also note that using algorithms coming from generic black-box optimization algorithms creates a wide range of attacks, all of them have to be detected.

## Citation

```
@misc{lognormal_nevergrad,
  author = {Evergrad, N.},
  title = {Log-Normal mutations in Nevergrad},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/facebookresearch/nevergrad/optimization/optimizerlib.py}},
  commit = {6f952d4e7a883fa2bbf3fe0fd0e38b22bd953a1f}
}
```
