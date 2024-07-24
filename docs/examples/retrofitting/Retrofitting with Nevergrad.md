
Comments welcome here: https://www.facebook.com/groups/nevergradusers/
## What is retrofitting ?

Retrofitting is, in general, the addition of new features.

In machine learning, Retrofitting is typically the modification of a model using high-level information.

## When is classical gradient-based deep learning limited ?

Consider a model obtained by deep learning:
- MiDaS for depth estimation https://github.com/isl-org/MiDaS 
- Arnold for killing monsters at Doom https://github.com/glample/Arnold
- Code generation

Then, one can find non-differentiable criteria which are close to the expected figure of merit, but can not easily be used in a deep-learning optimization:
- In MiDaS, many use cases need excellent performance for an ad hoc loss function, e.g. the frequency of failing by more than X%. This loss function has a gradient zero almost everywhere.
- In Doom, we might consider a combination of kills per life and life expectancy. These criteria are not directly differentiable
- In Code generation, we might consider performance (speed) at test time.

## Nevergrad for retrofitting

Nevergrad does not use gradients, this is all the point of Nevergrad. Therefore we propose the following approach:
- Identify a small set of parameters, which have a big impact on the behavior of the model. For example, rescaling factors, or a single layer of a deep net.
- identify a loss function and data. It makes sense only if you use elements in this loss function that can not be used in a classical deep learning framework:
	- because it is not differentiable
	- because it can be computed only after running many time steps, as in reinforcement learning.

Then, use Nevergrad for optimizing these parameters using that loss function.

We got positive results for MiDaS, Arnold, Code generation, and others.

## Enhancements

Post in  https://www.facebook.com/groups/nevergradusers/ if you need code or help.


## Citation

```
@misc{retrofitting_nevergrad,
  author = {Evergrad, N.},
  title = {Retrofitting with Nevergrad},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/common/sphere.py}},
  commit = {01e1bc02e366783d37adfbf7af6326457977ef1f}
}



```
