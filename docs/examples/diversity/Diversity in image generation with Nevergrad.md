
Related paper: https://arxiv.org/abs/2310.12583 

Related plugin: https://github.com/mathuvu/sd-webui-diversity 

Comments welcome here: https://www.facebook.com/groups/nevergradusers/

## Diversity

Randomly generating points in a domain can be disappointing: sometimes ten random values are poorly distributed in the domain.

Therefore, quasi-random has been invented for improving the diversity (a.k.a reducing the redundancies).

For example,
- Van Der Corput sequences are more evenly distributed than random sequences in [0,1]
- Halton sequences are more evenly distributed than random points in the hypersquare
- Hammersley sequences are more evenly distributed than random points in the high-dimensional hypersquare.

## Diversity in high-dimensional contexts


Using Nevergrad, we create points which are more evenly distributed than random points in high-dimensional spheres or high-dimensional normal random variables.

For example:
`
`# x is a batch of 50 random latent vectors, with size 256x256 and 3 channels

`x = np.random.randn(50,256,256,3)`

`from nevergrad.common import sphere`

`derand_x = quasi_randomize(x)`

creates a point set with the same shape as x, and with some nice properties.

## Application to image generation

Many latent diffusion models use random latent variables.
They create an image, with as inputs:
- a randomly drawn normal latent tensor, e.g. 256x256x3
- a prompt, chosen by the user
When creating a batch of 50 images, they therefore might need a tensor of shape 50x256x256x3, i.e. 50 latent variables of shape 256x256x3.
If we quasi-randomize these latent variables, we get more diversity.

For example:
`# x is a batch of 50 random latent vectors, with size 256x256 and 3 channels
x = np.random.randn(50,256,256,3)`
`from nevergrad.common import sphere`
`derand_x = quasi_randomize(x)`
`Images=[]`
`for i in range(50):`
    `Images+= my_latent_image_generator(derand_x[i])   # Should have more diversity than with x[i]`


## Citation

```
@misc{imagesdiversity_nevergrad,
  author = {Evergrad, N.},
  title = {Image diversity with Nevergrad},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/common/sphere.py}},
  commit = {01e1bc02e366783d37adfbf7af6326457977ef1f}
}

@misc{zameshina2023diversediffusionenhancingimage,
      title={Diverse Diffusion: Enhancing Image Diversity in Text-to-Image Generation}, 
      author={Mariia Zameshina and Olivier Teytaud and Laurent Najman},
      year={2023},
      eprint={2310.12583},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2310.12583}, 
}

```
