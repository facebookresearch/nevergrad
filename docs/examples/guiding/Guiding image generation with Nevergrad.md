
People love image generators. However, the results is frequently not perfect.

Related paper: https://dl.acm.org/doi/abs/10.1145/3583131.3590471 (nominated for best paper award at Gecco)

Comments welcome here: https://www.facebook.com/groups/nevergradusers/

## Latent variables in image generation

Many latent diffusion models use random latent variables.
They create an image, with as inputs:
- a randomly drawn normal latent tensor, e.g. 256x256x3
- a prompt, chosen by the user
When creating a batch of 50 images, they therefore might need a tensor of shape 50x256x256x3, i.e. 50 latent variables of shape 256x256x3.

Typically, the first batch is randomly drawn.
## Applying the Voronoi crossover for latent image generation

When the user watches the 50 images, she might select her favorite ones, for example the images with indices 4, 16 and 48. This is a great information: we can then combine these 3 latent variables X4, X16 and X48.

The Voronoi crossover turns out to be a great idea.
- Randomly choose v1,v2,v3,v4,v5,v6 in D=[0,255]^2 (here the number of cells is twice the number of chosen images).
- Then, split [0,255] in 5 Voronoi cells: the cell V1 corresponding to v1 is the part of D that is closer to v1 than to v2,v3,v4,v5 or v6, and the cells V2,V3,V4,V5,V6 corresponding to v2,v3,v4,v5,v6 are similarly defined. These cells V1-V6 (except for equality cases) are a partition of D.
- Then create a new latent variable by using X4 for filling the cell V1, X16 for V2, X48 for V3, X4 for V4, X16 for V5, X48 for V6. 
The Voronoi crossover (which is randomized) can be applied for creating 50 new latent variables

## Enhancements
The paper above mentions other possible enhancements:
- if the user selects one single image, and specifies where in the image she is unhappy with the result, we might keep the same latent variable, except a small area close to the user click.
- machine learning can be applied for predicting the best latent variables; this is a surrogate model approach.

Post in  https://www.facebook.com/groups/nevergradusers/ if you need code or help.


## Citation

```
@misc{guidedimagegeneration_nevergrad,
  author = {Evergrad, N.},
  title = {Guiding latent image generation with Nevergrad},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/common/sphere.py}},
  commit = {01e1bc02e366783d37adfbf7af6326457977ef1f}
}

@inproceedings{10.1145/3583131.3590471,
author = {Videau, Mathurin and Knizev, Nickolai and Leite, Alessandro and Schoenauer, Marc and Teytaud, Olivier},
title = {Interactive Latent Diffusion Model},
year = {2023},
isbn = {9798400701191},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583131.3590471},
doi = {10.1145/3583131.3590471},
abstract = {This paper introduces Interactive Latent Diffusion Model (IELDM), an encapsulation of a popular text-to-image diffusion model into an Evolutionary framework, allowing the users to steer the design of images toward their goals, alleviating the tedious trial-and-error process that such tools frequently require. The users can not only designate their favourite images, allowing the system to build a surrogate model based on their goals and move in the same directions, but also click on some specific parts of the images to either locally refine the image through dedicated mutation, or recombine images by choosing on each one some regions they like. Experiments validate the benefits of IELDM, especially in a situation where Latent Diffusion Model is challenged by complex input prompts.},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
pages = {586–596},
numpages = {11},
location = {Lisbon, Portugal},
series = {GECCO '23}
}

```
