# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union
import numpy as np
from scipy import stats
from ..common.typetools import ArrayLike
from ..instrumentation import Instrumentation
from . import sequences
from . import base
from .oneshot import *

MetaCalais = SamplingSearch(
    cauchy=False, supercalais=True, sampler="Hammersley", scrambled=True).with_name("MetaCalais", register=True)
MetaCauchyCalais = SamplingSearch(
    cauchy=True, supercalais=True, sampler="Hammersley", scrambled=True).with_name("MetaCauchyCalais", register=True)
Calais1ScrHammersleySearch = SamplingSearch(
    scale=0.1, sampler="Hammersley", scrambled=True).with_name("Calais1ScrHammersleySearch", register=True)
Calais4ScrHammersleySearch = SamplingSearch(
    scale=0.4, sampler="Hammersley", scrambled=True).with_name("Calais4ScrHammersleySearch", register=True)
CauchyCalais4ScrHammersleySearch = SamplingSearch(
    scale=0.4, cauchy=True, sampler="Hammersley", scrambled=True).with_name("CauchyCalais4ScrHammersleySearch", register=True)
Calais1ScrHaltonSearch = SamplingSearch(
    scale=0.1, sampler="Halton", scrambled=True).with_name("Calais1ScrHaltonSearch", register=True)
Calais4ScrHaltonSearch = SamplingSearch(
    scale=0.4, sampler="Halton", scrambled=True).with_name("Calais4ScrHaltonSearch", register=True)
Calais7ScrHammersleySearch = SamplingSearch(
    scale=0.7, sampler="Hammersley", scrambled=True).with_name("Calais7ScrHammersleySearch", register=True)
CauchyCalais7ScrHammersleySearch = SamplingSearch(
    scale=0.7, cauchy=True, sampler="Hammersley", scrambled=True).with_name("CauchyCalais7ScrHammersleySearch", register=True)
Calais20ScrHaltonSearch = SamplingSearch(
    scale=2.0, sampler="Halton", scrambled=True).with_name("Calais20ScrHaltonSearch", register=True)
Calais20ScrHammersleySearch = SamplingSearch(
    scale=2.0, sampler="Hammersley", scrambled=True).with_name("Calais20ScrHammersleySearch", register=True)
Calais12ScrHaltonSearch = SamplingSearch(
    scale=1.2, sampler="Halton", scrambled=True).with_name("Calais12ScrHaltonSearch", register=True)
Calais12ScrHammersleySearch = SamplingSearch(
    scale=1.2, sampler="Hammersley", scrambled=True).with_name("Calais12ScrHammersleySearch", register=True)
CauchyCalais12ScrHammersleySearch = SamplingSearch(
    cauchy=True, scale=1.2, sampler="Hammersley", scrambled=True).with_name("CauchyCalais12ScrHammersleySearch", register=True)
Calais7ScrHaltonSearch = SamplingSearch(
    scale=0.7, sampler="Halton", scrambled=True).with_name("Calais7ScrHaltonSearch", register=True)
Calais0ScrHammersleySearch = SamplingSearch(
    scale=0.01, sampler="Hammersley", scrambled=True).with_name("Calais0ScrHammersleySearch", register=True)
Calais0ScrHaltonSearch = SamplingSearch(
    scale=0.01, sampler="Halton", scrambled=True).with_name("Calais0ScrHaltonSearch", register=True)
OCalais1ScrHammersleySearch = SamplingSearch(quasi_opposite="opposite",
    scale=0.1, sampler="Hammersley", scrambled=True).with_name("OCalais1ScrHammersleySearch", register=True)
OCalais4ScrHammersleySearch = SamplingSearch(quasi_opposite="opposite",
    scale=0.4, sampler="Hammersley", scrambled=True).with_name("OCalais4ScrHammersleySearch", register=True)
QOCalais4ScrHammersleySearch = SamplingSearch(quasi_opposite="quasi",
    scale=0.4, sampler="Hammersley", scrambled=True).with_name("QOCalais4ScrHammersleySearch", register=True)
OCalais1ScrHaltonSearch = SamplingSearch(quasi_opposite="opposite",
    scale=0.1, sampler="Halton", scrambled=True).with_name("OCalais1ScrHaltonSearch", register=True)
OCalais4ScrHaltonSearch = SamplingSearch(quasi_opposite="opposite",
    scale=0.4, sampler="Halton", scrambled=True).with_name("OCalais4ScrHaltonSearch", register=True)
OCalais7ScrHammersleySearch = SamplingSearch(quasi_opposite="opposite",
    scale=0.7, sampler="Hammersley", scrambled=True).with_name("OCalais7ScrHammersleySearch", register=True)
QOCalais7ScrHammersleySearch = SamplingSearch(quasi_opposite="quasi",
    scale=0.7, sampler="Hammersley", scrambled=True).with_name("QOCalais7ScrHammersleySearch", register=True)
OCalais20ScrHaltonSearch = SamplingSearch(quasi_opposite="opposite",
    scale=2.0, sampler="Halton", scrambled=True).with_name("OCalais20ScrHaltonSearch", register=True)
OCalais20ScrHammersleySearch = SamplingSearch(quasi_opposite="opposite",
    scale=2.0, sampler="Hammersley", scrambled=True).with_name("OCalais20ScrHammersleySearch", register=True)
OCalais12ScrHaltonSearch = SamplingSearch(quasi_opposite="opposite",
    scale=1.2, sampler="Halton", scrambled=True).with_name("OCalais12ScrHaltonSearch", register=True)
OCalais12ScrHammersleySearch = SamplingSearch(quasi_opposite="opposite",
    scale=1.2, sampler="Hammersley", scrambled=True).with_name("OCalais12ScrHammersleySearch", register=True)
OCalais7ScrHaltonSearch = SamplingSearch(quasi_opposite="opposite",
    scale=0.7, sampler="Halton", scrambled=True).with_name("OCalais7ScrHaltonSearch", register=True)
OCalais0ScrHammersleySearch = SamplingSearch(quasi_opposite="opposite",
    scale=0.01, sampler="Hammersley", scrambled=True).with_name("OCalais0ScrHammersleySearch", register=True)
OCalais0ScrHaltonSearch = SamplingSearch(quasi_opposite="opposite",
    scale=0.01, sampler="Halton", scrambled=True).with_name("OCalais0ScrHaltonSearch", register=True)


