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

MetaRecentering = SamplingSearch(
    cauchy=False, autorescale=True, sampler="Hammersley", scrambled=True).with_name("MetaRecentering", register=True)
MetaCauchyRecentering = SamplingSearch(
    cauchy=True, autorescale=True, sampler="Hammersley", scrambled=True).with_name("MetaCauchyRecentering", register=True)
Recentering1ScrHammersleySearch = SamplingSearch(
    scale=0.1, sampler="Hammersley", scrambled=True).with_name("Recentering1ScrHammersleySearch", register=True)
Recentering4ScrHammersleySearch = SamplingSearch(
    scale=0.4, sampler="Hammersley", scrambled=True).with_name("Recentering4ScrHammersleySearch", register=True)
CauchyRecentering4ScrHammersleySearch = SamplingSearch(
    scale=0.4, cauchy=True, sampler="Hammersley", scrambled=True).with_name("CauchyRecentering4ScrHammersleySearch", register=True)
Recentering1ScrHaltonSearch = SamplingSearch(
    scale=0.1, sampler="Halton", scrambled=True).with_name("Recentering1ScrHaltonSearch", register=True)
Recentering4ScrHaltonSearch = SamplingSearch(
    scale=0.4, sampler="Halton", scrambled=True).with_name("Recentering4ScrHaltonSearch", register=True)
Recentering7ScrHammersleySearch = SamplingSearch(
    scale=0.7, sampler="Hammersley", scrambled=True).with_name("Recentering7ScrHammersleySearch", register=True)
CauchyRecentering7ScrHammersleySearch = SamplingSearch(
    scale=0.7, cauchy=True, sampler="Hammersley", scrambled=True).with_name("CauchyRecentering7ScrHammersleySearch", register=True)
Recentering20ScrHaltonSearch = SamplingSearch(
    scale=2.0, sampler="Halton", scrambled=True).with_name("Recentering20ScrHaltonSearch", register=True)
Recentering20ScrHammersleySearch = SamplingSearch(
    scale=2.0, sampler="Hammersley", scrambled=True).with_name("Recentering20ScrHammersleySearch", register=True)
Recentering12ScrHaltonSearch = SamplingSearch(
    scale=1.2, sampler="Halton", scrambled=True).with_name("Recentering12ScrHaltonSearch", register=True)
Recentering12ScrHammersleySearch = SamplingSearch(
    scale=1.2, sampler="Hammersley", scrambled=True).with_name("Recentering12ScrHammersleySearch", register=True)
CauchyRecentering12ScrHammersleySearch = SamplingSearch(
    cauchy=True, scale=1.2, sampler="Hammersley", scrambled=True).with_name("CauchyRecentering12ScrHammersleySearch", register=True)
Recentering7ScrHaltonSearch = SamplingSearch(
    scale=0.7, sampler="Halton", scrambled=True).with_name("Recentering7ScrHaltonSearch", register=True)
Recentering0ScrHammersleySearch = SamplingSearch(
    scale=0.01, sampler="Hammersley", scrambled=True).with_name("Recentering0ScrHammersleySearch", register=True)
Recentering0ScrHaltonSearch = SamplingSearch(
    scale=0.01, sampler="Halton", scrambled=True).with_name("Recentering0ScrHaltonSearch", register=True)
ORecentering1ScrHammersleySearch = SamplingSearch(opposition_mode="opposite",
                                                  scale=0.1, sampler="Hammersley", scrambled=True).with_name("ORecentering1ScrHammersleySearch", register=True)
ORecentering4ScrHammersleySearch = SamplingSearch(opposition_mode="opposite",
                                                  scale=0.4, sampler="Hammersley", scrambled=True).with_name("ORecentering4ScrHammersleySearch", register=True)
QORecentering4ScrHammersleySearch = SamplingSearch(opposition_mode="quasi",
                                                   scale=0.4, sampler="Hammersley", scrambled=True).with_name("QORecentering4ScrHammersleySearch", register=True)
ORecentering1ScrHaltonSearch = SamplingSearch(opposition_mode="opposite",
                                              scale=0.1, sampler="Halton", scrambled=True).with_name("ORecentering1ScrHaltonSearch", register=True)
ORecentering4ScrHaltonSearch = SamplingSearch(opposition_mode="opposite",
                                              scale=0.4, sampler="Halton", scrambled=True).with_name("ORecentering4ScrHaltonSearch", register=True)
ORecentering7ScrHammersleySearch = SamplingSearch(opposition_mode="opposite",
                                                  scale=0.7, sampler="Hammersley", scrambled=True).with_name("ORecentering7ScrHammersleySearch", register=True)
QORecentering7ScrHammersleySearch = SamplingSearch(opposition_mode="quasi",
                                                   scale=0.7, sampler="Hammersley", scrambled=True).with_name("QORecentering7ScrHammersleySearch", register=True)
ORecentering20ScrHaltonSearch = SamplingSearch(opposition_mode="opposite",
                                               scale=2.0, sampler="Halton", scrambled=True).with_name("ORecentering20ScrHaltonSearch", register=True)
ORecentering20ScrHammersleySearch = SamplingSearch(opposition_mode="opposite",
                                                   scale=2.0, sampler="Hammersley", scrambled=True).with_name("ORecentering20ScrHammersleySearch", register=True)
ORecentering12ScrHaltonSearch = SamplingSearch(opposition_mode="opposite",
                                               scale=1.2, sampler="Halton", scrambled=True).with_name("ORecentering12ScrHaltonSearch", register=True)
ORecentering12ScrHammersleySearch = SamplingSearch(opposition_mode="opposite",
                                                   scale=1.2, sampler="Hammersley", scrambled=True).with_name("ORecentering12ScrHammersleySearch", register=True)
ORecentering7ScrHaltonSearch = SamplingSearch(opposition_mode="opposite",
                                              scale=0.7, sampler="Halton", scrambled=True).with_name("ORecentering7ScrHaltonSearch", register=True)
ORecentering0ScrHammersleySearch = SamplingSearch(opposition_mode="opposite",
                                                  scale=0.01, sampler="Hammersley", scrambled=True).with_name("ORecentering0ScrHammersleySearch", register=True)
ORecentering0ScrHaltonSearch = SamplingSearch(opposition_mode="opposite",
                                              scale=0.01, sampler="Halton", scrambled=True).with_name("ORecentering0ScrHaltonSearch", register=True)
