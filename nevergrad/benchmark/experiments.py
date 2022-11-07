# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
import typing as tp
import itertools
import numpy as np
import nevergrad as ng
import nevergrad.functions.corefuncs as corefuncs
from nevergrad.functions import base as fbase
from nevergrad.functions import ExperimentFunction
from nevergrad.functions import ArtificialFunction
from nevergrad.functions import FarOptimumFunction
from nevergrad.functions.fishing import OptimizeFish
from nevergrad.functions.pbt import PBT
from nevergrad.functions.ml import MLTuning
from nevergrad.functions import mlda as _mlda
from nevergrad.functions.photonics import Photonics
from nevergrad.functions.arcoating import ARCoating
from nevergrad.functions import images as imagesxp
from nevergrad.functions.powersystems import PowerSystem
from nevergrad.functions.ac import NgAquacrop
from nevergrad.functions.stsp import STSP
from nevergrad.functions.rocket import Rocket
from nevergrad.functions.irrigation import Irrigation
from nevergrad.functions.pcse import CropSimulator
from nevergrad.functions.mixsimulator import OptimizeMix
from nevergrad.functions.unitcommitment import UnitCommitmentProblem
from nevergrad.functions import control
from nevergrad.functions import rl
from nevergrad.functions.games import game
from nevergrad.functions import iohprofiler
from nevergrad.functions import helpers
from nevergrad.functions.cycling import Cycling
from .xpbase import Experiment as Experiment
from .xpbase import create_seed_generator
from .xpbase import registry as registry  # noqa
from .optgroups import get_optimizers

# register all experiments from other files
# pylint: disable=unused-import
from . import frozenexperiments  # noqa
from . import gymexperiments  # noqa

# pylint: disable=stop-iteration-return, too-many-nested-blocks, too-many-locals


centroids = {}
centroids["Fiji"] = (163.85316464458234, -17.31630942638265)
centroids["Tanzania"] = (34.75298985475595, -6.257732428506092)
centroids["Western Sahara"] = (-12.13783111160779, 24.291172960208623)
centroids["Canada"] = (-98.14238137209708, 61.46907614534896)
centroids["United States"] = (-112.5994359115045, 45.70562800215178)
centroids["Kazakhstan"] = (67.2846109811001, 48.19166075218232)
centroids["Uzbekistan"] = (63.20363952823182, 41.748602664652246)
centroids["Papua New Guinea"] = (145.31757462782247, -6.451644514630347)
centroids["Indonesia"] = (117.42340756227364, -2.221737936520542)
centroids["Argentina"] = (-65.17536077114173, -35.4468214894951)
centroids["Chile"] = (-71.5206439451643, -39.04701430994844)
centroids["Democratic Republic of the Congo"] = (23.582955831479083, -2.8502757110956667)
centroids["Somalia"] = (45.72670076723565, 4.752347756504953)
centroids["Kenya"] = (37.791555286661385, 0.5959662521769523)
centroids["Sudan"] = (29.862604012257922, 15.990585003116717)
centroids["Chad"] = (18.581329525332894, 15.328867399839682)
centroids["Haiti"] = (-72.65801330535574, 18.900700691843337)
centroids["Dominican Republic"] = (-70.46235845697531, 18.884487087982258)
centroids["Russian Federation"] = (96.80331818290134, 61.961663494923)
centroids["Bahamas"] = (-77.92997080393516, 25.515491725336624)
centroids["Falkland Islands / Malvinas"] = (-59.42097279311021, -51.71322176551185)
centroids["Norway"] = (15.468119955206761, 69.15685630975351)
centroids["Greenland"] = (-41.50018111492097, 74.77048769398986)
centroids["French Southern and Antarctic Lands"] = (69.5315804704237, -49.306454911671985)
centroids["Timor-Leste"] = (125.96630027368401, -8.767760362467003)
centroids["South Africa"] = (25.048013879861678, -28.947033259979115)
centroids["Lesotho"] = (28.170105295170494, -29.625290493692013)
centroids["Mexico"] = (-102.5763495239869, 23.935371902244835)
centroids["Uruguay"] = (-56.003278666548475, -32.780904365230825)
centroids["Brazil"] = (-53.05434003576711, -10.806773643498916)
centroids["Bolivia"] = (-64.64140560603113, -16.72898701530584)
centroids["Peru"] = (-74.39180581684722, -9.191562905134553)
centroids["Colombia"] = (-73.07773208697478, 3.927213862709704)
centroids["Panama"] = (-80.10916483549376, 8.530019388864652)
centroids["Costa Rica"] = (-84.17542309600948, 9.965671127464528)
centroids["Nicaragua"] = (-85.02031850080252, 12.848190428036988)
centroids["Honduras"] = (-86.58996383801542, 14.822947081652929)
centroids["El Salvador"] = (-88.87290317032377, 13.726091625794197)
centroids["Guatemala"] = (-90.36945836053154, 15.699360612026911)
centroids["Belize"] = (-88.70342125299318, 17.197089911451545)
centroids["Venezuela"] = (-66.16382727830238, 7.162132267639002)
centroids["Guyana"] = (-58.97120310856251, 4.790225375174759)
centroids["Suriname"] = (-55.91145629952073, 4.1200080317588865)
centroids["France"] = (-2.8766966992706267, 42.46070432663372)
centroids["Ecuador"] = (-78.38416674608374, -1.4547717055405804)
centroids["Puerto Rico"] = (-66.47922227695507, 18.2372245709719)
centroids["Jamaica"] = (-77.32425480164892, 18.137636127868436)
centroids["Cuba"] = (-78.96068490970256, 21.631751541025228)
centroids["Zimbabwe"] = (29.788548371892524, -18.906987947858802)
centroids["Botswana"] = (23.773081465789428, -22.099711378826413)
centroids["Namibia"] = (17.156168126194093, -22.099776931731068)
centroids["Senegal"] = (-14.50980278585943, 14.354139988452022)
centroids["Mali"] = (-3.543294339453343, 17.267772061700715)
centroids["Mauritania"] = (-10.326396925234992, 20.20926720635376)
centroids["Benin"] = (2.337377553496156, 9.647430780663699)
centroids["Niger"] = (9.324427099857923, 17.345552814745542)
centroids["Nigeria"] = (7.995127754089786, 9.548318418209965)
centroids["Cameroon"] = (12.611551546501774, 5.663095287992696)
centroids["Togo"] = (0.9964039436703582, 8.439541954669616)
centroids["Ghana"] = (-1.2369685557063992, 7.928651813099648)
centroids["Côte d'Ivoire"] = (-5.6120436452252225, 7.5537550070104915)
centroids["Guinea"] = (-11.060853741185456, 10.44827287727134)
centroids["Guinea-Bissau"] = (-15.110623751667879, 12.022704382325685)
centroids["Liberia"] = (-9.410836154371117, 6.431619862252993)
centroids["Sierra Leone"] = (-11.795257428559948, 8.53035372615305)
centroids["Burkina Faso"] = (-1.77653745205594, 12.311650494136712)
centroids["Central African Republic"] = (20.374347291243915, 6.5427787059213145)
centroids["Republic of the Congo"] = (15.13446176741353, -0.8378010872252886)
centroids["Gabon"] = (11.687751174902044, -0.6470481398040288)
centroids["Equatorial Guinea"] = (10.366031325064027, 1.6458643199600753)
centroids["Zambia"] = (27.727591918760577, -13.395067553524187)
centroids["Malawi"] = (34.193605326922366, -13.172834992341919)
centroids["Mozambique"] = (35.47261597864435, -17.230448975659677)
centroids["Kingdom of eSwatini"] = (31.39525590206532, -26.48985528852001)
centroids["Angola"] = (17.47057255231345, -12.245869036133188)
centroids["Burundi"] = (29.91389229542573, -3.3773910753554657)
centroids["Israel"] = (35.003851206429005, 31.4849193900197)
centroids["Lebanon"] = (35.87098632001643, 33.91182720781994)
centroids["Madagascar"] = (46.69117091471639, -19.356114077828778)
centroids["Palestine"] = (35.27331962289024, 31.94113662241515)
centroids["The Gambia"] = (-15.431872807730837, 13.47533435870166)
centroids["Tunisia"] = (9.534716120695835, 34.172939036882376)
centroids["Algeria"] = (2.5980477916183444, 28.185481278657537)
centroids["Jordan"] = (36.77945490632519, 31.245490584748417)
centroids["United Arab Emirates"] = (54.20671476159633, 23.86863365334761)
centroids["Qatar"] = (51.1835025789133, 25.32185097420669)
centroids["Kuwait"] = (47.600098887626416, 29.307266634033564)
centroids["Iraq"] = (43.75691096461423, 33.03682096372491)
centroids["Oman"] = (56.09867281997542, 20.611174374229545)
centroids["Vanuatu"] = (167.07375126822674, -15.542677057554924)
centroids["Cambodia"] = (104.87608532525192, 12.684728629393506)
centroids["Thailand"] = (101.00613354626108, 15.01697499141648)
centroids["Lao PDR"] = (103.75025989504465, 18.444978089036088)
centroids["Myanmar"] = (96.50584094206161, 21.016999873773827)
centroids["Vietnam"] = (106.28584079705195, 16.657937753254938)
centroids["Dem. Rep. Korea"] = (127.16501590888976, 40.14302033650109)
centroids["Republic of Korea"] = (127.8213171283307, 36.42759860415487)
centroids["Mongolia"] = (102.94640620846634, 46.82368112626357)
centroids["India"] = (79.59370376325381, 22.92500640740852)
centroids["Bangladesh"] = (90.26792827719598, 23.83946179534406)
centroids["Bhutan"] = (90.4724248062037, 27.427968649102027)
centroids["Nepal"] = (84.01317367692529, 28.23944001904935)
centroids["Pakistan"] = (69.41399806318127, 29.973460025547393)
centroids["Afghanistan"] = (66.08669022192831, 33.85639928169076)
centroids["Tajikistan"] = (71.03443504896113, 38.58308146421082)
centroids["Kyrgyzstan"] = (74.62040481092558, 41.50689371318262)
centroids["Turkmenistan"] = (59.27543026236141, 39.09124018017583)
centroids["Iran"] = (54.285451496891426, 32.51891731762539)
centroids["Syria"] = (38.54423941961137, 35.012614281129)
centroids["Armenia"] = (45.00029001101479, 40.21660761230143)
centroids["Sweden"] = (16.59626584684802, 62.811484968080336)
centroids["Belarus"] = (27.98135261544803, 53.50634479481114)
centroids["Ukraine"] = (31.229122070266495, 49.14882260840351)
centroids["Poland"] = (19.31101430844868, 52.14826021933187)
centroids["Austria"] = (14.076158884337072, 47.6139487927463)
centroids["Hungary"] = (19.357628627745918, 47.19995117195427)
centroids["Moldova"] = (28.41048279080328, 47.20367642606752)
centroids["Romania"] = (24.943252494635377, 45.857101035738005)
centroids["Lithuania"] = (23.88064027584349, 55.284319484766066)
centroids["Latvia"] = (24.833296149803438, 56.80717513427924)
centroids["Estonia"] = (25.824725613026608, 58.643695426630906)
centroids["Germany"] = (10.288485092742851, 51.13372269040778)
centroids["Bulgaria"] = (25.19511095327711, 42.7531187620217)
centroids["Greece"] = (22.719813447095053, 39.066715899713955)
centroids["Turkey"] = (35.116900150938406, 39.06837194061471)
centroids["Albania"] = (20.03242643144321, 41.141353306048785)
centroids["Croatia"] = (16.566189771645533, 45.01623399593114)
centroids["Switzerland"] = (8.118300613385486, 46.79173768366762)
centroids["Luxembourg"] = (5.965223432343999, 49.76570507415103)
centroids["Belgium"] = (4.580834113854935, 50.65244095902296)
centroids["Netherlands"] = (5.512217100965399, 52.298700374441786)
centroids["Portugal"] = (-8.055765588295687, 39.63404977497817)
centroids["Spain"] = (-3.6170206023873743, 40.348656106226734)
centroids["Ireland"] = (-8.010236544877012, 53.18059120995006)
centroids["New Caledonia"] = (165.53447460087543, -21.26135761252651)
centroids["Solomon Islands"] = (159.9666154474296, -8.852497470848528)
centroids["New Zealand"] = (172.70192594405574, -41.662578757158684)
centroids["Australia"] = (134.50277547536595, -25.730654779726077)
centroids["Sri Lanka"] = (80.66723574931648, 7.700534417248105)
centroids["China"] = (103.88361230063249, 36.555066531858685)
centroids["Taiwan"] = (120.97480073748623, 23.740964979784938)
centroids["Italy"] = (12.140788372235871, 42.751183052964265)
centroids["Denmark"] = (9.876372937675002, 56.06393446179454)
centroids["United Kingdom"] = (-2.8531353951805545, 53.91477348053706)
centroids["Iceland"] = (-18.761028770831768, 65.07427633529105)
centroids["Azerbaijan"] = (47.553909558006055, 40.22069054766167)
centroids["Georgia"] = (43.48154265409026, 42.16201501415301)
centroids["Philippines"] = (122.90267236988682, 11.763799362297663)
centroids["Malaysia"] = (109.6981484486297, 3.7255884257737155)
centroids["Brunei Darussalam"] = (114.91510877393951, 4.690250542520635)
centroids["Slovenia"] = (14.938152320795732, 46.12542205901039)
centroids["Finland"] = (26.211764610296353, 64.50409403963651)
centroids["Slovakia"] = (19.507657147433708, 48.7267113517275)
centroids["Czech Republic"] = (15.334558102365815, 49.775245294369)
centroids["Eritrea"] = (38.67818692786484, 15.427276751412931)
centroids["Japan"] = (138.06496213270776, 37.66311081170466)
centroids["Paraguay"] = (-58.387387833505706, -23.248041946292087)
centroids["Yemen"] = (47.535044758543485, 15.913231950143004)
centroids["Saudi Arabia"] = (44.51636376826477, 24.123289839105293)
centroids["Antarctica"] = (20.571000569842635, -80.49198288284343)
centroids["Northern Cyprus"] = (33.5582859592249, 35.273957681259716)
centroids["Cyprus"] = (33.03955380295407, 34.90706085094344)
centroids["Morocco"] = (-8.420479544549693, 29.885394698302058)
centroids["Egypt"] = (29.844461513124415, 26.50661999974957)
centroids["Libya"] = (17.974352779160352, 26.997460407020338)
centroids["Ethiopia"] = (39.551255792937745, 8.653999188132575)
centroids["Djibouti"] = (42.4980197360445, 11.773044395533926)
centroids["Somaliland"] = (46.23074953490769, 9.757971805222988)
centroids["Uganda"] = (32.35755031998686, 1.2954855035097297)
centroids["Rwanda"] = (29.91896392224289, -2.0135144658341346)
centroids["Bosnia and Herzegovina"] = (17.816883270390086, 44.1807677629747)
centroids["North Macedonia"] = (21.697903375280845, 41.60592964714007)
centroids["Serbia"] = (20.819651926382583, 44.23303653365162)
centroids["Montenegro"] = (19.2861817215929, 42.78903960655908)
centroids["Kosovo"] = (20.895355721342227, 42.579367131816994)
centroids["Trinidad and Tobago"] = (-61.33036691444967, 10.428237089201879)
centroids["South Sudan"] = (30.198617582461907, 7.292890133516845)
def skip_ci(*, reason: str) -> None:
    """Only use this if there is a good reason for not testing the xp,
    such as very slow for instance (>1min) with no way to make it faster.
    This is dangereous because it won't test reproducibility and the experiment
    may therefore be corrupted with no way to notice it automatically.
    """
    if os.environ.get("NEVERGRAD_PYTEST", False):  # break here for tests
        raise fbase.UnsupportedExperiment("Skipping CI: " + reason)


class _Constraint:
    def __init__(self, name: str, as_bool: bool) -> None:
        self.name = name
        self.as_bool = as_bool

    def __call__(self, data: np.ndarray) -> tp.Union[bool, float]:
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Unexpected inputs as np.ndarray, got {data}")
        if self.name == "sum":
            value = float(np.sum(data))
        elif self.name == "diff":
            value = float(np.sum(data[::2]) - np.sum(data[1::2]))
        elif self.name == "second_diff":
            value = float(2 * np.sum(data[1::2]) - 3 * np.sum(data[::2]))
        elif self.name == "ball":
            # Most points violate the constraint.
            value = float(np.sum(np.square(data)) - len(data) - np.sqrt(len(data)))
        else:
            raise NotImplementedError(f"Unknown function {self.name}")
        return value > 0 if self.as_bool else value


def keras_tuning(
    seed: tp.Optional[int] = None, overfitter: bool = False, seq: bool = False
) -> tp.Iterator[Experiment]:
    """Machine learning hyperparameter tuning experiment. Based on Keras models."""
    seedg = create_seed_generator(seed)
    # Continuous case,

    # First, a few functions with constraints.
    optims: tp.List[str] = ["PSO", "OnePlusOne"] + get_optimizers("basics", seed=next(seedg))  # type: ignore
    datasets = ["kerasBoston", "diabetes", "auto-mpg", "red-wine", "white-wine"]
    for dimension in [None]:
        for dataset in datasets:
            function = MLTuning(
                regressor="keras_dense_nn", data_dimension=dimension, dataset=dataset, overfitter=overfitter
            )
            for budget in [150, 500]:
                for num_workers in (
                    [1, budget // 4] if seq else [budget]
                ):  # Seq for sequential optimization experiments.
                    for optim in optims:
                        xp = Experiment(
                            function, optim, num_workers=num_workers, budget=budget, seed=next(seedg)
                        )
                        skip_ci(reason="too slow")
                        if not xp.is_incoherent:
                            yield xp


def mltuning(
    seed: tp.Optional[int] = None,
    overfitter: bool = False,
    seq: bool = False,
    nano: bool = False,
) -> tp.Iterator[Experiment]:
    """Machine learning hyperparameter tuning experiment. Based on scikit models."""
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))  # type: ignore
    if not seq:
        optims = get_optimizers("oneshot", seed=next(seedg))  # type: ignore
    for dimension in [None, 1, 2, 3]:
        if dimension is None:
            datasets = ["boston", "diabetes", "auto-mpg", "red-wine", "white-wine"]
        else:
            datasets = ["artificialcos", "artificial", "artificialsquare"]
        for regressor in ["mlp", "decision_tree", "decision_tree_depth"]:
            for dataset in datasets:
                function = MLTuning(
                    regressor=regressor, data_dimension=dimension, dataset=dataset, overfitter=overfitter
                )
                for budget in [150, 500] if not nano else [80, 160]:
                    # Seq for sequential optimization experiments.
                    parallelization = [1, budget // 4] if seq else [budget]
                    for num_workers in parallelization:

                        for optim in optims:
                            xp = Experiment(
                                function, optim, num_workers=num_workers, budget=budget, seed=next(seedg)
                            )
                            skip_ci(reason="too slow")
                            if not xp.is_incoherent:
                                yield xp


def naivemltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test."""
    return mltuning(seed, overfitter=True)


# We register only the sequential counterparts for the moment.
@registry.register
def seq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of keras tuning."""
    return keras_tuning(seed, overfitter=False, seq=True)


# We register only the sequential counterparts for the moment.
@registry.register
def naive_seq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Naive counterpart (no overfitting, see naivemltuning)of seq_keras_tuning."""
    return keras_tuning(seed, overfitter=True, seq=True)


@registry.register
def oneshot_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One-shot counterpart of Scikit tuning."""
    return mltuning(seed, overfitter=False, seq=False)


# We register only the (almost) sequential counterparts for the moment.
@registry.register
def seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of mltuning."""
    return mltuning(seed, overfitter=False, seq=True)


@registry.register
def nano_seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of seq_mltuning with smaller budget."""
    return mltuning(seed, overfitter=False, seq=True, nano=True)


@registry.register
def nano_naive_seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test,
    and with lower budget."""
    return mltuning(seed, overfitter=True, seq=True, nano=True)


@registry.register
def naive_seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test."""
    return mltuning(seed, overfitter=True, seq=True)


# pylint:disable=too-many-branches
@registry.register
def yawidebbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Yet Another Wide Black-Box Optimization Benchmark.
    The goal is basically to have a very wide family of problems: continuous and discrete,
    noisy and noise-free, mono- and multi-objective,  constrained and not constrained, sequential
    and parallel.

    TODO(oteytaud): this requires a significant improvement, covering mixed problems and different types of constraints.
    """
    seedg = create_seed_generator(seed)
    total_xp_per_optim = 0
    # Continuous case

    # First, a few functions with constraints.
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation, translation_factor=tf)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
        for tf in [0.1, 10.0]
    ]
    for i, func in enumerate(functions):
        func.parametrization.register_cheap_constraint(_Constraint("sum", as_bool=i % 2 == 0))
    assert len(functions) == 8
    # Then, let us build a constraint-free case. We include the noisy case.
    names = ["hm", "rastrigin", "sphere", "doublelinearslope", "ellipsoid"]

    functions += [
        ArtificialFunction(
            name,
            block_dimension=d,
            rotation=rotation,
            noise_level=nl,
            split=split,
            translation_factor=tf,
            num_blocks=num_blocks,
        )
        for name in names  # period 5
        for rotation in [True, False]  # period 2
        for nl in [0.0, 100.0]  # period 2
        for tf in [0.1, 10.0]
        for num_blocks in [1, 8]  # period 2
        for d in [5, 70, 10000]  # period 4
        for split in [True, False]  # period 2
    ][
        ::37
    ]  # 37 is coprime with all periods above so we sample correctly the possibilities.
    assert len(functions) == 21, f"{len(functions)} problems instead of 21. Yawidebbob should be standard."
    # This problem is intended as a stable basis forever.
    # The list of optimizers should contain only the basic for comparison and "baselines".
    optims: tp.List[str] = ["NGOpt10"] + get_optimizers("baselines", seed=next(seedg))  # type: ignore
    index = 0
    for function in functions:
        for budget in [50, 1500, 25000]:
            for nw in [1, budget] + ([] if budget <= 300 else [300]):
                index += 1
                if index % 5 == 0:
                    total_xp_per_optim += 1
                    for optim in optims:
                        xp = Experiment(function, optim, num_workers=nw, budget=budget, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

    assert total_xp_per_optim == 33, f"We have 33 single-obj xps per optimizer (got {total_xp_per_optim})."
    # Discrete, unordered.
    index = 0
    for nv in [200, 2000]:
        for arity in [2, 7, 37]:
            instrum = ng.p.TransitionChoice(range(arity), repetitions=nv)
            for name in ["onemax", "leadingones", "jump"]:
                index += 1
                if index % 4 != 0:
                    continue
                dfunc = ExperimentFunction(
                    corefuncs.DiscreteFunction(name, arity), instrum.set_name("transition")
                )
                dfunc.add_descriptors(arity=arity)
                for budget in [500, 1500, 5000]:
                    for nw in [1, 100]:
                        total_xp_per_optim += 1
                        for optim in optims:
                            yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))
    assert (
        total_xp_per_optim == 57
    ), f"Including discrete, we check xps per optimizer (got {total_xp_per_optim})."

    # The multiobjective case.
    # TODO the upper bounds are really not well set for this experiment with cigar
    mofuncs: tp.List[fbase.MultiExperiment] = []
    for name1 in ["sphere", "ellipsoid"]:
        for name2 in ["sphere", "hm"]:
            for tf in [0.25, 4.0]:
                mofuncs += [
                    fbase.MultiExperiment(
                        [
                            ArtificialFunction(name1, block_dimension=7),
                            ArtificialFunction(name2, block_dimension=7, translation_factor=tf),
                        ],
                        upper_bounds=np.array((100.0, 100.0)),
                    )
                ]
                mofuncs[-1].add_descriptors(num_objectives=2)
    for name1 in ["sphere", "ellipsoid"]:
        for name2 in ["sphere", "hm"]:
            for name3 in ["sphere", "hm"]:
                for tf in [0.25, 4.0]:
                    mofuncs += [
                        fbase.MultiExperiment(
                            [
                                ArtificialFunction(name1, block_dimension=7, translation_factor=1.0 / tf),
                                ArtificialFunction(name2, block_dimension=7, translation_factor=tf),
                                ArtificialFunction(name3, block_dimension=7),
                            ],
                            upper_bounds=np.array((100.0, 100.0, 100.0)),
                        )
                    ]
                    mofuncs[-1].add_descriptors(num_objectives=3)
    index = 0
    for mofunc in mofuncs[::3]:
        for budget in [2000, 4000, 8000]:
            for nw in [1, 20, 100]:
                index += 1
                if index % 5 == 0:
                    total_xp_per_optim += 1
                    for optim in optims:
                        yield Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))
    assert total_xp_per_optim == 71, f"We should have 71 xps per optimizer, not {total_xp_per_optim}."


# pylint: disable=redefined-outer-name
@registry.register
def parallel_small_budget(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization with small budgets"""
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))  # type: ignore
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "multipeak"]
    names += ["sphere", "cigar", "ellipsoid", "altellipsoid"]
    names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    # funcs
    functions = [
        ArtificialFunction(name, block_dimension=d, rotation=rotation)
        for name in names
        for rotation in [True, False]
        for d in [2, 4, 8]
    ]
    budgets = [10, 50, 100, 200, 400]
    for optim in optims:
        for function in functions:
            for budget in budgets:
                for nw in [2, 8, 16]:
                    for batch in [True, False]:
                        if nw < budget / 4:
                            xp = Experiment(
                                function,
                                optim,
                                num_workers=nw,
                                budget=budget,
                                batch_mode=batch,
                                seed=next(seedg),
                            )
                            if not xp.is_incoherent:
                                yield xp


@registry.register
def instrum_discrete(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Comparison of optimization algorithms equipped with distinct instrumentations.
    Onemax, Leadingones, Jump function."""
    # Discrete, unordered.

    seedg = create_seed_generator(seed)
    optims = get_optimizers("small_discrete", seed=next(seedg))
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ["Unordered", "Softmax", "Ordered"]:
                if instrum_str == "Softmax":
                    instrum: ng.p.Parameter = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    assert instrum_str in ("Ordered", "Unordered")
                    instrum = ng.p.TransitionChoice(
                        range(arity), repetitions=nv, ordered=instrum_str == "Ordered"
                    )
                for name in ["onemax", "leadingones", "jump"]:
                    dfunc = ExperimentFunction(
                        corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str)
                    )
                    dfunc.add_descriptors(arity=arity)
                    for optim in optims:
                        for nw in [1, 10]:
                            for budget in [50, 500, 5000]:
                                yield Experiment(
                                    dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg)
                                )


@registry.register
def sequential_instrum_discrete(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Sequential counterpart of instrum_discrete."""

    seedg = create_seed_generator(seed)
    # Discrete, unordered.
    optims = get_optimizers("discrete", seed=next(seedg))
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ["Unordered", "Softmax", "Ordered"]:
                if instrum_str == "Softmax":
                    instrum: ng.p.Parameter = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    instrum = ng.p.TransitionChoice(
                        range(arity), repetitions=nv, ordered=instrum_str == "Ordered"
                    )
                for name in ["onemax", "leadingones", "jump"]:
                    dfunc = ExperimentFunction(
                        corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str)
                    )
                    dfunc.add_descriptors(arity=arity)
                    for optim in optims:
                        for budget in [50, 500, 5000, 50000]:
                            yield Experiment(dfunc, optim, budget=budget, seed=next(seedg))


@registry.register
def deceptive(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Very difficult objective functions: one is highly multimodal (infinitely many local optima),
    one has an infinite condition number, one has an infinitely long path towards the optimum.
    Looks somehow fractal."""
    seedg = create_seed_generator(seed)
    names = ["deceptivemultimodal", "deceptiveillcond", "deceptivepath"]
    optims = get_optimizers("basics", seed=next(seedg))
    functions = [
        ArtificialFunction(
            name, block_dimension=2, num_blocks=n_blocks, rotation=rotation, aggregator=aggregator
        )
        for name in names
        for rotation in [False, True]
        for n_blocks in [1, 2, 8, 16]
        for aggregator in ["sum", "max"]
    ]
    for func in functions:
        for optim in optims:
            for budget in [25, 37, 50, 75, 87] + list(range(100, 20001, 500)):
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def parallel(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization on 3 classical objective functions: sphere, rastrigin, cigar.
    The number of workers is 20 % of the budget.
    Testing both no useless variables and 5/6 of useless variables."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims: tp.List[str] = get_optimizers("parallel_basics", seed=next(seedg))  # type: ignore
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [25]
        for uv_factor in [0, 5]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=int(budget / 5), seed=next(seedg))


@registry.register
def harderparallel(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization on 4 classical objective functions. More distinct settings than << parallel >>."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar", "ellipsoid"]
    optims = ["NGOpt10"] + get_optimizers("emna_variants", seed=next(seedg))  # type: ignore
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [5, 25]
        for uv_factor in [0, 5]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000]:
                for num_workers in [int(budget / 10), int(budget / 5), int(budget / 3)]:
                    yield Experiment(func, optim, budget=budget, num_workers=num_workers, seed=next(seedg))


@registry.register
def oneshot(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar).
    0 or 5 dummy variables per real variable.
    Base dimension 3 or 25.
    budget 30, 100 or 3000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = get_optimizers("oneshot", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [3, 10, 30, 100, 300, 1000, 3000]
        for uv_factor in [0]  # , 5]
    ]
    for func in functions:
        for optim in optims:
            # if not any(x in str(optim) for x in ["Tune", "Large", "Cauchy"]):
            # if "Meta" in str(optim):
            for budget in [100000, 30, 100, 300, 1000, 3000, 10000]:
                if func.dimension < 3000 or budget < 100000:
                    yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def doe(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar), simplified.
    Base dimension 2000 or 20000. No rotation, no dummy variable.
    Budget 30, 100, 3000, 10000, 30000, 100000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = get_optimizers("oneshot", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [2000, 20000]  # 3, 10, 25, 200, 2000]
        for uv_factor in [0]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000, 30000, 100000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def newdoe(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar), simplified.
    Tested on more dimensionalities than doe, namely 20, 200, 2000, 20000. No dummy variables.
    Budgets 30, 100, 3000, 10000, 30000, 100000, 300000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = get_optimizers("oneshot", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [2000, 20, 200, 20000]  # 3, 10, 25, 200, 2000]
        for uv_factor in [0]
    ]
    budgets = [30, 100, 3000, 10000, 30000, 100000, 300000]
    for func in functions:
        for optim in optims:
            for budget in budgets:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def fiveshots(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Five-shots optimization of 3 classical objective functions (sphere, rastrigin, cigar).
    Base dimension 3 or 25. 0 or 5 dummy variable per real variable. Budget 30, 100 or 3000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = get_optimizers("oneshot", "basics", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [3, 25]
        for uv_factor in [0, 5]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget // 5, seed=next(seedg))


@registry.register
def multimodal(seed: tp.Optional[int] = None, para: bool = False) -> tp.Iterator[Experiment]:
    """Experiment on multimodal functions, namely hm, rastrigin, griewank, rosenbrock, ackley, lunacek,
    deceptivemultimodal.
    0 or 5 dummy variable per real variable.
    Base dimension 3 or 25.
    Budget in 3000, 10000, 30000, 100000.
    Sequential.
    """
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal"]
    # Keep in mind that Rosenbrock is multimodal in high dimension http://ieeexplore.ieee.org/document/6792472/.
    optims = get_optimizers("basics", seed=next(seedg))
    if not para:
        optims += get_optimizers("scipy", seed=next(seedg))
    # + list(sorted(x for x, y in ng.optimizers.registry.items() if "Chain" in x or "BO" in x))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [3, 25]
        for uv_factor in [0, 5]
    ]
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000, 30000, 100000]:
                for nw in [1000] if para else [1]:
                    yield Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def hdmultimodal(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Experiment on multimodal functions, namely hm, rastrigin, griewank, rosenbrock, ackley, lunacek,
    deceptivemultimodal. Similar to multimodal, but dimension 20 or 100 or 1000. Budget 1000 or 10000, sequential."""
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal"]
    # Keep in mind that Rosenbrock is multimodal in high dimension http://ieeexplore.ieee.org/document/6792472/.

    optims = get_optimizers("basics", "multimodal", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=bd)
        for name in names
        for bd in [
            1000,
            6000,
            36000,
        ]  # This has been modified, given that it was not sufficiently high-dimensional for its name.
    ]
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000]:
                for nw in [1]:
                    yield Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def paramultimodal(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel counterpart of the multimodal experiment: 1000 workers."""
    return multimodal(seed, para=True)


@registry.register
def bonnans(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    instrum = ng.p.TransitionChoice(range(2), repetitions=100, ordered=False)
    softmax_instrum: ng.p.Parameter = ng.p.Choice(range(2), repetitions=100)
    optims = [
        "RotatedTwoPointsDE",
        "DiscreteLenglerOnePlusOne",
        "PortfolioDiscreteOnePlusOne",
        "FastGADiscreteOnePlusOne",
        "DiscreteDoerrOnePlusOne",
        "DiscreteBSOOnePlusOne",
        "AdaptiveDiscreteOnePlusOne",
        "GeneticDE",
        "DE",
        "TwoPointsDE",
        "DiscreteOnePlusOne",
        "CMA",
        "SQP",
        "MetaModel",
        "DiagonalCMA",
    ]
    for i in range(21):
        bonnans = corefuncs.BonnansFunction(index=i)
        for optim in optims:
            dfunc = ExperimentFunction(
                bonnans,
                instrum.set_name("bitmap") if "Discrete" in optim else softmax_instrum.set_name("softmax"),
            )
            for budget in [20, 50, 100]:
                yield Experiment(dfunc, optim, num_workers=1, budget=budget, seed=next(seedg))


# pylint: disable=redefined-outer-name,too-many-arguments
@registry.register
def yabbob(
    seed: tp.Optional[int] = None,
    parallel: bool = False,
    big: bool = False,
    small: bool = False,
    noise: bool = False,
    hd: bool = False,
    constraint_case: int = 0,
    split: bool = False,
    tuning: bool = False,
    reduction_factor: int = 1,
    bounded: bool = False,
    box: bool = False,
) -> tp.Iterator[Experiment]:
    """Yet Another Black-Box Optimization Benchmark.
    Related to, but without special effort for exactly sticking to, the BBOB/COCO dataset.
    Dimension 2, 10 and 50.
    Budget 50, 200, 800, 3200, 12800.
    Both rotated or not rotated.
    """
    seedg = create_seed_generator(seed)

    # List of objective functions.
    names = [
        "hm",
        "rastrigin",
        "griewank",
        "rosenbrock",
        "ackley",
        "lunacek",
        "deceptivemultimodal",
        "bucherastrigin",
        "multipeak",
    ]
    names += ["sphere", "doublelinearslope", "stepdoublelinearslope"]
    names += ["cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid", "discus", "bentcigar"]
    names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    # Deceptive path is related to the sharp ridge function; there is a long path to the optimum.
    # Deceptive illcond is related to the difference of powers function; the conditioning varies as we get closer to the optimum.
    # Deceptive multimodal is related to the Weierstrass function and to the Schaffers function.

    # Parametrizing the noise level.
    if noise:
        noise_level = 100000 if hd else 100
    else:
        noise_level = 0

    # Choosing the list of optimizers.
    optims: tp.List[str] = get_optimizers("competitive", seed=next(seedg))  # type: ignore
    if noise:
        optims += ["TBPSA", "SQP", "NoisyDiscreteOnePlusOne"]
    if hd:
        optims += ["OnePlusOne"]
        optims += get_optimizers("splitters", seed=next(seedg))  # type: ignore

    if hd and small:
        optims = ["BO", "CMA", "PSO", "DE"]

    if bounded:
        optims = ["BO", "PCABO", "BayesOptimBO", "CMA", "PSO", "DE"]
    if box:
        optims = ["DiagonalCMA", "Cobyla", "NGOpt16", "NGOpt15", "CMandAS2", "OnePlusOne"]
    # List of objective functions.
    functions = [
        ArtificialFunction(
            name,
            block_dimension=d,
            rotation=rotation,
            noise_level=noise_level,
            split=split,
            num_blocks=num_blocks,
            bounded=bounded or box,
        )
        for name in names
        for rotation in [True, False]
        for num_blocks in ([1] if not split else [7, 12])
        for d in (
            [100, 1000, 3000] if hd else ([2, 5, 10, 15] if tuning else ([40] if bounded else [2, 10, 50]))
        )
    ]

    assert reduction_factor in [1, 7, 13, 17]  # needs to be a cofactor
    functions = functions[::reduction_factor]

    # We possibly add constraints.
    max_num_constraints = 4
    constraints: tp.List[tp.Any] = [
        _Constraint(name, as_bool)
        for as_bool in [False, True]
        for name in ["sum", "diff", "second_diff", "ball"]
    ]
    assert (
        constraint_case < len(constraints) + max_num_constraints
    ), "constraint_case should be in 0, 1, ..., {len(constraints) + max_num_constraints - 1} (0 = no constraint)."
    # We reduce the number of tests when there are constraints, as the number of cases
    # is already multiplied by the number of constraint_case.
    for func in functions[:: 13 if constraint_case > 0 else 1]:
        # We add a window of the list of constraints. This windows finishes at "constraints" (hence, is empty if
        # constraint_case=0).
        for constraint in constraints[max(0, constraint_case - max_num_constraints) : constraint_case]:
            func.parametrization.register_cheap_constraint(constraint)

    budgets = (
        [40000, 80000, 160000, 320000]
        if (big and not noise)
        else ([50, 200, 800, 3200, 12800] if not noise else [3200, 12800])
    )
    if small and not noise:
        budgets = [10, 20, 40]
    if bounded:
        budgets = [10, 20, 40, 100, 300]
    for optim in optims:
        for function in functions:
            for budget in budgets:
                xp = Experiment(
                    function, optim, num_workers=100 if parallel else 1, budget=budget, seed=next(seedg)
                )
                if not xp.is_incoherent:
                    yield xp


@registry.register
def yahdlbbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with HD and low budget."""
    return yabbob(seed, hd=True, small=True)


@registry.register
def reduced_yahdlbbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with HD and low budget."""
    return yabbob(seed, hd=True, small=True, reduction_factor=17)


@registry.register
def yanoisysplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, noise=True, parallel=False, split=True)


@registry.register
def yahdnoisysplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, hd=True, noise=True, parallel=False, split=True)


@registry.register
def yaconstrainedbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, constraint_case=i) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yahdnoisybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return yabbob(seed, hd=True, noise=True)


@registry.register
def yabigbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, parallel=False, big=True)


@registry.register
def yasplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with splitting info in the instrumentation."""
    return yabbob(seed, parallel=False, split=True)


@registry.register
def yahdsplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yasplitbbob with more dimension."""
    return yabbob(seed, hd=True, split=True)


@registry.register
def yatuningbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with less budget and less dimension."""
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13, tuning=True)


@registry.register
def yatinybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with less budget and less xps."""
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13)


@registry.register
def yasmallbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with less budget."""
    return yabbob(seed, parallel=False, big=False, small=True)


@registry.register
def yahdbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return yabbob(seed, hd=True)


@registry.register
def yaparabbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization counterpart of yabbob."""
    return yabbob(seed, parallel=True, big=False)


@registry.register
def yanoisybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Noisy optimization counterpart of yabbob.
    This is supposed to be consistent with normal practices in noisy
    optimization: we distinguish recommendations and exploration.
    This is different from the original BBOB/COCO from that point of view.
    """
    return yabbob(seed, noise=True)


@registry.register
def yaboundedbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with bounded domain and dim only 40, (-5,5)**n by default."""
    return yabbob(seed, bounded=True)


@registry.register
def yaboxbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with bounded domain, (-5,5)**n by default."""
    return yabbob(seed, box=True)


@registry.register
def illcondi(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on ill cond problems.
    Cigar, Ellipsoid.
    Both rotated and unrotated.
    Budget 100, 1000, 10000.
    Dimension 50.
    """
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
    ]
    for optim in optims:
        for function in functions:
            for budget in [100, 1000, 10000]:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def illcondipara(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on ill-conditionned parallel optimization.
    50 workers in parallel.
    """
    seedg = create_seed_generator(seed)
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
    ]
    optims = get_optimizers("competitive", seed=next(seedg))
    for function in functions:
        for budget in [100, 1000, 10000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=50, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def constrained_illconditioned_parallel(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Many optimizers on ill cond problems with constraints."""
    seedg = create_seed_generator(seed)
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
    ]
    for func in functions:
        func.parametrization.register_cheap_constraint(_Constraint("sum", as_bool=False))
    for function in functions:
        for budget in [400, 4000, 40000]:
            optims: tp.List[str] = get_optimizers("large", seed=next(seedg))  # type: ignore
            for optim in optims:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def ranknoisy(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Noisy optimization methods on a few noisy problems.
    Cigar, Altcigar, Ellipsoid, Altellipsoid.
    Dimension 200, 2000, 20000.
    Budget 25000, 50000, 100000.
    No rotation.
    Noise level 10.
    With or without noise dissymmetry.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("progressive", seed=next(seedg)) + [  # type: ignore
        "OptimisticNoisyOnePlusOne",
        "OptimisticDiscreteOnePlusOne",
        "NGOpt10",
    ]

    # optims += ["NGO", "Shiwa", "DiagonalCMA"] + sorted(
    #    x for x, y in ng.optimizers.registry.items() if ("SPSA" in x or "TBPSA" in x or "ois" in x or "epea" in x or "Random" in x)
    # )
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [20000, 200, 2000]:
                for name in ["cigar", "altcigar", "ellipsoid", "altellipsoid"]:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(
                            name=name,
                            rotation=False,
                            block_dimension=d,
                            noise_level=10,
                            noise_dissymmetry=noise_dissymmetry,
                            translation_factor=1.0,
                        )
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def noisy(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Noisy optimization methods on a few noisy problems.
    Sphere, Rosenbrock, Cigar, Hm (= highly multimodal).
    Noise level 10.
    Noise dyssymmetry or not.
    Dimension 2, 20, 200, 2000.
    Budget 25000, 50000, 100000.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("progressive", seed=next(seedg)) + [  # type: ignore
        "OptimisticNoisyOnePlusOne",
        "OptimisticDiscreteOnePlusOne",
    ]
    optims += ["NGOpt10", "Shiwa", "DiagonalCMA"] + sorted(
        x
        for x, y in ng.optimizers.registry.items()
        if ("SPSA" in x or "TBPSA" in x or "ois" in x or "epea" in x or "Random" in x)
    )

    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [2, 20, 200, 2000]:
                for name in ["sphere", "rosenbrock", "cigar", "hm"]:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(
                            name=name,
                            rotation=True,
                            block_dimension=d,
                            noise_level=10,
                            noise_dissymmetry=noise_dissymmetry,
                            translation_factor=1.0,
                        )
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def paraalldes(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All DE methods on various functions. Parallel version.
    Dimension 5, 20, 100, 500, 2500.
    Sphere, Cigar, Hm, Ellipsoid.
    No rotation.
    """
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in sorted(x for x, y in ng.optimizers.registry.items() if "DE" in x and "Tune" in x):
            for rotation in [False]:
                for d in [5, 20, 100, 500, 2500]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(
                                function,
                                optim,
                                budget=budget,
                                seed=next(seedg),
                                num_workers=max(d, budget // 6),
                            )


@registry.register
def parahdbo4d(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All Bayesian optimization methods on various functions. Parallel version
    Dimension 20 and 2000.
    Budget 25, 31, 37, 43, 50, 60.
    Sphere, Cigar, Hm, Ellipsoid.
    No rotation.
    """
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in sorted(x for x, y in ng.optimizers.registry.items() if "BO" in x and "Tune" in x):
            for rotation in [False]:
                for d in [20, 2000]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(
                                function,
                                optim,
                                budget=budget,
                                seed=next(seedg),
                                num_workers=max(d, budget // 6),
                            )


@registry.register
def alldes(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All DE methods on various functions.
    Dimension 5, 20, 100.
    Sphere, Cigar, Hm, Ellipsoid.
    Budget 10, 100, 1000, 10000, 100000.
    """
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in sorted(x for x, y in ng.optimizers.registry.items() if "DE" in x or "Shiwa" in x):
            for rotation in [False]:
                for d in [5, 20, 100]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def hdbo4d(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All Bayesian optimization methods on various functions.
    Budget 25, 31, 37, 43, 50, 60.
    Dimension 20.
    Sphere, Cigar, Hm, Ellipsoid.
    """
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in get_optimizers("all_bo", seed=next(seedg)):
            for rotation in [False]:
                for d in [20]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def spsa_benchmark(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Some optimizers on a noisy optimization problem. This benchmark is based on the noisy benchmark.
    Budget 500, 1000, 2000, 4000, ... doubling... 128000.
    Rotation or not.
    Sphere, Sphere4, Cigar.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("spsa", seed=next(seedg))  # type: ignore
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ["sphere", "sphere4", "cigar"]:
                    function = ArtificialFunction(
                        name=name, rotation=rotation, block_dimension=20, noise_level=10
                    )
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def realworld(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Realworld optimization. This experiment contains:

     - a subset of MLDA (excluding the perceptron: 10 functions rescaled or not.
     - ARCoating https://arxiv.org/abs/1904.02907: 1 function.
     - The 007 game: 1 function, noisy.
     - PowerSystem: a power system simulation problem.
     - STSP: a simple TSP problem.
     -  MLDA, except the Perceptron.

    Budget 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800.
    Sequential or 10-parallel or 100-parallel.
    """
    funcs: tp.List[tp.Union[ExperimentFunction, rl.agents.TorchAgentFunction]] = [
        _mlda.Clustering.from_mlda(name, num, rescale)
        for name, num in [("Ruspini", 5), ("German towns", 10)]
        for rescale in [True, False]
    ]
    funcs += [
        _mlda.SammonMapping.from_mlda("Virus", rescale=False),
        _mlda.SammonMapping.from_mlda("Virus", rescale=True),
        # _mlda.SammonMapping.from_mlda("Employees"),
    ]
    funcs += [_mlda.Landscape(transform) for transform in [None, "square", "gaussian"]]

    # Adding ARCoating.
    funcs += [ARCoating()]
    funcs += [PowerSystem(), PowerSystem(13)]
    funcs += [STSP(), STSP(500)]
    funcs += [game.Game("war")]
    funcs += [game.Game("batawaf")]
    funcs += [game.Game("flip")]
    funcs += [game.Game("guesswho")]
    funcs += [game.Game("bigguesswho")]

    # 007 with 100 repetitions, both mono and multi architectures.
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {"mono": rl.agents.Perceptron, "multi": rl.agents.DenseNet}
    agents = {
        a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False)
        for a, m in modules.items()
    }
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    runner = rl.EnvironmentRunner(env.copy(), num_repetitions=100, max_step=50)
    for archi in ["mono", "multi"]:
        func = rl.agents.TorchAgentFunction(agents[archi], runner, reward_postprocessing=lambda x: 1 - x)
        funcs += [func]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def aquacrop_fao(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """FAO Crop simulator. Maximize yield."""

    funcs = [NgAquacrop(i, 300.0 + 150.0 * np.cos(i)) for i in range(3, 7)]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def fishing(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Lotka-Volterra equations"""
    funcs = [OptimizeFish(i) for i in [17, 35, 52, 70, 88, 105]]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def rocket(seed: tp.Optional[int] = None, seq: bool = False) -> tp.Iterator[Experiment]:
    """Rocket simulator. Maximize max altitude by choosing the thrust schedule, given a total thrust.
    Budget 25, 50, ..., 1600.
    Sequential or 30 workers."""
    funcs = [Rocket(i) for i in range(17)]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for num_workers in [1] if seq else [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        skip_ci(reason="Too slow")
                        if not xp.is_incoherent:
                            yield xp

@registry.register
def irrigation(seed: tp.Optional[int] = None, benin: bool = False, variety_choice: bool = False, rice: bool = False, multi_crop: bool = False, kenya: bool = False, year_min:int=2006, year_max:int=2006) -> tp.Iterator[Experiment]:
    """Irrigation simulator. Maximize leaf area index,
    so that you get a lot of primary production.
    Sequential or 30 workers."""
    if kenya:
        addresses = []
        #lat_center = float(os.environ.get("ng_latitude", "0."))
        #lon_center = float(os.environ.get("ng_longitude", "37."))
        country = os.environ.get("ng_country", "Kenya")
        ng_latitude = centroids[country][1]
        ng_longitude = centroids[country][0]


        for lat in [ng_latitude-4.+8. * e/6. for e in range(7)]: #list(range(-4,5)):
            for lon in [ng_longitude-3.+6. * e/6. for e in range(7)]: #list(range(34,40)):
                addresses += [(lat, lon)]
        funcs = [Irrigation(0, benin=benin, variety_choice=variety_choice, rice=rice, multi_crop=multi_crop, address=ad, year_min=year_min,year_max=year_max) for ad in addresses]
    else:
        funcs = [Irrigation(i, benin=benin, variety_choice=variety_choice, rice=rice, multi_crop=multi_crop, address=None,year_min=year_min, year_max=year_max) for i in range(67)]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims = ["DiagonalCMA", "CMA", "DE", "PSO", "TwoPointsDE", "DiscreteLenglerOnePlusOne"]
    optims += ["NGOptRW", "NGTuned"]
    optims = ["NGOptRW"]
    if rice:
        optims = optims[-2:]
    for budget in [int(os.environ.get("ng_budget", "250"))]: #, 50, 100, 200]:
        for num_workers in [1]: #, 30, 60]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        skip_ci(reason="Too slow")
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def crop_and_variety_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, variety_choice=True, multi_crop=True)


@registry.register
def benin_crop_and_variety_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, benin=True, variety_choice=True, multi_crop=True)


@registry.register
def kenya_crop_and_variety_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, kenya=True, variety_choice=True, multi_crop=True)


@registry.register
def kenya_2011_crop_and_variety_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, kenya=True, variety_choice=True, multi_crop=True, year_min=2011)


@registry.register
def kenya_many_crop_and_variety_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, kenya=True, variety_choice=True, multi_crop=True, year_min=2006, year_max=2011)

@registry.register
def kenya_new_many_crop_and_variety_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, kenya=True, variety_choice=True, multi_crop=True, year_min=2015, year_max=2020)

@registry.register
def kenya_old_many_crop_and_variety_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, kenya=True, variety_choice=True, multi_crop=True, year_min=1996, year_max=2001)

@registry.register
def kenya_mat_many_crop_and_variety_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, kenya=True, variety_choice=True, multi_crop=True, year_min=1980, year_max=1985)


@registry.register
def variety_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, variety_choice=True)


@registry.register
def benin_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, benin=True)


@registry.register
def benin_variety_choice_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, benin=True, variety_choice=True)


@registry.register
def benin_rice_variety_choice_irrigation(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return irrigation(seed, benin=True, variety_choice=True, rice=True)


@registry.register
def crop_simulator(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Crop simulator.

    Low dimensional problem, only 2 vars. This is optimization for model identification: we want
    to find parameters so that the simulation matches observations.
    """
    funcs = [CropSimulator()]
    seedg = create_seed_generator(seed)
    optims = ["DE", "PSO", "CMA", "NGOpt"]
    for budget in [25, 50, 100, 200]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mono_rocket(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Sequential counterpart of the rocket problem."""
    return rocket(seed, seq=True)


@registry.register
def mixsimulator(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MixSimulator of power plants
    Budget 20, 40, ..., 1600.
    Sequential or 30 workers."""
    funcs = [OptimizeMix()]
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))  # type: ignore

    for budget in [20, 40, 80, 160]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def control_problem(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MuJoCo testbed. Learn linear policy for different control problems.
    Budget 500, 1000, 3000, 5000."""
    seedg = create_seed_generator(seed)
    num_rollouts = 1
    funcs = [
        Env(num_rollouts=num_rollouts, random_state=seed)
        for Env in [
            control.Swimmer,
            control.HalfCheetah,
            control.Hopper,
            control.Walker2d,
            control.Ant,
            control.Humanoid,
        ]
    ]

    sigmas = [0.1, 0.1, 0.1, 0.1, 0.01, 0.001]
    funcs2 = []
    for sigma, func in zip(sigmas, funcs):
        f = func.copy()
        param: ng.p.Tuple = f.parametrization.copy()  # type: ignore
        for array in param:
            array.set_mutation(sigma=sigma)  # type: ignore
        param.set_name(f"sigma={sigma}")

        f.parametrization = param
        f.parametrization.freeze()
        funcs2.append(f)
    optims = get_optimizers("basics")

    for budget in [50, 75, 100, 150, 200, 250, 300, 400, 500, 1000, 3000, 5000, 8000, 16000, 32000, 64000]:
        for algo in optims:
            for fu in funcs2:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def neuro_control_problem(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MuJoCo testbed. Learn neural policies."""
    seedg = create_seed_generator(seed)
    num_rollouts = 1
    funcs = [
        Env(num_rollouts=num_rollouts, intermediate_layer_dim=(50,), random_state=seed)
        for Env in [
            control.Swimmer,
            control.HalfCheetah,
            control.Hopper,
            control.Walker2d,
            control.Ant,
            control.Humanoid,
        ]
    ]

    optims = ["CMA", "NGOpt4", "DiagonalCMA", "NGOpt8", "MetaModel", "ChainCMAPowell"]

    for budget in [50, 500, 5000, 10000, 20000, 35000, 50000, 100000, 200000]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def olympus_surfaces(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Olympus surfaces"""
    from nevergrad.functions.olympussurfaces import OlympusSurface

    funcs = []
    for kind in OlympusSurface.SURFACE_KINDS:
        for k in range(2, 5):
            for noise in ["GaussianNoise", "UniformNoise", "GammaNoise"]:
                for noise_scale in [0.5, 1]:
                    funcs.append(OlympusSurface(kind, 10**k, noise, noise_scale))

    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", "noisy", seed=next(seedg))
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:  # , 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def olympus_emulators(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Olympus emulators"""
    from nevergrad.functions.olympussurfaces import OlympusEmulator

    funcs = []
    for dataset_kind in OlympusEmulator.DATASETS:
        for model_kind in ["BayesNeuralNet", "NeuralNet"]:
            funcs.append(OlympusEmulator(dataset_kind, model_kind))

    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", "noisy", seed=next(seedg))
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:  # , 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def simple_tsp(seed: tp.Optional[int] = None, complex_tsp: bool = False) -> tp.Iterator[Experiment]:
    """Simple TSP problems. Please note that the methods we use could be applied or complex variants, whereas
    specialized methods can not always do it; therefore this comparisons from a black-box point of view makes sense
    even if white-box methods are not included though they could do this more efficiently.
    10, 100, 1000, 10000 cities.
    Budgets doubling from 25, 50, 100, 200, ... up  to 25600

    """
    funcs = [STSP(10**k, complex_tsp) for k in range(2, 6)]
    seedg = create_seed_generator(seed)
    optims = [
        "RotatedTwoPointsDE",
        "DiscreteLenglerOnePlusOne",
        "DiscreteDoerrOnePlusOne",
        "DiscreteBSOOnePlusOne",
        "AdaptiveDiscreteOnePlusOne",
        "GeneticDE",
        "DE",
        "TwoPointsDE",
        "DiscreteOnePlusOne",
        "CMA",
        "MetaModel",
        "DiagonalCMA",
    ]
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:  # , 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def complex_tsp(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of simple_tsp with non-planar term."""
    return simple_tsp(seed, complex_tsp=True)


@registry.register
def sequential_fastgames(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimization of policies for games, i.e. direct policy search.
    Budget 12800, 25600, 51200, 102400.
    Games: War, Batawaf, Flip, GuessWho,  BigGuessWho."""
    funcs = [game.Game(name) for name in ["war", "batawaf", "flip", "guesswho", "bigguesswho"]]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("noisy", "splitters", "progressive", seed=next(seedg))
    for budget in [12800, 25600, 51200, 102400]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def powersystems(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Unit commitment problem, i.e. management of dams for hydroelectric planning."""
    funcs: tp.List[ExperimentFunction] = []
    for dams in [3, 5, 9, 13]:
        funcs += [PowerSystem(dams, depth=2, width=3)]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", "noisy", "splitters", "progressive", seed=next(seedg))
    budgets = [3200, 6400, 12800]
    for budget in budgets:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mlda(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MLDA (machine learning and data analysis) testbed."""
    funcs: tp.List[ExperimentFunction] = [
        _mlda.Clustering.from_mlda(name, num, rescale)
        for name, num in [("Ruspini", 5), ("German towns", 10)]
        for rescale in [True, False]
    ]
    funcs += [
        _mlda.SammonMapping.from_mlda("Virus", rescale=False),
        _mlda.SammonMapping.from_mlda("Virus", rescale=True),
        # _mlda.SammonMapping.from_mlda("Employees"),
    ]
    funcs += [_mlda.Perceptron.from_mlda(name) for name in ["quadratic", "sine", "abs", "heaviside"]]
    funcs += [_mlda.Landscape(transform) for transform in [None, "square", "gaussian"]]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mldakmeans(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MLDA (machine learning and data analysis) testbed, restricted to the K-means part."""
    funcs: tp.List[ExperimentFunction] = [
        _mlda.Clustering.from_mlda(name, num, rescale)
        for name, num in [("Ruspini", 5), ("German towns", 10), ("Ruspini", 50), ("German towns", 100)]
        for rescale in [True, False]
    ]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("splitters", "progressive", seed=next(seedg))
    for budget in [1000, 10000]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def image_similarity(
    seed: tp.Optional[int] = None, with_pgan: bool = False, similarity: bool = True
) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structured_moo", seed=next(seedg))
    funcs: tp.List[ExperimentFunction] = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in imagesxp.imagelosses.registry.values()
        if loss.REQUIRES_REFERENCE == similarity
    ]
    for budget in [100 * 5**k for k in range(3)]:
        for func in funcs:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                skip_ci(reason="too slow")
                if not xp.is_incoherent:
                    yield xp


@registry.register
def image_similarity_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity, using PGan as a representation."""
    return image_similarity(seed, with_pgan=True)


@registry.register
def image_single_quality(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity, but based on image quality assessment."""
    return image_similarity(seed, with_pgan=False, similarity=False)


@registry.register
def image_single_quality_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity_pgan, but based on image quality assessment."""
    return image_similarity(seed, with_pgan=True, similarity=False)


@registry.register
def image_multi_similarity(
    seed: tp.Optional[int] = None, cross_valid: bool = False, with_pgan: bool = False
) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structured_moo", seed=next(seedg))
    funcs: tp.List[ExperimentFunction] = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in imagesxp.imagelosses.registry.values()
        if loss.REQUIRES_REFERENCE
    ]
    base_values: tp.List[tp.Any] = [func(func.parametrization.sample().value) for func in funcs]
    if cross_valid:
        skip_ci(reason="Too slow")
        mofuncs: tp.List[tp.Any] = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(
            funcs, pareto_size=25
        )
    else:
        mofuncs = [fbase.MultiExperiment(funcs, upper_bounds=base_values)]
    for budget in [100 * 5**k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for mofunc in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@registry.register
def image_multi_similarity_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity, using PGan as a representation."""
    return image_multi_similarity(seed, with_pgan=True)


@registry.register
def image_multi_similarity_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_multi_similarity with cross-validation."""
    return image_multi_similarity(seed, cross_valid=True)


@registry.register
def image_multi_similarity_pgan_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_multi_similarity with cross-validation."""
    return image_multi_similarity(seed, cross_valid=True, with_pgan=True)


@registry.register
def image_quality_proxy(seed: tp.Optional[int] = None, with_pgan: bool = False) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo", seed=next(seedg))
    iqa, blur, brisque = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in (imagesxp.imagelosses.Koncept512, imagesxp.imagelosses.Blur, imagesxp.imagelosses.Brisque)
    ]
    # TODO: add the proxy info in the parametrization.
    for budget in [100 * 5**k for k in range(3)]:
        for algo in optims:
            for func in [blur, brisque]:
                # We optimize on blur or brisque and check performance on iqa.
                sfunc = helpers.SpecialEvaluationExperiment(func, evaluation=iqa)
                sfunc.add_descriptors(non_proxy_function=False)
                xp = Experiment(sfunc, algo, budget, num_workers=1, seed=next(seedg))
                yield xp


@registry.register
def image_quality_proxy_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return image_quality_proxy(seed, with_pgan=True)


@registry.register
def image_quality(
    seed: tp.Optional[int] = None, cross_val: bool = False, with_pgan: bool = False, num_images: int = 1
) -> tp.Iterator[Experiment]:
    """Optimizing images for quality:
    we optimize K512, Blur and Brisque.

    With num_images > 1, we are doing morphing.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo", seed=next(seedg))
    # We optimize func_blur or func_brisque and check performance on func_iqa.
    funcs: tp.List[ExperimentFunction] = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan, num_images=num_images)
        for loss in (
            imagesxp.imagelosses.Koncept512,
            imagesxp.imagelosses.Blur,
            imagesxp.imagelosses.Brisque,
        )
    ]
    # TODO: add the proxy info in the parametrization.
    mofuncs: tp.Sequence[ExperimentFunction]
    if cross_val:
        mofuncs = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(
            experiments=[funcs[0], funcs[2]],
            # Blur is not good enough as an IQA for being in the list.
            training_only_experiments=[funcs[1]],
            pareto_size=16,
        )
    else:
        upper_bounds = [func(func.parametrization.value) for func in funcs]
        mofuncs = [fbase.MultiExperiment(funcs, upper_bounds=upper_bounds)]  # type: ignore
    for budget in [100 * 5**k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for func in mofuncs:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@registry.register
def morphing_pgan_quality(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return image_quality(seed, with_pgan=True, num_images=2)


@registry.register
def image_quality_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_quality with cross-validation."""
    return image_quality(seed, cross_val=True)


@registry.register
def image_quality_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_quality with cross-validation."""
    return image_quality(seed, with_pgan=True)


@registry.register
def image_quality_cv_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_quality with cross-validation."""
    return image_quality(seed, cross_val=True, with_pgan=True)


@registry.register
def image_similarity_and_quality(
    seed: tp.Optional[int] = None, cross_val: bool = False, with_pgan: bool = False
) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo", seed=next(seedg))

    # 3 losses functions including 2 iqas.
    func_iqa = imagesxp.Image(loss=imagesxp.imagelosses.Koncept512, with_pgan=with_pgan)
    func_blur = imagesxp.Image(loss=imagesxp.imagelosses.Blur, with_pgan=with_pgan)
    base_blur_value: float = func_blur(func_blur.parametrization.value)  # type: ignore
    for func in [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in imagesxp.imagelosses.registry.values()
        if loss.REQUIRES_REFERENCE
    ]:

        # Creating a reference value.
        base_value: float = func(func.parametrization.value)  # type: ignore
        mofuncs: tp.Iterable[fbase.ExperimentFunction]
        if cross_val:
            mofuncs = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(
                training_only_experiments=[func, func_blur], experiments=[func_iqa], pareto_size=16
            )
        else:
            mofuncs = [
                fbase.MultiExperiment(
                    [func, func_blur, func_iqa], upper_bounds=[base_value, base_blur_value, 100.0]
                )
            ]
        for budget in [100 * 5**k for k in range(3)]:
            for algo in optims:
                for mofunc in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=1, seed=next(seedg))
                    yield xp


@registry.register
def image_similarity_and_quality_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity_and_quality with cross-validation."""
    return image_similarity_and_quality(seed, cross_val=True)


@registry.register
def image_similarity_and_quality_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity_and_quality with cross-validation."""
    return image_similarity_and_quality(seed, with_pgan=True)


@registry.register
def image_similarity_and_quality_cv_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity_and_quality with cross-validation."""
    return image_similarity_and_quality(seed, cross_val=True, with_pgan=True)


@registry.register
def double_o_seven(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimization of policies for the 007 game.
    Sequential or 10-parallel or 100-parallel. Various numbers of averagings: 1, 10 or 100."""
    # pylint: disable=too-many-locals
    seedg = create_seed_generator(seed)
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {"mono": rl.agents.Perceptron, "multi": rl.agents.DenseNet}
    agents = {
        a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False)
        for a, m in modules.items()
    }
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
    optimizers: tp.List[tp.Any] = ["PSO", dde, "MetaTuneRecentering", "DiagonalCMA"]
    for num_repetitions in [1, 10, 100]:
        for archi in ["mono", "multi"]:
            for optim in optimizers:
                for env_budget in [5000, 10000, 20000, 40000]:
                    for num_workers in [1, 10, 100]:
                        # careful, not threadsafe
                        runner = rl.EnvironmentRunner(
                            env.copy(), num_repetitions=num_repetitions, max_step=50
                        )
                        func = rl.agents.TorchAgentFunction(
                            agents[archi], runner, reward_postprocessing=lambda x: 1 - x
                        )
                        opt_budget = env_budget // num_repetitions
                        yield Experiment(
                            func,
                            optim,
                            budget=opt_budget,
                            num_workers=num_workers,
                            seed=next(seedg),
                        )


@registry.register
def multiobjective_example(
    seed: tp.Optional[int] = None, hd: bool = False, many: bool = False
) -> tp.Iterator[Experiment]:
    """Optimization of 2 and 3 objective functions in Sphere, Ellipsoid, Cigar, Hm.
    Dimension 6 and 7.
    Budget 100 to 3200
    """
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structure", "structured_moo", seed=next(seedg))
    optims += [
        ng.families.DifferentialEvolution(multiobjective_adaptation=False).set_name("DE-noadapt"),
        ng.families.DifferentialEvolution(crossover="twopoints", multiobjective_adaptation=False).set_name(
            "TwoPointsDE-noadapt"
        ),
    ]
    optims += ["DiscreteOnePlusOne", "DiscreteLenglerOnePlusOne"]
    popsizes = [20, 40, 80]
    optims += [
        ng.families.EvolutionStrategy(
            recombination_ratio=recomb, only_offsprings=only, popsize=pop, offsprings=pop * 5
        )
        for only in [True, False]
        for recomb in [0.1, 0.5]
        for pop in popsizes
    ]

    mofuncs: tp.List[fbase.MultiExperiment] = []
    dim = 2000 if hd else 7
    for name1, name2 in itertools.product(["sphere"], ["sphere", "hm"]):
        mofuncs.append(
            fbase.MultiExperiment(
                [
                    ArtificialFunction(name1, block_dimension=dim),
                    ArtificialFunction(name2, block_dimension=dim),
                ]
                + (
                    [
                        # Addendum for many-objective optim.
                        ArtificialFunction(name1, block_dimension=dim),
                        ArtificialFunction(name2, block_dimension=dim),
                    ]
                    if many
                    else []
                ),
                upper_bounds=[100, 100] * (2 if many else 1),
            )
        )
        mofuncs.append(
            fbase.MultiExperiment(
                [
                    ArtificialFunction(name1, block_dimension=dim - 1),
                    ArtificialFunction("sphere", block_dimension=dim - 1),
                    ArtificialFunction(name2, block_dimension=dim - 1),
                ]
                + (
                    [
                        ArtificialFunction(
                            name1, block_dimension=dim - 1
                        ),  # Addendum for many-objective optim.
                        ArtificialFunction("sphere", block_dimension=dim - 1),
                        ArtificialFunction(name2, block_dimension=dim - 1),
                    ]
                    if many
                    else []
                ),
                upper_bounds=[100, 100, 100.0] * (2 if many else 1),
            )
        )
    for mofunc in mofuncs:
        for optim in optims:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1, 100]:
                    yield Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def multiobjective_example_hd(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of moo with high dimension."""
    return multiobjective_example(seed, hd=True)


@registry.register
def multiobjective_example_many_hd(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of moo with high dimension and more objective functions."""
    return multiobjective_example(seed, hd=True, many=True)


@registry.register
def multiobjective_example_many(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of moo with more objective functions."""
    return multiobjective_example(seed, many=True)


@registry.register
def pbt(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    optimizers = [
        "CMA",
        "TwoPointsDE",
        "Shiwa",
        "OnePlusOne",
        "DE",
        "PSO",
        "NaiveTBPSA",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "PortfolioNoisyDiscreteOnePlusOne",
    ]  # type: ignore
    for func in PBT.itercases():
        for optim in optimizers:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@registry.register
def far_optimum_es(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    optims = get_optimizers("es", "basics", seed=next(seedg))  # type: ignore
    for func in FarOptimumFunction.itercases():
        for optim in optims:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@registry.register
def photonics(seed: tp.Optional[int] = None, as_tuple: bool = False) -> tp.Iterator[Experiment]:
    """Too small for being interesting: Bragg mirror + Chirped + Morpho butterfly."""
    seedg = create_seed_generator(seed)
    optims = get_optimizers("es", "basics", "splitters", seed=next(seedg))  # type: ignore
    for method in ["clipping", "tanh"]:  # , "arctan"]:
        for name in ["bragg", "chirped", "morpho", "cf_photosic_realistic", "cf_photosic_reference"]:
            func = Photonics(name, 60 if name == "morpho" else 80, bounding_method=method, as_tuple=as_tuple)
            for budget in [1e3, 1e4, 1e5, 1e6]:
                for algo in optims:
                    xp = Experiment(func, algo, int(budget), num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@registry.register
def photonics2(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return photonics(seed, as_tuple=True)


@registry.register
def adversarial_attack(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Pretrained ResNes50 under black-box attacked.
    Square attacks:
    100 queries ==> 0.1743119266055046
    200 queries ==> 0.09043250327653997
    300 queries ==> 0.05111402359108781
    400 queries ==> 0.04325032765399738
    1700 queries ==> 0.001310615989515072
    """
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structure", "structured_moo", seed=next(seedg))
    folder = os.environ.get("NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER", None)
    # folder = "/datasets01/imagenet_full_size/061417/val"

    if folder is None:
        warnings.warn(
            "Using random images, set variable NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER to specify a folder"
        )
    for func in imagesxp.ImageAdversarial.make_folder_functions(folder=folder):
        for budget in [100, 200, 300, 400, 1700]:
            for num_workers in [1]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


def pbo_suite(seed: tp.Optional[int] = None, reduced: bool = False) -> tp.Iterator[Experiment]:
    # Discrete, unordered.
    dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
    seedg = create_seed_generator(seed)
    index = 0
    list_optims = [
        "DiscreteOnePlusOne",
        "Shiwa",
        "CMA",
        "PSO",
        "TwoPointsDE",
        "DE",
        "OnePlusOne",
        "AdaptiveDiscreteOnePlusOne",
        "CMandAS2",
        "PortfolioDiscreteOnePlusOne",
        "DoubleFastGADiscreteOnePlusOne",
        "MultiDiscrete",
        "cGA",
        dde,
    ]
    if reduced:
        list_optims = [
            x
            for x in ng.optimizers.registry.keys()
            if "iscre" in x and "ois" not in x and "ptim" not in x and "oerr" not in x
        ]
    for dim in [16, 64, 100]:
        for fid in range(1, 24):
            for iid in range(1, 5):
                index += 1
                if reduced and index % 13:
                    continue
                for instrumentation in ["Unordered"] if reduced else ["Softmax", "Ordered", "Unordered"]:
                    try:
                        func = iohprofiler.PBOFunction(fid, iid, dim, instrumentation=instrumentation)
                    except ModuleNotFoundError as e:
                        raise fbase.UnsupportedExperiment("IOHexperimenter needs to be installed") from e
                    for optim in list_optims:
                        for nw in [1, 10]:
                            for budget in [100, 1000, 10000]:
                                yield Experiment(func, optim, num_workers=nw, budget=budget, seed=next(seedg))  # type: ignore


@registry.register
def pbo_reduced_suite(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return pbo_suite(seed, reduced=True)


def causal_similarity(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Finding the best causal graph"""
    # pylint: disable=import-outside-toplevel
    from nevergrad.functions.causaldiscovery import CausalDiscovery

    seedg = create_seed_generator(seed)
    optims = ["CMA", "NGOpt8", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"]
    func = CausalDiscovery()
    for budget in [100 * 5**k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


def unit_commitment(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Unit commitment problem."""
    seedg = create_seed_generator(seed)
    optims = ["CMA", "NGOpt8", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"]
    for num_timepoint in [5, 10, 20]:
        for num_generator in [3, 8]:
            func = UnitCommitmentProblem(num_timepoints=num_timepoint, num_generators=num_generator)
            for budget in [100 * 5**k for k in range(3)]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


def team_cycling(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Experiment to optimise team pursuit track cycling problem."""
    seedg = create_seed_generator(seed)
    optims = ["NGOpt10", "CMA", "DE"]
    funcs = [Cycling(num) for num in [30, 31, 61, 22, 23, 45]]
    for function in funcs:
        for budget in [3000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=10, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp
