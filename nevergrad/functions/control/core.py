# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Trained policies were extracted from https://github.com/modestyachts/ARS
# under their own license. See ARS_LICENSE file in this file's directory

import typing as tp
import numpy as np
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
from .. import base
from .mujoco import GenericMujocoEnv


class BaseFunction(ExperimentFunction):
    """This (abstract) class is a generic wrapper of OpenAI Gym env for policy evaluation.
    Attributes need to be override accordingly to have concrete working function.

    Attributes.
    ------------
    env_name: str
        Gym OpenAI environment name
    policy_dim: tuple
        Shape of the policy
    state_mean: list
        Average state values of multiple independent runs.
        Current implementations use values from https://github.com/modestyachts/ARS
    state_std: list
        Standard deviation of state values of multiple independent runs.
        Current implementations use values from https://github.com/modestyachts/ARS

    Parameters
    -----------
    num_rollouts: int
        number of independent runs.
    random_state: int or None
        random state for reproducibility in Gym environment.
    """

    def __init__(self, num_rollouts: int, activation: str = "tanh", random_state: tp.Optional[int] = None) -> None:
        list_parametrizations = [p.Array(shape=(a,b)).set_name(r"layer_{a}_{b}") for a, b in zip(self.policy_dim[:-1], self.policy_dim[1:])]
        parametrization = p.Instrumentation(*list_parametrizations).set_name(self.env_name)
        super().__init__(self._simulate, parametrization)
        self.num_rollouts = num_rollouts
        self.random_state = random_state
        self.activation = activation
        self.add_descriptors(num_rollouts=num_rollouts)
        if not hasattr(self, "layer_rescaling_coef"): self.layer_rescaling_coef = np.ones(len(self.policy_dim) - 1)
        self._descriptors.pop("random_state", None)  # remove it from automatically added descriptors

    def _simulate(self, x: tp.Tuple) -> float:
        env = GenericMujocoEnv(env_name=self.env_name,
                               state_mean=self.state_mean if len(self.policy_dim) == 2 else None,
                               state_std=self.state_std if len(self.policy_dim) == 2 else None,
                               num_rollouts=self.num_rollouts,
                               activation=self.activation,
                               layer_rescaling_coef=self.layer_rescaling_coef,
                               random_state=self.random_state)
        loss = env(x[0])
        #base.update_leaderboard(f'{self.env_name},{self.parametrization.dimension}', loss, x, verbose=True)
        return loss

    @property
    def env_name(self) -> str:
        raise NotImplementedError

    @property
    def state_mean(self):
        raise NotImplementedError

    @property
    def state_std(self):
        raise NotImplementedError

    @property
    def policy_dim(self):
        raise NotImplementedError

    # pylint: disable=arguments-differ
    def evaluation_function(self, x: tp.Tuple) -> float:  # type: ignore
        # pylint: disable=not-callable
        loss = self.function(x)
        assert isinstance(loss, float)
        base.update_leaderboard(f'{self.env_name},{self.parametrization.dimension}', loss, x, verbose=True)
        return loss


class Ant(BaseFunction):
    env_name = 'Ant-v2'
    policy_dim: tp.Tuple[int, ...] = (111, 8)
    # Values from https://github.com/modestyachts/ARS/tree/master/trained_policies/Ant-v1
    state_mean = [0.556454034343523, 0.9186531687987207, -0.003597273626800536, -0.062027209820968335, -0.22668151226915723, -0.16602131317263671, 0.7987064451368338, -0.08837934771162403, -0.5953573569489272, 0.31458301497655355, -0.5859666391196517, 0.06840799715850679, 0.6996188089127978, 2.957808943475944, 0.13145167715976466, 0.000408239775951425, 0.0006251405790721873, -0.01324649360933024, -0.010626814219628156, -0.0009340884177352034, 0.011235220933040631, -0.0037160116570636036, -0.018631124226896525, -0.00019431878659138357, -0.023557209761883236, 0.013354894059913155, 0.012594611303292104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0002492521170156384, 5.2942688175970466e-06, 4.758743764016147e-06, 0.00023284707214249071, 0.000575762572920161, 0.008342223628535816, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.805817666962799e-05, 2.944689402684725e-05, 4.622284386412991e-05, 2.0463934941826443e-05, -2.982578719158166e-07, 0.0003232789482430665, 0.01190863730406915, -0.028847450862012118, -
                  0.0016049128587483268, -0.011546822405861629, -0.005185736733314698, 0.04395870932465012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00010306612506050966, 4.182155321673672e-05, -3.6599965153626765e-05, 3.7417588383893424e-05, -1.9852387226708935e-05, 0.0004990991660521935, 0.11191092578395258, 0.007451270486499179, -0.043879628923914234, 0.029746731314372638, 0.004893585846331603, 0.1289749574211565, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00011265043476062721, -3.2651266270538316e-05, 4.192724068962384e-05, -3.599135906255506e-05, -5.0593649720061946e-05, 0.0005518224383528348, -0.06107825100454972, 0.06997376760122764, -0.0012712356710709012, -0.04546462254285545, -0.01656209815809629, 0.08931254442546284, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.558307634083136e-05, -6.29738077240061e-05, -6.133269876741076e-05, -4.7529805786119285e-05, 2.6168802203125756e-05, 0.0003865204399435512, -0.08619662966579106, -0.07392835616333841, 0.06349519253475051, 0.04758718422237922, 0.014684068888490893, 0.10487554379090397]
    state_std = [0.09823861308889045, 0.17141367557864626, 0.0923065066441211, 0.09857913658212542, 0.23067414103638456, 0.31929625189221916, 0.2789541347319086, 0.4252996376350357, 0.17273612261821686, 0.3024306630085608, 0.15440628974854603, 0.42709280554821294, 0.21228009754109947, 1.627223228431405, 0.8187247777350416, 0.9552450884883069, 0.9842769566724734, 1.1630255246361634, 0.9579164107073792, 2.1506606935604275, 3.602875428572795, 5.628269094686002, 0.9920290672268143, 2.1223003669071345, 1.0003529831706024, 5.57262446791663, 2.12884125072627, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.04912611953512608, 0.04889164520640186, 0.010828667552457222, 0.07233078284166466, 0.07267150019759298, 0.0909353971096799, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.015251132328387708, 0.01519859897632306, 0.014332713037703749, 0.016533211312666475, 0.01655178210918886, 0.017826147794067864, 0.19608990202601936, 0.20036636045336445, 0.1867258748179935,
                 0.19829593162153397, 0.19688339050234013, 0.20454779790272495, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.018432894986970972, 0.018497451738447097, 0.01722172494351193, 0.020457389878712977, 0.020483738957125595, 0.022133816429764575, 0.3350906849993091, 0.33250359129341234, 0.33900096746510755, 0.3443797803052648, 0.3439163943012149, 0.33448413391786447, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.019536561966571657, 0.0193568339032099, 0.0178721598998528, 0.0215286944065503, 0.021618066835186175, 0.023317064797031488, 0.28256515904161045, 0.2805053208089945, 0.2749447000464192, 0.2834385598495182, 0.2828637149337391, 0.2845542953486725, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.01632940767643436, 0.01639546342576569, 0.015535938645943628, 0.018094969304990823, 0.01806480253410683, 0.019505571384448334, 0.3051507639463494, 0.30417415230423156, 0.29980481621099997, 0.30983874726354155, 0.3104093749664208, 0.30598187568861357]


class Swimmer(BaseFunction):
    env_name = 'Swimmer-v2'
    policy_dim: tp.Tuple[int, ...] = (8, 2)
    # These values are from https://github.com/modestyachts/ARS/tree/master/trained_policies/Swimmer-v1
    state_mean = 0
    state_std = 1


class HalfCheetah(BaseFunction):
    env_name = 'HalfCheetah-v2'
    policy_dim: tp.Tuple[int, ...] = (17, 6)
    # These values are from https://github.com/modestyachts/ARS/tree/master/trained_policies/HalfCheetah-v1
    state_mean = [-0.09292822734440935, 0.07602245420984527, 0.0899374674155617,
                  0.02011249469254209, 0.0798156434227294, -0.08428448356578108,
                  -0.0215013060089565, -0.03109106115312925, 4.822879258044724,
                  -0.053464747098009684, -0.0318163014347514, -0.03715201186578856,
                  -0.1867529885329672, 0.06631679668009596, -0.09290028455365171,
                  0.15043390964540423, 0.2042574041974179]
    state_std = [0.06852462787695646, 0.3584372681186917, 0.3787484779100599,
                 0.36028136793720056, 0.40588221665043894, 0.44399708334861426,
                 0.3060632009780872, 0.3746540859283749, 2.249878345532693,
                 0.7502914419398155, 1.9942769393981352, 8.823299886314189,
                 7.667188604733051, 9.357981552790314, 10.539094922171378,
                 8.2635861157824, 9.381873068915496]


class Hopper(BaseFunction):
    env_name = 'Hopper-v2'
    policy_dim: tp.Tuple[int, ...] = (11, 3)
    # These values are from https://github.com/modestyachts/ARS/tree/master/trained_policies/Hopper-v1
    state_mean = [1.41599384098897, -0.05478601506999949, -0.2552221576338897,
                  -0.2540472105290307, 0.2752508450342385, 2.608895291030614,
                  -0.0085351966531823, 0.006837504545317028, -0.07123674129372454,
                  -0.0504483911501519, -0.45569643721456643]
    state_std = [0.19805723063173689, 0.07824487681531123, 0.17120271410673388,
                 0.32000514149385056, 0.6240188361452976, 0.8281416142079099,
                 1.5191581374558094, 1.1737837205443291, 1.8776124850625862,
                 3.634827607782184, 5.716475201040653]


class Walker2d(BaseFunction):
    env_name = 'Walker2d-v2'
    policy_dim: tp.Tuple[int, ...] = (17, 6)
    # These values are from https://github.com/modestyachts/ARS/tree/master/trained_policies/Walker2d-v1/gait5_reward_11200.npz
    state_mean = [0.8488498617542702, -0.6611547735815038, -1.2396970579853086,
                  -0.8950078841472574, -0.24708567001479317, -0.026436682326840807,
                  -2.0152198942892237, 0.1772209506456087, 7.124879504132625,
                  -0.08177482100506603, -0.0973090698139016, -0.2565691144289943,
                  0.007630772271257735, 0.6811054136606614, -0.02307099876146226,
                  -0.196427427788658, -1.1504456811780719]
    state_std = [0.05610221999432808, 0.11317036984128691, 0.3121976594462603,
                 0.31169452648959445, 0.6468349769198745, 0.05473553450998447,
                 0.2433434422998723, 0.9155127813052794, 3.2146809848738442,
                 0.5296362410124568, 2.2232944517739037, 2.992691886327696,
                 4.95843329244578, 8.56053510942742, 3.039674909483538,
                 4.235747025296144, 8.980030325898877]


class Humanoid(BaseFunction):
    env_name = 'Humanoid-v2'
    policy_dim: tp.Tuple[int, ...] = (376, 17)
    # These values are from https://github.com/modestyachts/ARS/tree/master/trained_policies/Humanoid-v1/policy_reward_11600
    state_mean = [1.223512941151342, 0.9285620267217891, -0.04600612288662621, -0.24887096958019758, -0.22325955719472168, -0.17629463383430644, 0.47841679596165526, -0.18890539170346107, -0.19861972206496825, 0.13014262053110004, -0.03651792561052908, -0.7429555858796192, -0.04746818133811182, 0.011198448910765477, -0.2910494124293159, -1.5267713093717408, -1.0860041310082882, -1.244709380985192, 0.7792656365408227, 1.2563074982027704, 1.077594999103332, -1.5002654383639942, 5.332955853399292, 0.030647237247258364, -0.024606249596252006, -0.3399735071215924, 0.3497503796975279, -0.21264331665159478, -0.046535856125467005, 0.027206740034526798, -0.01843168788695134, -0.06392540632051621, -0.013627498342509871, -0.13801869738264383, -0.17097839147116675, -0.25660044197293674, -0.04987117645475261, -0.11400057167681625, -0.10262052970843882, -0.055106111711606016, -0.07942557939501395, 0.03894579493830569, 0.08768440576334802, 0.060153409243360094, -0.10358948903618044, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.6864405737063366, 1.7076508830342796, 0.09763867145634159, 0.0177693503584045, 0.19197507541602837, -0.13022395106115325, -0.3395871721740677, 0.2562544681872592, 3.5792264913034333, 8.322078939354402, 0.06310964955706921, 0.07021720642108964, 0.018755612200744218, -0.00024661224176565904, -0.02273113952129911, 0.003641050523315394, 0.13493216255476234, -0.02224559995972755, 0.3309184911098229, 2.0357520395249753, 0.039372239564168805, 0.05767216431731622, 0.06351784734223695, -0.002171507224015927, -0.0022645721008887386, 0.002453001129138291, 0.3577885947420132, -0.03661370708511777, 0.0011729236960336075, 5.852787113641054, 0.3218533810252938, 0.23757173763500847, 0.10905681377403795, 0.008554557009504018, -0.002592520618933238, -0.12606972016255838, 0.03745658906944364, -0.6138709131240883, -0.8633463756366405, 4.525556257740668, 0.7663807298580719, 0.7507047423815559, 0.13615329633035106, -0.04288862308679571, -0.1620679984315845, -0.20773659001748804, -0.29522562094396376, -0.4226395778601069, -1.3271240326474132, 2.63249442247751, 0.7723128871941001, 0.8545894755744188, 0.1546490059888952, -0.05508466315544032, -0.25608217015596735, -0.15029310219501313, -0.40902946045662475, -0.23415939268459104, -1.128662676072067, 1.767145867645593, 0.20876401940390307, 0.3496278074348039, 0.17926903865754054, 0.006094541232859462, 0.14443296438594702, 0.0007344993293173703, 0.7734068461356229, 0.05155437153664577, -0.814065770161627, 4.525556257740668, 0.49616999663471184, 0.5529967194894083, 0.0979037368810912, 0.003485975410377934, 0.10490800735541388, 0.0501923166476425, 0.3258194930620621, 0.08316548725040934, -1.0567633678115018, 2.63249442247751, 0.4996283348106809, 0.48641724714175905, 0.04794369090820097, -0.0058817462375533765, -0.0038912269269175967, 0.10011275977805349, 0.019911103116523225, 0.20842188994890332, -0.8158057732039949, 1.767145867645593, 0.26094507244463144, 0.30392082021440503, 0.13265630113878285, -0.05431343766698027, 0.11776893466040421, 0.0899084310891727, -0.34511356582586145, -0.25235943483624973, 0.5768083784781206, 1.594059841558777, 0.2699901240424498, 0.2859992911038798, 0.13466612926092844, -0.05567769331909149, 0.1041584857476096, 0.10643703984120968, -0.2717156620480613, -0.25857766325871656, 0.4908634951002241, 1.198343130581343, 0.4159715826831813, 0.26370098868805497, 0.16162316986601624, -0.0018967799393832562, -0.00784557474517397, -0.1862393523515524, 0.015258213240316195, 0.48040886397289984, 0.6367434103635446, 1.594059841558777, 0.6289351408862088, 0.14525028999224587, 0.5126330919829216, 0.00625267449663808, 0.0015535620305504336, -0.24735990025481183, -0.0039015204723075194, 0.760661207912407, 0.39003480548206326, 1.198343130581343, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.009984320860184852, -0.3726808236676091, 0.24820933826970693, 5.502147872070437, 0.0880912529108143, 0.016142302334884586, -0.2180804235944024, -0.10501848181941069, -0.05474741133691679, 5.327275570497641, -0.06344911018678119, -0.0018479691003698354, 0.1384063661121756, 0.3758516263819816, -0.22938686291078486, 5.236642459391339, -0.08320873140803955, 0.003093592788745551, 0.23325350435603381, -0.3918345829164, 0.2942527417014775, 5.206484767706763, 0.10870508038885734, 0.0981200303221886, -0.9521365950481183, 0.36293547344041416, -0.3384414723392472, 5.910080127581048, 0.26773018061211007, -0.5945929428933572, -0.9521365950481183, 0.36293547344041416, -0.3384414723392472, 5.910080127581048, 0.26773018061211007, -0.5945929428933572, 0.3587232211841977, -0.4104791162839761, 1.5986149190475882, 5.204750377221477, -0.26852433811945536, 0.19813165370374033, -0.3062371808686244, 0.004238222556072027, -2.2209658318223635, 5.742622248085341, 0.6722949974624708, -0.500485113914267, -0.3062371808686244, 0.004238222556072027, -2.2209658318223635, 5.742622248085341, 0.6722949974624708, -0.500485113914267, 0.26012708391359357, 0.08467556295331613, -0.21391009904140584, 5.3210057409790155, 0.03506536347635228, -0.001275684097233967, 0.042059501206005254, -0.23778440296675274, -0.1226403571516149, 5.348137526673431, -0.043272898300492156, 0.06022962996263737, 0.2937088892684222, -0.1977541092634194, -0.28966619577855046, 5.298852322727817, 0.23447012164868147, -0.025647992690023803, 0.3599616142316438, 0.2572452733614258, -0.10288112745228332, 5.19810241527611, 0.20728458029872535, -0.10683192755398696, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29556968580778875, -8.260898269152916, 5.587670071623525, 8.924550388476465, 7.307887810310615, -5.797082988253691, 21.76326786049523, 0.5281607939082064, 3.0078721315484658, -8.25644817359464, 12.92236085784975, -3.07346302346624, -0.15972201114587897, 3.8488401400073924, 2.9013015503137547, -3.0030744277640964, -4.125822702844248, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0005411157531993445, 0.00018425900752185883, -7.056479603492841e-05, 0.0004850329710863729, 0.0013067914565368577, -0.0003632249271021157, -1.499782615828459e-05, 1.5396507342996525e-05, -8.412078417064285e-07, 0.00010005310427990925, 0.000202125250115722, -0.00023887949816504833, 2.358707586858282, -2.038571404847552, -0.4050387837681951, 4.231544859434408, -2.40103925025192, 33.864309172326074, -0.5579333098944332, 0.34076320959977724, -0.08857964550304018, -0.9922056623081122, -1.3998733333878197, 1.0435800890962725, -2.0834375475277884, 1.2667537566732037, -0.30076659164163855, -2.936598660670857, -4.701568329244183, 0.5482891196808485, -10.245820936980875, 7.420004642248826, 0.4255468058894504, 17.167740585723312, 16.175268727949796, 181.48592716060185, 0.11325750561423298, -0.06582083133392769, 0.0526578564471078, 0.2371962961790206, 0.3042510288306412, -0.12644331987411908, 2.523881380476937, -1.547423424434062, 0.33845447817037794, 3.7044012690867003, 5.77270922236946, -1.4549487373900374, 8.414220368253863, -4.740154811733693, 0.37906093062842394, -6.603153276131259, -14.672811393696326, 152.19174733969905, -9.948554129284157e-07, -3.1729733223247383e-07, 1.6278277777934387e-07, -3.6080215121179876e-05, -4.327504944517283e-05, 0.00011092777930590747, 0.0005553154635918675, -0.00021156350367889417, 7.217738550388757e-05, -0.0007368680078812343, -0.001578912969766702, 0.0007730670670347712, -7.128808629768361e-07, 4.5728191857655706e-07, 2.2865347644914397e-07, 9.111110122544389e-06, 1.160719855911783e-05, 9.381773874831127e-06, 5.569740868840155e-06, -1.4891833711621733e-06, 4.342784980667928e-06, 6.777033412499121e-05, 8.971821928675218e-05, 3.889718928290386e-05]
    state_std = [0.027220943204269023, 0.03371739859577961, 0.08672716681743708, 0.05806815850513448, 0.10883032708169692, 0.4016638402629323, 0.08844073386589447, 0.2110629853334749, 0.24960364959583276, 0.3587053640151962, 0.2923748496012834, 0.6022790345863119, 0.10635529785354092, 0.27413209706901964, 0.5359593301373943, 1.1750147170843526, 0.3561241697832425, 0.23480406053326436, 0.14437043059667623, 0.27474052240250313, 0.3243886340303913, 0.18167117971436647, 1.3600379607064053, 0.3218201992448027, 0.5296170780949837, 2.772994615925601, 2.164550492666567, 4.4057685665946, 8.587107721379661, 2.8676340745769813, 4.88090236180869, 6.052503126570467, 7.503054069971461, 8.305551440794792, 16.229904432174102, 4.5671057449517205, 6.114614529333562, 11.845036547829535, 25.913615598243307, 6.252996778447118, 4.662819752229859, 2.8957190183908565, 6.052463628549816, 6.384019209205117, 5.0252937265931035, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.14238959435961462, 0.15193351258039872, 0.02688268828918757, 0.016475850422186383, 0.11543445291824593, 0.0823683956228624, 0.2518946921093844, 0.1719574248033898, 0.17329794081014366, 1.000431359973871e-08, 0.018478908043846144, 0.02158172877456581, 0.005750979917882275, 0.003355592768176608, 0.00950928580095848, 0.009213885716264863, 0.04213918034993743, 0.05463338285150308, 0.05641535066785075, 1.0000747565482142e-08, 0.005257009841282607, 0.013779048588095219, 0.013687172475653606, 0.006885761634218394, 0.009213393296520895, 0.003311783377220073, 0.11257854322792597, 0.11138235337945995, 0.16028902749863805, 1.0004120117325544e-08, 0.0626332661938698, 0.06765339292738642, 0.047407472159472344, 0.025488317118020798, 0.035085467198771765, 0.032714226833568054, 0.15957042246295786, 0.14690742873683094, 0.17841778151136162, 1.0003444225085218e-08, 0.11855546076295732, 0.1601995018375117, 0.05705779799801954, 0.03282487431040264, 0.10613039235022993, 0.04756340733017935, 0.19614743790486447, 0.11434942867009171, 0.13718925851422167, 1.0002730173229638e-08, 0.19787882550074856, 0.18946485032798877, 0.09441604754591658, 0.03393819538720024, 0.12257527328131092, 0.05519359584132488, 0.19627223909340552, 0.0748500807412641, 0.15313598337114612, 1.0001191898980395e-08, 0.033710679183069324, 0.07266497927654184, 0.08869154115095376, 0.02801793963860596, 0.05256426516615806, 0.03856252413251216, 0.2516601273928587, 0.16221167659334101, 0.07653053984565042, 1.0003444225085218e-08, 0.27148907547270107, 0.21859413334486488, 0.06325191955230788, 0.02302418243440717, 0.08862609459915445, 0.08045492694676731, 0.25942521478053826, 0.16343372818748067, 0.3276168047959743, 1.0002730173229638e-08, 0.3741935150587287, 0.37066538654036724, 0.02743098103313232, 0.01985411168739766, 0.0869625119569415, 0.0701596813703393, 0.1636434670512358, 0.08415384433332133, 0.3997864847957144, 1.0001191898980395e-08, 0.05864711018558324, 0.03873655463517765, 0.02074513521037753, 0.014516459355615415, 0.032864206630095566, 0.028424329271307287, 0.09841092395038074, 0.062363602397650215, 0.05955480591755076, 1.0001024762056902e-08, 0.05010976811305272, 0.0371667135860929, 0.022916757861489532, 0.021714149304502807, 0.042049527956671544, 0.026636578083260077, 0.10537753915021739, 0.05734094560989781, 0.045451706430412284, 1.0001132995384249e-08, 0.024747012172012858, 0.044248606007555084, 0.031031649204226817, 0.023018344234887928, 0.02826943388733819, 0.011944649045757363, 0.07241029584495748, 0.05091277302407085, 0.05390791668152994, 1.0001024762056902e-08, 0.08845251074562006, 0.04669266079787208, 0.07326745630888133, 0.07993263635812788, 0.043248320224860116, 0.046195495431723146, 0.1277474869177115, 0.06732868237334777, 0.060832036356245775, 1.0001132995384249e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 3.9109477625812987, 2.726824503866877, 3.0255580693328183, 1.6749492698497939, 1.4795655652623088, 0.568143823199565, 1.8472520684575449, 2.1900181369366662, 6.182629688404042, 1.447357032272057, 0.5200673335962976, 0.5733433997780786, 3.709921882183585, 3.5247172723139735, 6.242286558556476, 1.402497552638402, 0.47761843119554187, 0.6076037954486013, 5.026204300107199, 5.510097585745013, 3.8589660463280833, 1.568759119818162, 0.5917527332763451, 1.029123122193732, 4.2874345506067675, 9.124784725671551, 5.203728864594666, 5.510637312923947, 2.9473125952312884, 1.427647390892429, 4.2874345506067675, 9.124784725671551, 5.203728864594666, 5.510637312923947, 2.9473125952312884, 1.427647390892429, 7.374427871946933, 9.244789166256488, 4.310956745613239, 1.7092202217527879, 1.1301077085172486, 1.3007111382890009, 8.813130907184895, 13.431406143255899, 3.814435203519042, 7.597007883908303, 5.119591868447307, 3.868104648610097, 8.813130907184895, 13.431406143255899, 3.814435203519042, 7.597007883908303, 5.119591868447307, 3.868104648610097, 3.6034477180617595, 2.5595084056197304, 4.7728021487075205, 1.7315994634713594, 2.159881144941995, 1.229858099522896, 3.6150217079294142, 2.7974447506138613, 4.811699918909136, 1.895348859567646, 2.182604565616805, 1.3472341883825647, 3.4523845665883925, 4.336426110196122, 4.04002517042012, 2.902216962903632, 1.8809257426448065, 1.3918353572611668, 4.063872791376027, 5.196521788130874, 3.84390228786295, 2.7695155263972278, 1.8656837839324878, 2.401450349582291, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 39.31823891856991, 30.278412023383265, 35.40810910501004, 37.979941531373115, 36.78424202471111, 104.80727284053687, 58.22875905949396, 36.0392502862179, 36.61813110510474, 104.8525398280304, 77.23646063277525, 7.73032038936829, 9.568131917771131, 7.980462646552434, 8.858260913945225, 8.862554883315907, 8.45701715868272, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 0.18830165214399114, 0.1342474923753114, 0.05608616493243241, 0.3031034157531549, 0.42340271911121846, 0.23239945675276474, 0.036639848667331484, 0.03854555438139759, 0.021014154359306533, 0.36699977573403253, 0.31037005974137877, 0.5009260515131433, 10.623571227644051, 9.684407819966891, 2.8137093873627883, 36.145319299009536, 19.48448754425835, 150.7907242219333, 11.261628724818983, 7.10279219157998, 2.426720375288641, 20.394312177459685, 27.983819034394788, 23.497420472071315, 22.907935979558523, 14.003856465530275, 3.9708956438441114, 32.23167397585521, 51.218607076172326, 6.116705780736451, 37.32494107277529, 86.34020135034994, 13.888911368783146, 110.89589625498978, 62.09513943796971, 508.12193622186663, 4.752152694418997, 2.5225532126056023, 2.1366178081751994, 9.131638795299004, 12.759569014410413, 5.344681451091026, 25.044004502345594, 15.4964832317486, 4.123332097154291, 37.00187110793075, 56.77840690678802, 23.617605924413496, 57.00451427503392, 59.54812478177995, 13.271760340369124, 125.76601693382943, 79.10729615187654, 593.56498013792, 0.009841201295063823, 0.016483578826677756, 0.008428110524717015, 0.08510005348913796, 0.0765137611941976, 0.146946643258817, 0.19104081083076538, 0.13768028999889403, 0.05997267118679376, 0.358176890915227, 0.47261113152738704, 0.3491160050383241, 0.009979212934286682, 0.011861627518704817, 0.009040977965359007, 0.05760460551559038, 0.05535268123974551, 0.05381626379965322, 0.026251746223865467, 0.03118102610736434, 0.023451607610137155, 0.11839197501185912, 0.12161710627794166, 0.11164783758519019]


class NeuroAnt(Ant):
    policy_dim = (111, 50, 8)

    
class NeuroSwimmer(Swimmer):
    policy_dim = (8, 50, 2)


class NeuroHalfCheetah(HalfCheetah):
    policy_dim = (17, 50, 6)


class NeuroHopper(Hopper):
    policy_dim = (11, 50, 3)

    
class NeuroWalker2d(Walker2d):
    policy_dim = (17, 50, 6)

    
class NeuroHumanoid(Humanoid):
    policy_dim = (376, 50, 17)
