import nevergrad.benchmark as benchmark
import nevergrad as ng

registry = benchmark.registry

#{
# 'function': Instance of ArtificialFunction(aggregator='max', block_dimension=2, bounded=False, discrete=False, expo=1.0, function_class='ArtificialFunction', hashing=False, name='sphere', noise_dissymmetry=False, noise_level=1, num_blocks=1, rotation=False, split=False, translation_factor=1.0, useful_dimensions=2, useless_variables=0, zero_pen=False),
# 'constraint_violation': None, 
# 'seed': None, 
# 'optimsettings': Experiment: OnePlusOne<budget=4, num_workers=2, batch_mode=True>, 
# 'result': {'loss': nan, 'elapsed_budget': nan, 'elapsed_time': nan, 'error': ''}, 
# '_optimizer': None}

# optimsettings:
# {'_setting_names': ['optimizer', 'budget', 'num_workers', 'batch_mode'], 'optimizer': 'OnePlusOne', 'budget': 4, 'num_workers': 2, 'executor': <nevergrad.benchmark.execution.MockedTimedExecutor object at 0x30d01caf0>}
allpbs = {}
allbudgets = {}
for k in registry.keys():
    num = 0
    try:
     for u in registry[k]():
        num += 1
        if num > 12:
            print("There are other problems in ", k, ": we stop at ", num, ".")
            break
        loss_function = u.function
        domain = loss_function.parametrization
        try:
            checker = domain._constraint_checkers
        except:
            checker = []
        dimension = domain.dimension
        try:
            shape = domain.value.shape
        except:
            shape = None
        fc = ng.p.helpers.analyze(domain).continuous
        c = domain.has_constraints
        t = domain.hptuning
        b = ng.p.helpers.Normalizer(domain).fully_bounded
        r = domain.real_world
        try:
            no = len(loss_function(domain.sample().value))
        except:
            no = 1  # float has no len
        n = not (ng.p.helpers.analyze(domain).deterministic and domain.function.deterministic)
        short_description = f"Name={k}, dimension={dimension}, shape-of-search-space={shape}, constraints={u.constraint_violation}, specialConstraints={checker}/{c}, fully_continuous={fc}, is_tuning{t}, real_world={r}, noisy={n}, domain={domain}, fully-bounded={b}, number-of-objectives={no}"
        if short_description not in allpbs:  # We use only one of the many problems per name/shape/dimension, but you might want to get plenty of them :-)
            print(short_description)
            full_description = f"Name = {k}, Domain = {domain}, dimension={dimension}, example candidate = {domain.sample().value}, example of loss {loss_function(domain.sample().value)}"
            allpbs[short_description] = full_description
            allbudgets[short_description] = [u.optimsettings.budget]
        else:
            allbudgets[short_description] += [u.optimsettings.budget]
        # print(full_description)  # This might take a lot of room
    except Exception as e:
        print(f"We fail to analyze problem {k} due to {e}")
       
# global overview
for ku in allbudgets:
    print(ku, allbudgets[ku])
    #print(allpbs[ku])  # This takes a lot of room
