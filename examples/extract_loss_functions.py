import nevergrad.benchmark as benchmark

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
for k in registry.keys():
    try:
     for u in registry[k]():
        loss_function = u.function
        domain = loss_function.parametrization
        try:
            checker = domain._constraint_checkers
        except:
            checker = []
        dimension = domain.dimension
        try:
            shape = domain.sample().value.shape
        except:
            shape = None
        short_description = f"Name={k}, dimension={dimension}, shape={shape}, constraints={u.constraint_violation}, specialConstraints={checker}"
        if short_description not in allpbs:  # We use only one of the many problems per name/shape/dimension, but you might want to get plenty of them :-)
            print(short_description)
            full_description = f"Name = {k}, Domain = {domain}, dimension={dimension}, example candidate = {domain.sample().value}, example of loss {loss_function(domain.sample().value)}"
            allpbs[short_description] = full_description
        # print(full_description)  # This might take a lot of room
    except Exception as e:
        print(f"We fail to analyze problem {k} due to {e}")
       




