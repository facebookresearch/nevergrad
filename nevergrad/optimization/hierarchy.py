# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import defaultdict
from nevergrad import optimizers


class OptimizerHierarchy:
    """A class providing the functionality to represent the class hierarchy of the Optimizers in Nevergrad in the form of a tree."""

    def __init__(self):
        self._opt_hierarchy_lists = self._get_hierarchy_lists()
        self.hierarchy_tree = self._tree()
        self._build_tree()

    def _get_hierarchy_lists(self):
        """Gets the class hierarchy of a given instance of optimizer and attaches to the hierarchy
        of its related _OptimizerClass as list of lists corresponding to each instance / leaf class inheriting from Optimizer."""
        opt_hierarchies = []
        for attribute, value in vars(optimizers).items():
            if isinstance(value, optimizers.base.ConfiguredOptimizer):
                # class hierarchy of type of instance followed by class hierarchy of instance's _OptimizerClass
                # take lists without "Object" and "ConfiguredOptimizer" at the end
                hierarchy = [*(type(value).__mro__)[:-2], *(value._OptimizerClass.__mro__)[:-1]]
                # convert to readable class names and adding instance name to start of list
                hierarchy_string = [attribute, *map(lambda class_: class_.__name__, hierarchy)]
                opt_hierarchies.append(hierarchy_string)
            elif isinstance(value, type) and issubclass(value, optimizers.base.Optimizer):
                # take list without "Object" at the end
                hierarchy = [*((value).__mro__)[:-1]]
                hierarchy_string = list(map(lambda class_: class_.__name__, hierarchy))
                opt_hierarchies.append(hierarchy_string)
        return opt_hierarchies

    def _build_tree(self):
        """Builds dict-of-dicts that better represents the tree structure in a class hierarchy
        using two for loops to traverse the list of lists and build dictionary as it goes."""
        for opt_hierarchy in self._opt_hierarchy_lists:
            opt_hierarchy.reverse()
            self._insert_into_tree(opt_hierarchy)

    def write_json(self, path):
        """Writes a beautified dict-of-dicts to json file."""
        with open(path, "w") as f:
            json_string = json.dumps(self.hierarchy_tree, sort_keys=True, indent=2)
            f.write(json_string)

    def _tree(self):
        return defaultdict(self._tree)

    def _insert_into_tree(self, hierarchy_list):
        tree = self.hierarchy_tree
        for element in hierarchy_list:
            tree = tree[element]
