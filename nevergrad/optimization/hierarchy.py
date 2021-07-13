# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from nevergrad import optimizers


class _OptimizerHierarchy:
    """A class providing the functionality to represent the class hierarchy of the Optimizers in Nevergrad in the form of a tree."""

    def __init__(self, path):
        self.path = path

    def _build_hierarchy(self) -> None:
        """Takes exhaustive list of every optimizer instance in Nevergrad, lists their ancestry
        and then turns that into a tree that more efficiently represents the relationships between classes."""
        opt_hierarchies = _get_instance_hierarchies() + _get_class_hierarchies()
        tree = _build_tree(opt_hierarchies)
        # Write to json file
        _write_json(tree, self.path)


def _inheritors(base_optimizer):
    """Provides every class that inherents from a given class i.e.
    base_optimizer using a DFS-style algorithm to traverse the inheritance tree."""
    subclasses = set()
    optimizer_stack = [
        base_optimizer
    ]  # stack for DFS starting with the base optimizer i.e. the root node in inheritance tree.
    while optimizer_stack:
        class_ = optimizer_stack.pop()  # pop from the top (DFS)
        for child in class_.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)  # add any unvisited children to subclass set
                optimizer_stack.append(child)  # and stack to be visited hence allowing for DFS to take place.
    return subclasses


def _get_instance_hierarchies():
    """Gets the class hierarchy of a given instance of optimizer and attaches to the hierarchy
    of its related _OptimizerClass as list of lists corresponding to each instance."""
    opt_hierarchies = []
    for attribute, value in vars(optimizers).items():
        if isinstance(value, optimizers.base.ConfiguredOptimizer):
            hierarchy = list(
                type(value).__mro__
            )  # listed linearization of class that corresponds to type of optimizer
            hierarchy.insert(0, attribute)  # add name of optimizer instance to start of list
            hierarchy.pop()  # remove top "object" class from list because irrelevant
            hierarchy.pop()  # remove "ConfiguredOptimizer" class from end of list
            hierarchy_opt_class = list(value._OptimizerClass.__mro__)  # to get into main optimizer hierarchy
            hierarchy = hierarchy + hierarchy_opt_class
            hierarchy.pop()  # remove top "object" class
            hierarchy_string = []
            for class_ in hierarchy:
                # extract name from classes in list for a more readable output.
                if not isinstance(class_, str):
                    hierarchy_string.append(class_.__name__)
                else:
                    hierarchy_string.append(class_)
            opt_hierarchies.append(hierarchy_string)

    return opt_hierarchies


def _get_class_hierarchies():
    """Gets the class hierarchy of a given leaf class of base.Optimizer and returns list of lists representing
    individual hierarchies."""
    opts = _inheritors(optimizers.base.Optimizer)
    opt_hierarchies = []
    for opt in opts:
        # listed linearization of class that corresponds to type of optimizer
        hierarchy = list(opt.__mro__)
        hierarchy.pop()  # rremove top "object" class from list because irrelevant
        hierarchy_string = []
        for ancestor in hierarchy:
            # extract name from classes in list for a more readable output.
            hierarchy_string.append(ancestor.__name__)
        opt_hierarchies.append(hierarchy_string)
    return opt_hierarchies


def _build_tree(opt_hierarchies):

    """Builds dict-of-dicts that better represents the tree structure in a class hierarchy
    using two for loops to traverse the list of lists and build dictionary as it goes."""
    tree = {}
    # traverse list of optimizer ancestor lists
    for opt_hierarchy in opt_hierarchies:
        current_tree = tree

        # start from the back of given list
        for key in opt_hierarchy[::-1]:
            if key not in current_tree:
                current_tree[
                    key
                ] = (
                    {}
                )  # if element not present in the given level of the tree then add new key to dict to represent a new edge
            current_tree = current_tree[
                key
            ]  # if element already in tree at the given level then "follow edge" down the tree

    return tree


def _write_json(tree, path):
    """Writes a beautified dict-of-dicts to json file."""
    with open(path, "w") as f:
        json_string = json.dumps(tree, sort_keys=True, indent=2)
        f.write(json_string)


def build_hierarchy(path):
    optimizer_hierarchy = _OptimizerHierarchy(path)
    optimizer_hierarchy._build_hierarchy()
