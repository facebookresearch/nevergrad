import json
from . import base


class OptimizerHierarchy:
    """A class providing the functionality to represent the class hierarchy of the Optimizers in Nevergrad in the form of a tree."""

    def __init__(self):
        # fetch list of every class that inherits from either Optimizer or ConfiguredOptimizer both directly and indirectly.
        self.optimizers = inheritors(base.Optimizer).union(inheritors(base.ConfiguredOptimizer))

    def build_hierarchy(self) -> None:
        """Takes exhaustive list of every optimizer in Nevergrad, lists their ancestry
        and then turns that into a tree that more efficiently represents the relationships between classes."""
        classes = []
        for opt in self.optimizers:
            # get ancestors of each optimizers in the form of a list of lists using the "method resolution order" attribute of class.
            ancestors = list(opt.__mro__)  # convert from tuple to list
            ancestors.pop()  # remove overarching "object" class from list because irrelevant
            ancestors_string = []
            for ancestor in ancestors:
                # extract name from classes in list for a more readable output.
                ancestors_string.append(ancestor.__name__)
            classes.append(ancestors_string)

        tree = {}
        tree = build_tree(classes)

        # Write to json file
        write_json(tree)


def inheritors(base_optimizer):
    """Provides every class that inherents from a given class i.e.
    base_optimizer using a DFS-style algorithm to traverse the inheritance tree."""
    # instantiate set that will contain every class that inherits from a given base optimizer
    subclasses = set()
    # stack for DFS starting with the base optimizer i.e. the root node in inheritance tree.
    optimizer_stack = [base_optimizer]
    # while optimizer stack isn't empty
    while optimizer_stack:
        parent = optimizer_stack.pop()  # pop from the top (DFS)
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)  # add any unvisited children to subclass set
                optimizer_stack.append(child)  # and stack to be visited hence allowing for DFS to take place.
    return subclasses


def build_tree(optimizer_ancestor_list):
    """Builds dict-of-dicts that better represents the tree structure in a class hierarchy
    using two for loops to traverse the list of lists and build dictionary as it goes."""
    tree = {}
    # traverse list of optimizer ancestor lists
    for optimizer_ancestors in optimizer_ancestor_list:
        current_tree = tree

        # start from the back of given list
        for key in optimizer_ancestors[::-1]:
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


def write_json(tree):
    """Writes a beautified dict-of-dicts to json file."""
    # writes to same directory
    with open("optimizer_class_hierarchy.json", "w") as f:
        json_string = json.dumps(tree, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        f.write(json_string)
