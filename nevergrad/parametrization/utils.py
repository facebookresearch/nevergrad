# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Descriptors:
    """Provides access to a set of descriptors for the parametrization
    This can be used within optimizers.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        deterministic: bool = True,
        deterministic_function: bool = True,
        continuous: bool = True,
        metrizable: bool = True,
        ordered: bool = True,
    ) -> None:
        self.deterministic = deterministic
        self.deterministic_function = deterministic_function
        self.continuous = continuous
        self.metrizable = metrizable
        self.ordered = ordered

    def __and__(self, other: "Descriptors") -> "Descriptors":
        values = {field: getattr(self, field) & getattr(other, field) for field in self.__dict__}
        return Descriptors(**values)


class NotSupportedError(RuntimeError):
    """This type of operation is not supported by the parameter.
    """
