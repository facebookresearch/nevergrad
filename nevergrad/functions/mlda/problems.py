# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Ideas and reference implementations from:
# - Pascal Kerschke, University of Muenster
# - Marcus Gallagher, University of Queensland
# - Mike Preuss, LIACS, Leiden University

import numpy as np
import scipy.spatial
from nevergrad.parametrization import parameter as p
import nevergrad.common.typing as tp
from ..base import ExperimentFunction
from . import datasets


def _kmeans_distance(points: np.ndarray, centers: np.ndarray) -> float:
    """Computes the distance between points and centers
    after affecting each points to the closest center.
    """
    assert points.shape[1] == centers.shape[1]
    distances = np.sum((points[:, :, None] - centers.T[None, :, :])**2, axis=1)
    affect = np.min(distances, axis=1)
    return float(np.sum(affect))


class Clustering(ExperimentFunction):
    """Cost function of a clustering problem.

    Parameters
    ----------
    points: np.ndarray
        k x n array where k is the number of points and n their coordinates
    num_clusters: int
        number of clusters to find
    """

    def __init__(self, points: np.ndarray, num_clusters: int, rescale: bool = True) -> None:
        self.num_clusters = num_clusters
        self._points = np.array(points, copy=True)
        if rescale:
            self._points -= np.mean(self._points, axis=0, keepdims=True)
            self._points /= np.std(self._points, axis=0, keepdims=True)
        super().__init__(self._compute_distance, p.Array(shape=(num_clusters, points.shape[1])))
        self.register_initialization(points=points, num_clusters=num_clusters, rescale=rescale)
        self._descriptors.update(num_clusters=num_clusters, rescale=rescale)

    @classmethod
    def from_mlda(cls, name: str, num_clusters: int, rescale: bool = True) -> "Clustering":
        """Clustering problem defined in
        Machine Learning and Data Analysis (MLDA) Problem Set v1.0, Gallagher et al., 2018, PPSN
        https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view

        Parameters
        ----------
        name: str
            name of the dataset ("Ruspini", or "German towns")
        num_clusters: int
            number of clusters to find

        Note
        ----
        The MLDA problems are P1.a Clustering("Ruspini", 5) and  P1.b Clustering("German towns", 10)
        """
        assert name in ["Ruspini", "German towns"]
        points = datasets.get_data(name)
        pb = cls(points=points, num_clusters=num_clusters, rescale=rescale)
        assert pb._initialization_kwargs is not None
        del pb._initialization_kwargs["points"]
        pb._initialization_kwargs["name"] = name
        pb._initialization_func = cls.from_mlda  # type: ignore
        pb._descriptors.update(name=name)
        return pb

    def _compute_distance(self, centers: np.ndarray) -> float:
        """Sum of minimum squared distances to closest centroid
        centers must be of size num_clusters x n
        """
        return _kmeans_distance(self._points, centers)


class Perceptron(ExperimentFunction):
    """Perceptron function

    Parameters
    ----------
    x: np.ndarray
        the input data
    y: np.ndarray
        the data to predict from the input data
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        assert x.ndim == 1
        assert y.ndim == 1
        self._x = x
        self._y = y
        super().__init__(self._compute_loss, p.Array(shape=(10,)))
        self.register_initialization(x=x, y=y)

    @classmethod
    def from_mlda(cls, name: str) -> "Perceptron":
        """Perceptron problem defined in
        Machine Learning and Data Analysis (MLDA) Problem Set v1.0, Gallagher et al., 2018, PPSN
        https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view

        Parameters
        ----------
        name: str
            name of the dataset (among "quadratic", "sine", "abs" and "heaviside")

        Note
        ----
        quadratic is coined P2.a, sine P2.b, abs P2.c and heaviside P2.d
        """
        data = datasets.make_perceptron_data(name)
        pb = cls(data[:, 0], data[:, 1])
        pb._descriptors.update(name=name)
        pb._initialization_kwargs = {"name": name}
        pb._initialization_func = cls.from_mlda  # type: ignore
        return pb

    def apply(self, parameters: tp.ArrayLike) -> np.ndarray:
        """Apply the perceptron transform to x using the provided parameters

        Parameters
        ----------
        parameters: ArrayLike
            parameters of the perceptron

        Returns
        -------
        np.ndarray
            transformed data
        """
        parameters = np.array(parameters, copy=False)
        assert parameters.shape == (10,)
        tmp = np.tanh(self._x[:, None] * parameters[None, :3] + parameters[None, 3: 6])
        tmp *= parameters[None, 6: 9]
        output: np.ndarray = np.sum(tmp, axis=1) + parameters[-1]
        return output

    def _compute_loss(self, x: tp.ArrayLike) -> float:
        """Compute perceptron
        """
        gx = self.apply(x)
        return float(np.mean((gx - self._y)**2))


class SammonMapping(ExperimentFunction):
    """Sammon mapping function
    """

    def __init__(self, proximity_array: np.ndarray) -> None:
        self._proximity = proximity_array
        self._proximity_2 = self._proximity**2
        self._proximity_2[self._proximity_2 == 0] = 1  # avoid ZeroDivision (for diagonal terms, or identical points)
        super().__init__(self._compute_distance, p.Array(shape=(self._proximity.shape[0], 2)))
        self.register_initialization(proximity_array=proximity_array)

    @classmethod
    def from_mlda(cls, name: str, rescale: bool = False) -> "SammonMapping":
        """Mapping problem defined in
        Machine Learning and Data Analysis (MLDA) Problem Set v1.0, Gallagher et al., 2018, PPSN
        https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view

        Parameters
        ----------
        name: str
            name of the dataset (among "Virus", "Employees")

        Notes
        -----
        - "Virus" dataset is P3.a and "Employees" dataset is P3.b
        - for "Employees", we use the online proximity matrix
        - for "Virus", we compute a proximity matrix from raw data (no normalization)
        """
        assert name in ["Virus", "Employees"], f'Unkwnown name {name}'
        raw_data = datasets.get_data(name)
        if name == "Employees":
            if rescale:
                raise ValueError("Impossible to rescale with 'Employees'")
            proximity = np.array(raw_data.iloc[:, 1:], dtype=float)  # this is alreary the proximity matrix
            # in this proximity matrix, (5, 31) and (8, 41) are identical
        else:
            if rescale:
                raw_data -= np.mean(raw_data, axis=0, keepdims=True)
                raw_data /= np.std(raw_data, axis=0, keepdims=True)
            proximity = scipy.spatial.distance_matrix(raw_data, raw_data)  # for Virus, the proximity matrix must be computed
        pb = cls(proximity)
        pb._descriptors.update(name=name, rescale=rescale)
        pb._initialization_kwargs = {"name": name, "rescale": rescale}
        pb._initialization_func = cls.from_mlda  # type: ignore
        return pb

    @classmethod
    def from_2d_circle(cls, num_points: int = 12) -> "SammonMapping":
        """Simple test case where the points are in a 2d circle.
        """
        idata = np.exp(np.linspace(0, 2 * np.pi, num_points) * 1j)
        data = np.zeros((num_points, 2))
        data[:, 0] = np.real(idata)
        data[:, 1] = np.imag(idata)
        instance = cls(scipy.spatial.distance_matrix(data, data))
        instance._descriptors.update(name="circle", num_points=num_points)
        instance._initialization_kwargs = {"num_points": num_points}
        instance._initialization_func = cls.from_2d_circle  # type: ignore
        return instance

    def _compute_distance(self, x: np.ndarray) -> float:
        """Compute the Sammon mapping metric for the input data
        """
        proximity = scipy.spatial.distance_matrix(x, x)
        norm_prox = (proximity - self._proximity)**2 / self._proximity_2
        return float(np.sum(norm_prox.ravel()))


class Landscape(ExperimentFunction):
    """Planet Earth Landscape problem defined in
    Machine Learning and Data Analysis (MLDA) Problem Set v1.0, Gallagher et al., 2018, PPSN
    https://drive.google.com/file/d/1fc1sVwoLJ0LsQ5fzi4jo3rDJHQ6VGQ1h/view
    and reverted to look for the minimum value (0). This is problem P4.

    Parameters
    ----------
    transform: None, "gaussian" or "square"
        whether use the image [0, 4319]x[0, 2159] (None) or to use a Gaussian transform.

    Note
    ----
    - the initial image is 4320x2160
    - sampling outside yields a +inf value (except for Gaussian, since large values are mapped to border indices)
    - the image is actually a variant of the one proposed in the article. Indeed, this image
      has a much better z-resolution. It is not directly proportional to the altitude though
      since it is an artificial rescaling to greyscale of a color image.
    """

    def __init__(self, transform: tp.Optional[str] = None) -> None:
        super().__init__(self._get_pixel_value, p.Instrumentation(p.Scalar(), p.Scalar()).set_name("standard"))
        self.register_initialization(transform=transform)
        self._image = datasets.get_data("Landscape")
        if transform == "gaussian":
            variables = list(p.TransitionChoice(list(range(x))) for x in self._image.shape)
            self.parametrization = p.Instrumentation(*variables).set_name("gaussian")
        elif transform == "square":
            stds = (np.array(self._image.shape) - 1.) / 2.
            variables2 = list(p.Scalar(init=s).set_mutation(sigma=s) for s in stds)
            self.parametrization = p.Instrumentation(*variables2).set_name("square")  # maybe buggy, try again?
        elif transform is not None:
            raise ValueError(f"Unknown transform {transform}")
        self._max = float(self._image.max())

    def _get_pixel_value(self, x: float, y: float) -> float:
        z = np.round([x, y])
        if np.min(z) < 0 or z[0] >= self._image.shape[0] or z[1] >= self._image.shape[1]:
            return float("inf")  # could propose other limit conditions
        return float(self._max - self._image[int(z[0]), int(z[1])])
