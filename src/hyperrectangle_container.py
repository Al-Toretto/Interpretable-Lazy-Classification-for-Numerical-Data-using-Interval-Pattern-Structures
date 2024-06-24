from typing import List, Dict, Tuple


from rtree import index
from src.hyperrectangle import Hyperrectangle


class HyperrectangleContrainer:
    def __init__(self, hyperrectangles: Dict[int, Hyperrectangle] = None):
        self.hyperrectangles_dict: Dict[int, Hyperrectangle] = {}
        self.positive_tree: index.Index = None
        self.negative_tree: index.Index = None
        self.positive_keys = []
        self.negative_keys = []
        if hyperrectangles is not None:
            self.build_container(hyperrectangles)

    def _find_all_hyperrectangle_containing_point(
        self, point: Tuple[float, ...] | List[float]
    ) -> List[Hyperrectangle]:
        point_as_hyperrectangle = tuple(
            item for pair in zip(point, point) for item in pair
        )
        return list(self.positive_tree.intersection(point_as_hyperrectangle)) + list(
            self.negative_tree.intersection(point_as_hyperrectangle)
        )

    def build_container(self, hyperrectangles: Dict[int, Hyperrectangle]) -> None:
        prop = index.Property()
        prop.dimension = len(next(iter(hyperrectangles.values())).intervals)
        self.positive_tree = index.Index(properties=prop, interleaved=False)
        self.negative_tree = index.Index(properties=prop, interleaved=False)
        for key, rect in hyperrectangles.items():
            self.insert(key, rect)

    def insert(self, key, hyperrectangle: Hyperrectangle) -> None:
        if self.positive_tree is None or self.negative_tree is None:
            self.build_container({key: hyperrectangle})
            return

        if key in self.hyperrectangles_dict:
            raise KeyError(f"key {key} already exists in the container!")

        self.hyperrectangles_dict[key] = hyperrectangle

        if hyperrectangle.is_positive:
            self.positive_keys.append(key)
            self.positive_tree.insert(key, hyperrectangle.flatten())
        else:
            self.negative_keys.append(key)
            self.negative_tree.insert(key, hyperrectangle.flatten())

    def find_all_hyperrectangle_containing_point(
        self, point: Tuple[float, ...]
    ) -> List[Hyperrectangle]:
        if self.positive_tree is None or self.negative_tree is None:
            return []  # container empty
        hyperrectangle_indices = self._find_all_hyperrectangle_containing_point(point)
        return [self.hyperrectangles_dict[ind] for ind in hyperrectangle_indices]

    def find_id_smallest_hyperrectangle_containing_point(
        self, point: Tuple[float, ...] | List[float]
    ) -> int | None:
        if self.positive_tree is None or self.negative_tree is None:
            return None  # container empty
        hyperrectangle_indices = self._find_all_hyperrectangle_containing_point(point)
        if len(hyperrectangle_indices) == 0:
            return None

        min_hyperrectangle_index = hyperrectangle_indices[0]
        min_volume = self.hyperrectangles_dict[min_hyperrectangle_index].volume

        for k in hyperrectangle_indices:
            if self.hyperrectangles_dict[k].volume < min_volume:
                min_volume = self.hyperrectangles_dict[k].volume
                min_hyperrectangle_index = k

        return min_hyperrectangle_index

    def _find_nearest_id_hyperrectangle_to_point_in_rtree(
        self, point: Tuple[float, ...] | List[float], rtree: index.Index
    ) -> Hyperrectangle | None:
        point_as_hyperrectangle = tuple(
            item for pair in zip(point, point) for item in pair
        )
        nearest_hyperrectangle_indices = list(rtree.nearest(point_as_hyperrectangle))
        return next(iter(nearest_hyperrectangle_indices), None)

    def find_id_nearest_hyperrectangle_to_point(
        self,
        point: Tuple[float, ...] | List[float],
        is_positive: bool | None = None,
        p: int = 2,
    ) -> Hyperrectangle | None:
        if self.positive_tree is None or self.negative_tree is None:
            return None  # container empty
        id_nearest_positive: Hyperrectangle = None
        id_nearest_negative: Hyperrectangle = None
        if is_positive or is_positive is None:
            index = self._find_nearest_id_hyperrectangle_to_point_in_rtree(
                point, self.positive_tree
            )
            id_nearest_positive = index
        if not is_positive:
            index = self._find_nearest_id_hyperrectangle_to_point_in_rtree(
                point, self.negative_tree
            )
            id_nearest_negative = index

        if is_positive:
            return id_nearest_positive
        if is_positive is False:
            return id_nearest_negative

        if id_nearest_positive is None and id_nearest_negative is None:
            return None
        if id_nearest_positive is None:
            return id_nearest_negative
        if id_nearest_negative is None:
            return id_nearest_positive

        distance_to_positive = self.hyperrectangles_dict[
            id_nearest_positive
        ].distance_to_point(point, p)
        distance_to_negative = self.hyperrectangles_dict[
            id_nearest_negative
        ].distance_to_point(point, p)

        return (
            id_nearest_positive
            if distance_to_positive < distance_to_negative
            else id_nearest_negative
        )

    def find_keys_of_intersecting_hyperrectangle_with_given_hyperrectangle(
        self, hyperrectangle: Hyperrectangle, is_positive: bool | None = None
    ) -> List[int]:
        if is_positive:
            return list(self.positive_tree.intersection(hyperrectangle.flatten()))
        elif is_positive is False:
            return list(self.negative_tree.intersection(hyperrectangle.flatten()))
        else:
            return list(
                self.positive_tree.intersection(hyperrectangle.flatten())
            ) + list(self.negative_tree.intersection(hyperrectangle.flatten()))
