from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod
from enum import Enum

# import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline

class SolidType(Enum):
    Transformer = 1
    Estimator = 2

class OutputType(Enum):
    Regression = 1
    Classification = 2

# class Solid(TransformerMixin):
#     def transform(self, X, **kwargs):
#         print('custom step: transform')
#         return self

#     def fit(self, X, y=None, **kwargs):
#         print('custom step: fit')
#         return self

#     def get_params(self, **kwargs):
#         """ function for using in GRID search
#         """
#         return {}

# ----------------------------------------------------------


class Solid(ABC, TransformerMixin, BaseEstimator):
    """
    The base Component class declares common operations for both simple and
    complex objects of a composition.
    """

    @property
    def parent(self) -> Solid:
        return self._parent

    @parent.setter
    def parent(self, parent: Solid):
        """
        Optionally, the base Component can declare an interface for setting and
        accessing a parent of the component in a tree structure. It can also
        provide some default implementation for these methods.
        """

        self._parent = parent

    """
    In some cases, it would be beneficial to define the child-management
    operations right in the base Component class. This way, you won't need to
    expose any concrete component classes to the client code, even during the
    object tree assembly. The downside is that these methods will be empty for
    the leaf-level components.
    """

    def add(self, component: Solid) -> None:
        pass

    def remove(self, component: Solid) -> None:
        pass

    def is_composite(self) -> bool:
        """
        You can provide a method that lets the client code figure out whether a
        component can bear children.
        """
        return False

    @abstractmethod
    def operation(self) -> str:
        """
        The base Component may implement some default behavior or leave it to
        concrete classes (by declaring the method containing the behavior as
        "abstract").
        """
        pass

    @abstractmethod
    def transform(self, *arg, **kwargs):
        return arg

    @abstractmethod
    def fit(self, *arg, **kwargs):
        return self   

    


class Leaf(Solid):
    """
    The Leaf class represents the end objects of a composition. A leaf can't
    have any children.

    Usually, it's the Leaf objects that do the actual work, whereas Composite
    objects only delegate to their sub-components.
    """
    def __init__(self, desc: str):
        self._desc: str = desc

    def operation(self) -> str:
        return "Leaf"

    def transform(self, *arg, **kwargs):
        print('trans: {}'.format(self._desc))
        return arg[0]

    def fit(self, *arg, ** kwargs) -> self:
        print('fit: {}: {}'.format(self._desc, arg))
        return self

    def get_params():
        return super().get_params()


class Composite(Solid):
    """
    The Composite class represents the complex components that may have
    children. Usually, the Composite objects delegate the actual work to their
    children and then "sum-up" the result.

    Pipeline of transforms with a final estimator.
    """

    def __init__(self) -> None:
        self.steps: List[Solid] = []

    """
    A composite object can add or remove other components (both simple or
    complex) to or from its child list.
    """

    def add(self, component: Solid) -> None:
        self.steps.append(component)
        component.parent = self

    def remove(self, component: Solid) -> None:
        self.steps.remove(component)
        component.parent = None

    def is_composite(self) -> bool:
        return True

    def operation(self) -> str:
        """
        The Composite executes its primary logic in a particular way. It
        traverses recursively through all its children, collecting and summing
        their results. Since the composite's children pass these calls to their
        children and so forth, the whole object tree is traversed as a result.
        """

        # results = []
        # for child in self._children:
        #     results.append(child.operation())

        return make_pipeline(*self.steps).operation()

    def transform(self, *arg, **kwargs):
        print('custom step: transform')
        # return make_pipeline(*self._children)
        return make_pipeline(*self.steps).transform(*arg, **kwargs)
    
    def fit(self, *arg, **kwargs):
        print('custom step: fit')
        return make_pipeline(*self.steps).fit(*arg, **kwargs)


def client_code(component: Solid) -> None:
    """
    The client code works with all of the components via the base interface.
    """

    print(f"RESULT: {component.operation()}", end="")


def client_code2(component1: Solid, component2: Solid) -> None:
    """
    Thanks to the fact that the child-management operations are declared in the
    base Component class, the client code can work with any component, simple or
    complex, without depending on their concrete classes.
    """

    if component1.is_composite():
        component1.add(component2)

    print(f"RESULT: {component1.operation()}", end="")


if __name__ == "__main__":
    # This way the client code can support the simple leaf components...
    simple = Leaf("1")
    print("Client: I've got a simple component:")
    client_code(simple)
    print("\n")

    # ...as well as the complex composites.
    tree = Composite()

    branch1 = Composite()
    branch1.add(Leaf("2"))
    branch1.add(Leaf("3"))

    branch2 = Composite()
    branch2.add(Leaf("4"))

    tree.add(branch1)
    tree.add(branch2)

    print("Client: Now I've got a composite tree:")
    client_code(tree)
    print("\n")

    print("Client: I don't need to check the components classes even when managing the tree:")
    client_code2(tree, simple)
