"""
The module implements a variety of composition strategies to combine several heterogeneous estimators into uniform objects.
"""

from ._models_union import ModelsUnion
from ._tutor_m import TutorM, BaseTutor

__all__ = ['ModelsUnion', 'TutorM', 'BaseTutor']
