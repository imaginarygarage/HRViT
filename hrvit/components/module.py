from keras import layers
from tensorflow import Tensor
from typing import Any, Optional

from .branch import Branch, BranchDefinition


class Module(layers.Layer):
    """
    A module of the HRViT model.

    Args:
        branch_definitions: a definition for each branch in the module.
        name: Name of the layer.
    """

    def __init__(
        self,
        branch_definitions: tuple[BranchDefinition, ...],
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.branch_definitions = branch_definitions
        self.branches = [
            Branch.from_definition(definition) for definition in branch_definitions
        ]

    def call(self, inputs: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Input shapes: N x (batch_size, height, width, token_depth)
        Output shape: M x (batch_size, height, width, token_depth)
        """
        inputs = (inputs,) if isinstance(inputs, Tensor) else inputs
        outputs = []
        for branch in self.branches:
            outputs.append(branch(inputs))
        return tuple(outputs)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config().copy()
        config.update({
            'branch_definitions': self.branch_definitions,
            'name': self.name,
        })
        return config
