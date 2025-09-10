import functools

from mujoco_playground._src.locomotion import register_environment
from tensegrity_playground.envs.tensaur import walk


register_environment(
    "TensegrityQuadrupedWalk",
    functools.partial(walk.Walk, task="flat_terrain"),
    walk.default_config,
)
