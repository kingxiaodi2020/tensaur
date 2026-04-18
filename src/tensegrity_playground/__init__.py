import functools

from mujoco_playground._src.locomotion import register_environment
from tensegrity_playground.envs.tensaur import walk
from tensegrity_playground.envs.tensaur import walk_turn_spot 
from tensegrity_playground.envs.tensaur import walk_side_global
from tensegrity_playground.envs.tensaur import walk_turn_r
from tensegrity_playground.envs.tensaur import run
from tensegrity_playground.envs.tensaur import radius

register_environment(
    "PleurobotWalk",
    functools.partial(run.Walk, task="flat_terrain"),
    run.default_config,
)

register_environment(
    "PleurobotRadius",
    functools.partial(radius.Walk, task="flat_terrain"),
    radius.default_config,
)

register_environment(
    "TensegrityQuadrupedWalk",
    functools.partial(walk.Walk, task="flat_terrain"),
    walk.default_config,
)

# Turn on spot environment
register_environment(
    "TensegrityQuadrupedTurnSpot",
    functools.partial(walk_turn_spot.Walk, task="flat_terrain"),
    walk_turn_spot.default_config,
)

register_environment(
    "TensegrityQuadrupedSidewalk",
    functools.partial(walk_side_global.Walk, task="flat_terrain"),
    walk_side_global.default_config,
)

register_environment(
    "TensegrityQuadrupedTurnRadius",
    functools.partial(walk_turn_r.Walk, task="flat_terrain"),
    walk_turn_r.default_config,
)

