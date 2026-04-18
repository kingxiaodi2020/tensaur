#!/usr/bin/env python3
"""List MuJoCo collidable pairs from contype/conaffinity masks.

Outputs:
1) geom-level allowed pairs
2) body-level allowed pairs
3) split of body-floor vs body-body pairs
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import mujoco


def geom_name(model: mujoco.MjModel, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"


def body_name(model: mujoco.MjModel, bid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"


def is_floor_geom(model: mujoco.MjModel, gid: int) -> bool:
    name = geom_name(model, gid).lower()
    is_plane = int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    return is_plane or ("floor" in name) or ("ground" in name)


def main() -> None:
    parser = argparse.ArgumentParser(description="List collidable pairs for a MuJoCo XML model")
    parser.add_argument(
        "--xml",
        default="/media/di/4441-E469/tensaur-main/src/tensegrity_playground/envs/tensaur/xmls/PleurobotII/pleurobot_tensegrity_0.xml",
        help="Path to MuJoCo XML",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml)
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    # Keep geoms that participate in mask logic (non-(0,0))
    active_geoms: list[tuple[int, str, int, str, int, int]] = []
    floor_body_ids: set[int] = set()

    for gid in range(model.ngeom):
        ctype = int(model.geom_contype[gid])
        caff = int(model.geom_conaffinity[gid])
        if ctype == 0 and caff == 0:
            continue

        bid = int(model.geom_bodyid[gid])
        gname = geom_name(model, gid)
        bname = body_name(model, bid)
        active_geoms.append((gid, gname, bid, bname, ctype, caff))

        if is_floor_geom(model, gid):
            floor_body_ids.add(bid)

    # Geom-level pair check
    geom_pairs: list[tuple[str, str, int, int]] = []
    for g1, g2 in combinations(active_geoms, 2):
        _, n1, b1, _, c1, a1 = g1
        _, n2, b2, _, c2, a2 = g2
        if (c1 & a2) != 0 or (c2 & a1) != 0:
            geom_pairs.append((n1, n2, b1, b2))

    # Body-level pair aggregation
    body_pairs_ids = {(min(b1, b2), max(b1, b2)) for _, _, b1, b2 in geom_pairs if b1 != b2}

    body_floor_pairs: list[tuple[str, str]] = []
    body_body_pairs: list[tuple[str, str]] = []

    for b1, b2 in sorted(body_pairs_ids):
        n1 = body_name(model, b1)
        n2 = body_name(model, b2)
        if b1 in floor_body_ids or b2 in floor_body_ids:
            body_floor_pairs.append((n1, n2))
        else:
            body_body_pairs.append((n1, n2))

    print(f"XML: {xml_path}")
    print(f"ACTIVE_GEOMS: {len(active_geoms)}")
    print(f"GEOM_COLLIDABLE_PAIRS: {len(geom_pairs)}")
    print(f"BODY_COLLIDABLE_PAIRS: {len(body_pairs_ids)}")
    print(f"BODY_FLOOR_PAIRS: {len(body_floor_pairs)}")
    print(f"BODY_BODY_PAIRS: {len(body_body_pairs)}")

    print("\nBODY_FLOOR_PAIRS_START")
    for a, b in body_floor_pairs:
        print(f"{a} <-> {b}")
    print("BODY_FLOOR_PAIRS_END")

    print("\nBODY_BODY_PAIRS_START")
    for a, b in body_body_pairs:
        print(f"{a} <-> {b}")
    print("BODY_BODY_PAIRS_END")


if __name__ == "__main__":
    main()
