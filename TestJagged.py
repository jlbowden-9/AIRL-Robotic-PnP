import numpy as np
import pybullet as p

def attach_jagged_bottom(
    pallet_id: int,
    client: int,
    max_jagged: float = 0.004,     # meters (4mm)
    n_bumps: int = 80,
    bump_xy=(0.02, 0.02),          # meters (size of each bump in X/Y)
    margin: float = 0.01,          # keep bumps away from edges
    x_range=(0.0, 0.56515),        # pallet footprint in base_link frame
    y_range=(-0.60325, 0.0),
    bottom_z: float = 0.0,         # underside plane in base_link frame (your URDF is consistent w/ z=0 bottom)
    min_bump_h: float = 0.0005,    # meters
    patch_mass: float = 0.01,      # small so dynamics barely change
    rng: np.random.Generator | None = None,
):
    """
    Creates a compound collision shape made of many small boxes (bumps) and fixed-attaches it to pallet_id.
    Returns (patch_body_id, constraint_id).
    """
    assert max_jagged > 0
    if rng is None:
        rng = np.random.default_rng()

    sx, sy = bump_xy
    # Sample positions inside footprint
    x_lo = x_range[0] + margin + sx / 2.0
    x_hi = x_range[1] - margin - sx / 2.0
    y_lo = y_range[0] + margin + sy / 2.0
    y_hi = y_range[1] - margin - sy / 2.0
    if x_hi <= x_lo or y_hi <= y_lo:
        raise ValueError("Invalid sampling region: reduce bump_xy or margin.")

    # Build arrays for a compound collision shape
    shapeTypes = [p.GEOM_BOX] * n_bumps
    halfExtents = []
    framePositions = []
    frameOrns = [[0, 0, 0, 1]] * n_bumps

    for _ in range(n_bumps):
        h = float(rng.uniform(min_bump_h, max_jagged))
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(y_lo, y_hi))

        halfExtents.append([sx / 2.0, sy / 2.0, h / 2.0])
        # Put the TOP of the bump at bottom_z (so it protrudes downward)
        framePositions.append([x, y, bottom_z - h / 2.0])

    col_id = p.createCollisionShapeArray(
        shapeTypes=shapeTypes,
        halfExtents=halfExtents,
        collisionFramePositions=framePositions,
        collisionFrameOrientations=frameOrns,
        physicsClientId=client,
    )

    pallet_pos, pallet_orn = p.getBasePositionAndOrientation(pallet_id, physicsClientId=client)

    patch_id = p.createMultiBody(
        baseMass=patch_mass,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=-1,    # invisible (collision only)
        basePosition=pallet_pos,
        baseOrientation=pallet_orn,
        physicsClientId=client,
    )

    # Fixed joint so it moves/rotates with the pallet
    cid = p.createConstraint(
        parentBodyUniqueId=pallet_id,
        parentLinkIndex=-1,
        childBodyUniqueId=patch_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=[0, 0, 0, 1],
        physicsClientId=client,
    )

    # Prevent self-collision between pallet and patch
    p.setCollisionFilterPair(patch_id, pallet_id, -1, -1, enableCollision=0, physicsClientId=client)

    # Match friction/restitution to the pallet (optional but recommended)
    dyn = p.getDynamicsInfo(pallet_id, -1, physicsClientId=client)
    lateralFriction = dyn[1]
    restitution = dyn[5]
    rollingFriction = dyn[6]
    spinningFriction = dyn[7]
    p.changeDynamics(
        patch_id, -1,
        lateralFriction=lateralFriction,
        restitution=restitution,
        rollingFriction=rollingFriction,
        spinningFriction=spinningFriction,
        physicsClientId=client,
    )

    return patch_id, cid