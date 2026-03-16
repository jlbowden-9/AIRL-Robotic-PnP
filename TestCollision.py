import pybullet as p
import time
import pybullet_data
import numpy as np


import numpy as np
import pybullet as p

def attach_inner_nesting_wall_spheres(
    pallet_id: int,
    client: int,
    n_spheres: int = 60,

    # Random radius and protrusion (meters)
    min_radius: float = 0.002,
    max_radius: float = 0.006,
    min_protrusion: float = -0.01,
    max_protrusion: float = -0.005,


    # Outer
    x_outer_range=(0.0, 0.56515),
    y_outer_range=(-0.60325, 0.0),

    # Inner
    inner_x_size: float = 0.53,
    inner_y_size: float = 0.57,

    # Vertical
    bottom_z: float = 0.0,
    lower_height: float = 0.03,
    margin_z: float = 0.002,

    edge_margin: float = 0.01,
    patch_mass: float = 1e-3,

    debug_points: bool = False,
    rng: np.random.Generator | None = None,
):
    """
    Adds collision-only spheres on the INNER lower walls of the pallet,
    using the nest footprint from the URDF.

    Returns: (patch_body_id, constraint_id)
    """
    if rng is None:
        rng = np.random.default_rng()

    if max_radius <= 0 or min_radius <= 0 or min_radius > max_radius:
        raise ValueError("Bad radius bounds.")
    if min_protrusion < 0 or max_protrusion < 0:
        raise ValueError("Bad protrusion bounds.")

    # ---- Compute inner wall planes in base_link frame ----
    x0, x1 = x_outer_range
    y0, y1 = y_outer_range

    outer_x_len = x1 - x0
    outer_y_len = y1 - y0

    # thickness on each side
    t_x = (outer_x_len - inner_x_size) / 2.0
    t_y = (outer_y_len - inner_y_size) / 2.0
    if t_x <= 0 or t_y <= 0:
        raise ValueError("Inner size larger than outer? Check inner_x_size/inner_y_size.")

    x_in_min = x0 + t_x
    x_in_max = x1 - t_x
    y_in_min = y0 + t_y
    y_in_max = y1 - t_y

    # Conservative radius for z-band margins
    r_safe = max_radius

    # Z band (lower walls only)
    z_lo = bottom_z + margin_z + r_safe
    z_hi = bottom_z + lower_height - margin_z - r_safe
    if z_hi <= z_lo:
        raise ValueError("Z band invalid. Increase lower_height or reduce max_radius/margin_z.")

    # Sampling ranges along each wall (along-wall direction)
    x_along_lo = x_in_min + edge_margin + r_safe
    x_along_hi = x_in_max - edge_margin - r_safe
    y_along_lo = y_in_min + edge_margin + r_safe
    y_along_hi = y_in_max - edge_margin - r_safe
    if x_along_hi <= x_along_lo or y_along_hi <= y_along_lo:
        raise ValueError("XY along-wall region collapsed. Reduce edge_margin or max_radius.")

    # ---- Build compound shape ----
    base_pos, base_orn = p.getBasePositionAndOrientation(pallet_id, physicsClientId=client)
    dx = 0.282575
    dy = 0.301625
    dz = .05
    #base_pos = (x,y,z)

    shapeTypes = [p.GEOM_SPHERE] * n_spheres
    radii = []
    framePositions = []
    frameOrns = [[0, 0, 0, 1]] * n_spheres

    for _ in range(n_spheres):
        # Random radius & protrusion
        r = float(rng.uniform(min_radius, max_radius))
        pmax = min(max_protrusion, r)
        pmin = min(min_protrusion, pmax)
        protr = float(rng.uniform(pmin, pmax))
        inset = r - protr  # distance from inner wall plane to sphere center (inside material)

        # Choose which inner wall: 0=x_in_min, 1=x_in_max, 2=y_in_min, 3=y_in_max
        wall = int(rng.integers(0, 4))
        z = 0

        if wall == 0:
            # inner wall at x_in_min, cavity is x >= x_in_min
            y = float(rng.uniform(y_along_lo, y_along_hi))
            x = x_in_min - inset    # center inside wall (toward outer side)
            framePositions.append([x-dx, y+dy, z-dz])

        elif wall == 1:
            # inner wall at x_in_max, cavity is x <= x_in_max
            y = float(rng.uniform(y_along_lo, y_along_hi))
            x = x_in_max + inset    # center inside wall (toward outer side)
            framePositions.append([x-dx, y+dy, z-dz])

        elif wall == 2:
            # inner wall at y_in_min, cavity is y >= y_in_min
            x = float(rng.uniform(x_along_lo, x_along_hi))
            y = y_in_min - inset    # center inside wall
            framePositions.append([x-dx, y+dy, z-dz])

        else:
            # inner wall at y_in_max, cavity is y <= y_in_max
            x = float(rng.uniform(x_along_lo, x_along_hi))
            y = y_in_max + inset    # center inside wall
            framePositions.append([x-dx, y+dy, z-dz])

        radii.append(r)

    col_id = p.createCollisionShapeArray(
        shapeTypes=shapeTypes,
        radii=radii,
        collisionFramePositions=framePositions,   # base_link frame
        collisionFrameOrientations=frameOrns,
        physicsClientId=client,
    )

    patch_id = p.createMultiBody(
        baseMass=patch_mass,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=-1,
        basePosition=base_pos,
        baseOrientation=base_orn,
        physicsClientId=client,
    )

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

    # No self-collision
    p.setCollisionFilterPair(patch_id, pallet_id, -1, -1, 0, physicsClientId=client)

    # Optional: debug points to verify placement
    if debug_points:
        world_pts = []
        for lp in framePositions:
            wp, _ = p.multiplyTransforms(base_pos, base_orn, lp, [0, 0, 0, 1], physicsClientId=client)
            world_pts.append(wp)
        p.addUserDebugPoints(
            world_pts,
            [[1, 0, 0]] * len(world_pts),
            pointSize=4,
            lifeTime=0,
            physicsClientId=client,
        )

    return patch_id, cid


def draw_aabb_for_link(body_id, link_index):
    aabb_min, aabb_max = p.getAABB(body_id, link_index)
    color = [1, 0, 0]

    corners = [
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]]
    ]

    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    for start, end in edges:
        p.addUserDebugLine(corners[start], corners[end], color, 1)
def draw_rotated_box(center_pos, orientation, half_extents, color=[1,0,0], duration=0):
    # Define 8 corners of the box in local frame
    hx, hy, hz = half_extents
    local_corners = [
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy, -hz],
        [ hx, -hy, -hz]
    ]

    # Get rotation matrix from quaternion
    rot_matrix = p.getMatrixFromQuaternion(orientation)
    rot_matrix = [rot_matrix[0:3], rot_matrix[3:6], rot_matrix[6:9]]  # 3x3

    # Transform each corner into world space
    world_corners = []
    for corner in local_corners:
        rotated = [
            sum(rot_matrix[i][j] * corner[j] for j in range(3)) for i in range(3)
        ]
        world = [center_pos[i] + rotated[i] for i in range(3)]
        world_corners.append(world)

    # Box edges
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    # Draw lines
    for start, end in edges:
        p.addUserDebugLine(world_corners[start], world_corners[end], color, 1, duration)

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
#planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
#boxId = p.loadURDF("Meshes\TestingFingerCollision",startPos, startOrientation, useFixedBase=True)
boxId2 = p.loadURDF("Meshes\Pallet.URDF",[-0.28127, 0.29983, 0], startOrientation, useFixedBase=True)
boxId5 = p.loadURDF("Meshes\Pallet.URDF",[-0.28127, 0.29983, .1143], startOrientation, useFixedBase=False)


'''jagged_patch_id, jagged_cid = attach_inner_nesting_wall_spheres(
    pallet_id=boxId5,
    client=physicsClient,
    n_spheres=60,
    min_radius=0.002,
    max_radius=0.004,       # start small
    min_protrusion=0.000,
    max_protrusion=0.001,
    bottom_z=0.0,
    lower_height=0.02,      # bottom 2 cm
    debug_points=False,

)'''
#boxId3 = p.loadURDF("Meshes\BalenceRod",[0, 0, .5], startOrientation, useFixedBase=True)
#palm = p.loadURDF("Meshes/testingpalm.urdf",startPos, startOrientation, useFixedBase=True)
#pos, orn = p.getBasePositionAndOrientation(boxId)

#draw_rotated_box(pos, orn, half_extents=[0.035, 0.015, 0.025])
#draw_aabb_for_link(boxId,-1)
p.resetDebugVisualizerCamera(
    cameraDistance=.2,
    cameraYaw=90,
    cameraPitch=0,
    cameraTargetPosition=[0,0,.017]
)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId5)
print(cubePos,cubeOrn)
p.disconnect()
