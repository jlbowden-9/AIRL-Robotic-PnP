import pybullet as p
import pybullet_data
import time
import math
import traceback
import inspect
import pickle
import gymnasium as gym
import numpy as np
import SimplePalletEnv
from imitation.data.types import Trajectory

#------------- Initialization -------------------------------
GripperState = 0  # 1 is closed and 0 is open
step = 0  # used for program flow
OpenGripperStroke = .0254
CloseGripperStroke = 0 # make zero or less than zero
Ons = [0, 0, 0, 0, 0, 0, 0, 0, 0]
PalletFriction = 1.1
PosGain = 0.04 #.06
VeloGain = 1.45 #1.65
Place_Rotation = .67
contact_markers = []
expert_trajectories = []
current_episode = []
episode_idx = 0
updatePoints = 0
updateHistory = 1
errorTolerance = 0.002
Joint_Place_PT = [0,0,0,0,0,0,0,0,0,0]
current_pick_location = [0,0,0]
current_place_location = [0,0,0]
timestep = 0
pID = 0
#-------------End Initialization-----------------------------
#env = gym.make("PalletStacking-v0")

#-------------Functions---------------------------------
def save_as_trajectory(current_episode):
    # Build final_obs from the *current* sim state (same as you already do)
    robot_id = env.unwrapped.robot_id
    arm_idxs = []
    for i in range(p.getNumJoints(robot_id)):
        jtype = p.getJointInfo(robot_id, i)[2]
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            arm_idxs.append(i)
    arm_states = p.getJointStates(env.unwrapped.robot_id, arm_idxs)
    q_controlled = [s[0] for s in arm_states]  # 9 dims

    ee_state = p.getLinkState(robot_id, env.unwrapped.end_effector_index, computeForwardKinematics=True)
    ee_pos, ee_orn = ee_state[4], ee_state[5]

    # Pallet states (flattened)
    pallet_state_flat = []

    if (pID>=0):
        pos, orn = p.getBasePositionAndOrientation(env.unwrapped.pallets[pID])
        pallet_state_flat.extend(list(pos))  # X Y Z
        pallet_state_flat.extend(list(orn))  # Quaternians
    else:
        pos = [0,0,0]
        orn = [0,0,0,0]
        pallet_state_flat.extend(list(pos))  # X Y Z
        pallet_state_flat.extend(list(orn))  # Quaternians

    desired_flat = list(current_pick_location) + list(current_place_location)

    final_obs = np.array(
        q_controlled + list(ee_pos) + list(ee_orn) + pallet_state_flat + desired_flat,
        dtype=np.float32
    )

    obs_list  = [step[0] for step in current_episode]
    acts_list = [step[1] for step in current_episode]
    obs_list.append(final_obs)

    obs  = np.stack(obs_list)
    acts = np.stack(acts_list)

    traj = Trajectory(obs=obs, acts=acts, infos=None, terminal=True)
    print(f"Built trajectory: {len(obs)} obs / {len(acts)} acts")
    return traj


def log_expert(action):
    # --- Robot + pallet state ---
    robot_id = env.unwrapped.robot_id
    pallet_ids = env.unwrapped.pallets
    arm_idxs = []
    controlled_Action = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        jtype = info[2]
        name  = info[1].decode("utf-8").lower()
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                arm_idxs.append(i)
                #controlled_Action.append(action[i-1])
    arm_states = p.getJointStates(env.unwrapped.robot_id, arm_idxs)

    q_controlled = [s[0] for s in arm_states]  # 9 dimensions


    # End effector
    ee_state = p.getLinkState(robot_id, env.unwrapped.end_effector_index, computeForwardKinematics=True)
    ee_pos, ee_orn = ee_state[4], ee_state[5]

    # Pallet states (flattened)
    pallet_state_flat = []

    if (pID>=0):
        pos, orn = p.getBasePositionAndOrientation(env.unwrapped.pallets[pID])
        pallet_state_flat.extend(list(pos))  # X Y Z
        pallet_state_flat.extend(list(orn))  # Quaternians
    else:
        pos = [0,0,0]
        orn = [0,0,0,0]
        pallet_state_flat.extend(list(pos))  # X Y Z
        pallet_state_flat.extend(list(orn))  # Quaternians

    # Desired pick/place
    # current_pick_location and current_place_location are 3D positions
    desired_flat = list(current_pick_location) + list(current_place_location)

    # Final observation vector
    obs_vec = np.array(
        q_controlled +
        list(ee_pos) +
        list(ee_orn) +
        pallet_state_flat +
        desired_flat,
        dtype=np.float32
    )

    current_episode.append((obs_vec, action))


def attach_object_to_finger(Rid, finger_link, pallet_id):
    """
    Creates a fixed constraint between a finger link and an object
    at the finger's COM location.
    """
    # --- Get finger link state ---
    link_state = p.getLinkState(Rid, finger_link, computeForwardKinematics=True)
    finger_com_world = link_state[0]   # COM position (world)
    finger_orn_world = link_state[1]   # COM orientation (world)
    finger_origin_world = link_state[4]  # link frame origin
    finger_origin_orn = link_state[5]    # link frame orientation

    # --- Get object state ---
    obj_pos, obj_orn = p.getBasePositionAndOrientation(pallet_id)

    # --- Convert world COM to local coords (relative to finger frame) ---
    inv_finger_pos, inv_finger_orn = p.invertTransform(finger_origin_world, finger_origin_orn)
    parent_local_pos, parent_local_orn = p.multiplyTransforms(
        inv_finger_pos, inv_finger_orn,
        finger_com_world, finger_orn_world
    )

    # --- Convert world COM to local coords (relative to object frame) ---
    inv_obj_pos, inv_obj_orn = p.invertTransform(obj_pos, obj_orn)
    child_local_pos, child_local_orn = p.multiplyTransforms(
        inv_obj_pos, inv_obj_orn,
        finger_com_world, finger_orn_world
    )

    # --- Create fixed constraint ---
    cid = p.createConstraint(
        parentBodyUniqueId=Rid,
        parentLinkIndex=finger_link,
        childBodyUniqueId=pallet_id,
        childLinkIndex=-1,  # base link of object
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=parent_local_pos,
        childFramePosition=child_local_pos,
        parentFrameOrientation=parent_local_orn,
        childFrameOrientation=child_local_orn,
    )

    return cid

def safe_load_urdf(filename, position, fixed):
    flags = p.URDF_USE_SELF_COLLISION
    obj_id = p.loadURDF(filename, position, useFixedBase=fixed, flags=flags)
    if obj_id < 0:
        print(f"Failed to load {filename}")
    else:
        print(f"Loaded {filename} with ID {obj_id}")
    return obj_id

def draw_frame(position, orientation, length=.25, duration=0):
    rot_matrix = p.getMatrixFromQuaternion(orientation)
    x_axis = [rot_matrix[0]*length, rot_matrix[1]*length, rot_matrix[2]*length]
    y_axis = [rot_matrix[3]*length, rot_matrix[4]*length, rot_matrix[5]*length]
    z_axis = [rot_matrix[6]*length, rot_matrix[7]*length, rot_matrix[8]*length]

    p.addUserDebugLine(position, [position[0]+x_axis[0], position[1]+x_axis[1], position[2]+x_axis[2]], [1,0,0], 2, duration)
    p.addUserDebugLine(position, [position[0]+y_axis[0], position[1]+y_axis[1], position[2]+y_axis[2]], [0,1,0], 2, duration)
    p.addUserDebugLine(position, [position[0]+z_axis[0], position[1]+z_axis[1], position[2]+z_axis[2]], [0,0,1], 2, duration)

def draw_com_sphere(position, radius=0.04, color=[1, 0, 1], duration=0):
    """Draw a small sphere at the given position to mark the center of mass."""
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color + [1.0]
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position
    )
def CloseGripper():
    # Apply constant inward force
    for Gripper_index in [8, 9, 10]:
        gripper_force = 500.0

        p.setJointMotorControl2(
            env.unwrapped.robot_id,
            Gripper_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=CloseGripperStroke,
            force= gripper_force
        )

def OpenGripper():
    # Apply constant outward force on all three prismatic fingers
    #gripper_force = 200.0
    for Gripper_index in [8, 9, 10]:

        p.setJointMotorControl2(
            env.unwrapped.robot_id,
            Gripper_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=OpenGripperStroke,
            force=500
        )


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

def print_collision_contacts(body_a, body_b):
    contacts = p.getContactPoints(bodyA=body_a, bodyB=body_b)
    if contacts:
        print(f"\nCollision detected between Body {body_a} and Body {body_b}:")
        for c in contacts:
            print(f"LinkA {c[3]} <--> LinkB {c[4]} at position {c[5]}")
def draw_contact_points(bodyA, bodyB):
    global contact_markers

    # clear old markers
    for m in contact_markers:
        p.removeUserDebugItem(m)
    contact_markers = []

    # draw new ones
    contacts = p.getContactPoints(bodyA, bodyB)
    for c in contacts:
        pos = c[5]   # world contact position
        marker = p.addUserDebugLine(
            pos, [pos[0], pos[1], pos[2] + 0.05], [1, 0, 0], 2
        )
        text = p.addUserDebugText("X", pos, textColorRGB=[1,0,0], textSize=1.5)
        contact_markers.extend([marker, text])


def Move_To_Position(tPos):
    joint_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    joint_forces = [
        3000,
        2000,
        2000,
        1000,
        1000,
        1000,
        1000,
        1000,
        300,
        300,
        300
    ]
    MaxVelocities = [3, 2, 2, 2, 5, 5, 10, 5, 5, 5, 5]
    for i, joint_index in enumerate(joint_indices):
        if joint_index > 7:
            continue  # Skip gripper here

        maxforce = joint_forces[i]

        p.setJointMotorControl2(
            env.unwrapped.robot_id,
            joint_index,
            p.POSITION_CONTROL,
            targetPosition=tPos[i],
            force=maxforce,
            maxVelocity=MaxVelocities[i]
        )


def Check_Error(tPos):
    link_state = p.getLinkState(env.unwrapped.robot_id, end_effector_index, computeForwardKinematics=True)
    current_pos = link_state[4]
    error = math.sqrt((tPos[0] - current_pos[0]) ** 2 +
                      (tPos[1] - current_pos[1]) ** 2 +
                      (tPos[2] - current_pos[2]) ** 2)
    return error



#-------------End of Functions---------------------------------



try:
#Build the world
    env = gym.make("SimplePalletStacking-v0", render_mode="human")


    #p.changeDynamics(Pallet17, -1, lateralFriction=PalletFriction)
    #p.changeDynamics(Pallet17, -1, spinningFriction=PalletFriction)
    #p.changeDynamics(robot_id, 6, lateralFriction=PalletFriction)
    #p.changeDynamics(robot_id, 6, spinningFriction=PalletFriction)
    #p.changeDynamics(robot_id, 6, rollingFriction=PalletFriction)

    #p.changeDynamics(robot_id, 7, lateralFriction=PalletFriction)
    #p.changeDynamics(robot_id, 7, spinningFriction=PalletFriction)
    #p.changeDynamics(robot_id, 7, rollingFriction=PalletFriction)
    #p.changeDynamics(robot_id, 7, restitution=.5)

    #p.changeDynamics(robot_id, 8, lateralFriction=PalletFriction)
    #p.changeDynamics(robot_id, 8, spinningFriction=PalletFriction)
    #p.changeDynamics(robot_id, 8, rollingFriction=PalletFriction)
    #p.changeDynamics(robot_id, 8, restitution=.5)

    # Indices of your gripper joints
    Finger = 8  # main finger
    Finger1 = 9  # finger 2
    Finger2 = 10  # finger 3



    # Draw COM for base link
    #base_com = p.getBasePositionAndOrientation(robot_id)[0]
    #draw_com_sphere(base_com)

    # Draw COMs for all links
    num_joints = p.getNumJoints(env.unwrapped.robot_id)
    #for i in range(num_joints):
        #link_state = p.getLinkState(robot_id, i, computeLinkVelocity=0, computeForwardKinematics=1)
        #com_pos = link_state[0]
        #draw_com_sphere(com_pos)


    # Draw collision boundaries NO LONGER NEEDED unless we only want 1
    #for i in range(num_joints-3, num_joints-1):
       # draw_aabb_for_link(robot_id, i)

    # Camera reset with position in meters
    p.resetDebugVisualizerCamera(
        cameraDistance=1.,
        cameraYaw=-230,
        cameraPitch=-20,
        cameraTargetPosition=[.1, .8, 1.18]
    )

    # Draw coordinate frames
    #for obj_id in [robot_id, PalletBase1]:
        #for i in range(p.getNumJoints(obj_id)):
            #link_state = p.getLinkState(obj_id, i)
            #link_pos = link_state[4]
            #link_orn = link_state[5]
            #draw_frame(link_pos, link_orn)
    Fingure1Contraint = p.createConstraint(
        parentBodyUniqueId=env.unwrapped.robot_id,
        parentLinkIndex=Finger,
        childBodyUniqueId=env.unwrapped.robot_id,
        childLinkIndex=Finger1,
        jointType=p.JOINT_GEAR,
        jointAxis=[1, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0]
    )
    p.changeConstraint(Fingure1Contraint, gearRatio=-1, maxForce=1000)
    Fingure2Contraint = p.createConstraint(
        parentBodyUniqueId=env.unwrapped.robot_id,
        parentLinkIndex=Finger,
        childBodyUniqueId=env.unwrapped.robot_id,
        childLinkIndex=Finger2,
        jointType=p.JOINT_GEAR,
        jointAxis=[1, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0]
    )
    p.changeConstraint(Fingure2Contraint, gearRatio=-1, maxForce=1000)


    control_mode_slider = p.addUserDebugParameter("Control Mode (0: Manual, 1: IK)", 0, 1, 0)

    joint_indices = []
    joint_sliders = []

    for i in range(p.getNumJoints(env.unwrapped.robot_id)):
        info = p.getJointInfo(env.unwrapped.robot_id, i)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8")

        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            lower = info[8] if info[8] < info[9] else -3.14
            upper = info[9] if info[8] < info[9] else 3.14

            # Skip gripper joints for the main sliders
            if i in [8, 9, 10]:
                continue

            joint_indices.append(i)
            slider = p.addUserDebugParameter(f"Joint {i} ({joint_name})", lower, upper, 0)
            joint_sliders.append(slider)

    # --- Add a separate slider for the gripper ---
    gripper_slider = p.addUserDebugParameter("Gripper", -1, 1, 0)  # value >0=open, <=0=close

    ik_target_sliders = [
        p.addUserDebugParameter("IK Target X (m)", -2.54, 2.54, 0.127),
        p.addUserDebugParameter("IK Target Y (m)", -2.54, 2.54, 0.127),
        p.addUserDebugParameter("IK Target Z (m)", 0, 5.08, 0.127)
    ]

    for i in range(p.getNumJoints(env.unwrapped.robot_id)):
        if p.getJointInfo(env.unwrapped.robot_id, i)[12].decode("utf-8") == "Palm":
            end_effector_index = i
            break
    Above_Pick = [0.26, 0.77, 2.0]
    Approach_Grip = [0.276225, 0.803275, 1.3]
    Grip = [0.276225, 0.803275, 1.188]  # From CAD and placement in world
    Place_PT = [1, 0, 2.032]
    Approach_Place = [0.27623, -0.875, 2.0]
    Above_Place = [0.27623, -0.875, .85]
    Place = [0.27623, -0.875, 0.74]
    pID = 16
    start_time = time.time()
    while True:
        if updatePoints != updateHistory:
            if updatePoints == 1:
                pID = pID + 2
                Above_Pick[1] = Above_Pick[1] + .3
                Approach_Grip[1] = Approach_Grip[1] + .6223
                Grip[1] = Grip[1] + .6223
                Approach_Place[1] = Approach_Place[1] - .3
                Above_Place[1] = Above_Place[1] - .6223
                Place[1] = Place[1] - .6223
            elif updatePoints == 2:
                pID = pID + 1
                Above_Pick[0] = Above_Pick[0] + .3
                Approach_Grip[0] = Approach_Grip[0] + .59055
                Grip[0] = Grip[0] + .59055
                Approach_Place[0] = Approach_Place[0] + .3
                Above_Place[0] = Above_Place[0] + .59055 #.57785
                Place[0] = Place[0] + .59055
            elif updatePoints == 3:
                pID = pID - 2
                Above_Pick[1] = Above_Pick[1] - .3
                Approach_Grip[1] = Approach_Grip[1] - .6223
                Grip[1] = Grip[1] - .6223
                Approach_Place[1] = Approach_Place[1] + .3
                Above_Place[1] = Above_Place[1] + .6223
                Place[1] = Place[1] + .6223
            elif updatePoints == 4: #Next Layer
                pID = pID - 5
                if pID < 0:
                    step = 100
                Above_Pick[0] = Above_Pick[0] - .3
                Approach_Grip[0] = Approach_Grip[0] - .59055
                Approach_Grip[2] = Approach_Grip[2] - .1143
                Grip[0] = Grip[0] - .59055
                Grip[2] = Grip[2] - .1143
                Approach_Place[0] = Approach_Place[0] -.3
                Above_Place[0] = Above_Place[0] - .59055
                Above_Place[2] = Above_Place[2] + .1143
                Place[0] = Place[0] - .59055
                Place[2] = Place[2] + .1143
                updatePoints = 0

            Above_Pick_Orn = p.getQuaternionFromEuler([0, 1.5708, 0])
            Pick_Low_limits = [-.5, -.25, -0.5, -.2, -3.14, -3.14, -0.254, -0.254, -0.254]
            Pick_High_limits = [0, .5, 1.0, .5, 3.14, 3.14, 0.254, 0.254, 0.254 ]
            RestPoses = Joint_Place_PT
            JointRanges = [1.5, .35, 2.25, 1.5, 3.14, 3.14, 0.254, 0.254, 0.254]
            Joints_Above_Pick = p.calculateInverseKinematics(
                env.unwrapped.robot_id,
                end_effector_index,
                targetPosition=Above_Pick,
                targetOrientation=Above_Pick_Orn,
                lowerLimits=Pick_Low_limits,
                upperLimits=Pick_High_limits,
                jointRanges=JointRanges,
                restPoses=RestPoses,
            )
            Joints_Above_Pick = list(Joints_Above_Pick)
            #Joints_Above_Pick[5] = 0
            #Joints_Above_Pick[6] = OpenGripperStroke
            #Joints_Above_Pick[7] = OpenGripperStroke
            #Joints_Above_Pick[8] = OpenGripperStroke


            Pick_Low_limits = [-1.0, -1.35, -0.75, -.2, -3.14, -3.14, -0.254, -0.254, -0.254]
            Pick_High_limits = [0, 1.7, 1.75, .75, 3.14, 3.14, 0.254, 0.254, 0.254 ]
            RestPoses = Joints_Above_Pick
            JointRanges = [1.5, 2.35, 2.25, 1.5, 3.14, 3.14, 0.254, 0.254, 0.254]
             #From CAD and placement in world
            Joint_Approach_Grip = p.calculateInverseKinematics(
                env.unwrapped.robot_id,
                end_effector_index,
                targetPosition=Approach_Grip,
                targetOrientation=Above_Pick_Orn,
                lowerLimits=Pick_Low_limits,
                upperLimits=Pick_High_limits,
                jointRanges=JointRanges,
                restPoses=RestPoses,
                maxNumIterations=1000,
                residualThreshold=1e-4
            )
            Joint_Approach_Grip = list(Joint_Approach_Grip)
            #Joint_Approach_Grip[5] = 0
            #Joint_Approach_Grip[6] = OpenGripperStroke
            #Joint_Approach_Grip[7] = OpenGripperStroke
            #Joint_Approach_Grip[8] = OpenGripperStroke


            Pick_Low_limits = [-2, -2, -2, -.2, -3.14, -3.14, -0.254, -0.254, -0.254]
            Pick_High_limits = [2, 1.5, 1.5, .5, 3.14, 3.14, 0.254, 0.254, 0.254 ]
            RestPoses = Joint_Approach_Grip
            JointRanges = [2, 2, 2.25, 1.5, 3.14, 3.14, 0.254, 0.254, 0.254]
            Joint_Grip = p.calculateInverseKinematics(
                env.unwrapped.robot_id,
                end_effector_index,
                targetPosition=Grip,
                targetOrientation=Above_Pick_Orn,
                lowerLimits=Pick_Low_limits,
                upperLimits=Pick_High_limits,
                jointRanges=JointRanges,
                restPoses=RestPoses,
                maxNumIterations=1000,
                residualThreshold=1e-4
            )
            Joint_Grip = list(Joint_Grip)
            #Joint_Grip[5] = 0
            #Joint_Grip[6] = OpenGripperStroke
            #Joint_Grip[7] = OpenGripperStroke
            #Joint_Grip[8] = OpenGripperStroke
            # _________________________________________PASS THROUGH_____________________________________________
            Place_Low_limits = [-1.0, 0.0, -0.5, -.2, -3.14, -3.14, -0.254, -0.254, -0.254]
            Place_High_limits = [-2.0, .5, 1.0, .2, 3.14, 3.14, 0.254, 0.254, 0.254 ]
            RestPoses = Joints_Above_Pick
            JointRanges = [1.5, 2.25, 2.25, 1.5, 3.14, 3.14, 0.254, 0.254, 0.254]


            Joint_Place_PT = p.calculateInverseKinematics(
                env.unwrapped.robot_id,
                end_effector_index,
                targetPosition=Place_PT,
                targetOrientation=Above_Pick_Orn,
                maxNumIterations=1000,
                residualThreshold=1e-4
            )
            Joint_Place_PT = list(Joint_Place_PT)
            #Joint_Place_PT[5] = Place_Rotation
            #Joint_Place_PT[6] = CloseGripperStroke
            #Joint_Place_PT[7] = CloseGripperStroke
            #Joint_Place_PT[8] = CloseGripperStroke

    # _________________________________________APPROACH PLACE PALLET____________________________________________________
# Nedded  limits the inverse kinamatics is solving for wrong arm orientation
            Place_Low_limits = [-2.97, 0.0, -1.5, -1.5, -3.14, -3.14, -0.254, -0.254, -0.254]
            Place_High_limits = [-2.0, 1.75, 1.5, 1.5, 3.14, 3.14, 0.254, 0.254, 0.254 ]
            RestPoses = Joint_Place_PT
            JointRanges = [.25, 2.25, 2.25, 1.5, 3.14, 3.14, 0.254, 0.254, 0.254]

            Above_Place_Orn = p.getQuaternionFromEuler([0, 1.5708, 0])

            Joint_Approach_Place = p.calculateInverseKinematics(
                env.unwrapped.robot_id,
                end_effector_index,
                targetPosition=Approach_Place,
                targetOrientation=Above_Place_Orn,
                lowerLimits=Place_Low_limits,
                upperLimits=Place_High_limits,
                jointRanges=JointRanges,
                restPoses=RestPoses,
                maxNumIterations=1000,
                residualThreshold=1e-4
            )
            Joint_Approach_Place = list(Joint_Approach_Place)
            #Joint_Approach_Place[6] = 3.14 + Joint_Approach_Place[0]
            #Joint_Approach_Place[6] = CloseGripperStroke
            #Joint_Approach_Place[7] = CloseGripperStroke
            #Joint_Approach_Place[8] = CloseGripperStroke


        # _________________________________________ABOVE PLACE PALLET_____________________________________________________
        # Nedded  limits the inverse kinamatics is solving for wrong arm orientation
            Place_Low_limits = [-2.97, 0.0, -1.5, -1.5, -3.14, -3.14, -0.254, -0.254, -0.254]
            Place_High_limits = [-2.0, 1.75, 1.75, 1.5, 3.14, 3.14, 0.254, 0.254, 0.254 ]
            RestPoses = Joint_Approach_Place
            JointRanges = [.55, 2.25, 2.25, 1.5, 3.14, 3.14, 0.254, 0.254, 0.254]

            Joint_Above_Place = p.calculateInverseKinematics(
                env.unwrapped.robot_id,
                end_effector_index,
                targetPosition=Above_Place,
                targetOrientation=Above_Place_Orn,
                lowerLimits=Place_Low_limits,
                upperLimits=Place_High_limits,
                jointRanges=JointRanges,
                restPoses=RestPoses,
                maxNumIterations=10000,
                residualThreshold=1e-4
            )
            Joint_Above_Place = list(Joint_Above_Place)
            #Joint_Above_Place[5] = 3.14 + Joint_Above_Place[0]
            #Joint_Above_Place[6] = CloseGripperStroke
            #Joint_Above_Place[7] = CloseGripperStroke
            #Joint_Above_Place[8] = CloseGripperStroke

        #_________________________________________PLACE PALLET_____________________________________________________
            Place_Low_limits = [-2.97, -1.0, -1.5, -1.5, -3.14, -3.14, -0.254, -0.254, -0.254]
            Place_High_limits = [-2.0, 1.75, 1.0, 1.5, 3.14, 3.14, 0.254, 0.254, 0.254]
            RestPoses = Joint_Above_Place
            JointRanges = [.55, 2.25, 2.25, 2.25, 3.14, 3.14, 0.254, 0.254, 0.254]


            Joint_Place= p.calculateInverseKinematics(
                env.unwrapped.robot_id,
                end_effector_index,
                targetPosition=Place,
                targetOrientation=Above_Place_Orn,
                lowerLimits=Place_Low_limits,
                upperLimits=Place_High_limits,
                jointRanges=JointRanges,
                restPoses=RestPoses,
                maxNumIterations=10000,
                residualThreshold=1e-4
            )
            Joint_Place = list(Joint_Place)
            #Joint_Place[6] = CloseGripperStroke
            #Joint_Place[7] = CloseGripperStroke
            #Joint_Place[8] = CloseGripperStroke

    # _________________________________________END of POINTS_____________________________________________________
            updateHistory = updatePoints

        mode = p.readUserDebugParameter(control_mode_slider)

        if mode == 0:
            # Arm joints (skip gripper)
            action = np.zeros(9)
            for i, joint_index in enumerate(joint_indices):
                if joint_index in [8, 9, 10]:
                    continue  # Skip gripper here

                target_pos = p.readUserDebugParameter(joint_sliders[i])
                #target_pos = Joint_Grip[i]
                force = 200
                p.setJointMotorControl2(
                    env.unwrapped.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=force
                )

            gripper_target = p.readUserDebugParameter(gripper_slider)
            if gripper_target >= 0:
                OpenGripper()

            else:
                CloseGripper()

            #action[6] = action[7] = action[8] = OpenGripperStroke
        elif mode == 1:
            target_pos = [p.readUserDebugParameter(s) for s in ik_target_sliders]
            joint_poses = p.calculateInverseKinematics(env.unwrapped.robot_id, end_effector_index, target_pos, targetOrientation=Above_Pick_Orn)
            for i, joint_index in enumerate(joint_indices):
                if i < len(joint_poses):
                    p.setJointMotorControl2(env.unwrapped.robot_id, joint_index, p.POSITION_CONTROL, targetPosition=joint_poses[i], force=500)
        else:
            joint_forces = {
                0: 3000,
                1: 2000,
                2: 2000,
                3: 1000,
                4: 1000,
                5: 1000,
                6: 1000,
                7: 1000,
                8: 300,
                9: 300,
                10: 300,
            }
            if step == 0:

                OpenGripper()
                Move_To_Position(Joints_Above_Pick)
                error = Check_Error(Above_Pick)
                action = Joints_Above_Pick
                action[6] = action[7] = action[8] = OpenGripperStroke
                current_pick_location = Grip
                current_place_location = Place
                print(error)
                if error < .01:
                    step = 1

            elif step == 1:  # move to Approach grip pallet 1 position
                Move_To_Position(Joint_Approach_Grip)
                error = Check_Error(Approach_Grip)
                OpenGripper()
                action = Joint_Approach_Grip
                action[6] = action[7] = action[8] = OpenGripperStroke
                current_pick_location = Grip
                current_place_location = Place
                print(error)
                if error < errorTolerance:
                    step = 2
            elif step == 2:  # move Grip pallet 1
                Move_To_Position(Joint_Grip)
                error = Check_Error(Grip)
                OpenGripper()
                action = Joint_Grip
                action[6] = action[7] = action[8] = OpenGripperStroke
                current_pick_location = Grip
                current_place_location = Place
                print(error)
                if error < errorTolerance:
                    step = 3
            elif step == 3:

                Move_To_Position(Joint_Grip)
                CloseGripper()
                action = Joint_Grip
                action[6] = action[7] = action[8] = CloseGripperStroke
                current_pick_location = [0,0,0]
                current_place_location = Place
                joint_state = p.getJointState(env.unwrapped.robot_id, Finger)
                current_pos = joint_state[0]
                target_pos = CloseGripperStroke
                error = abs(current_pos - target_pos)
                print(error)
                if error < .01:
                    print('Closing Gripper')
                    if (Ons[0] == 0):
                        StartTime = time.time()
                        Ons[0] = 1
                    currentTime = time.time()
                    if (currentTime - StartTime) > 1.2:
                        p.setCollisionFilterPair(env.unwrapped.robot_id, env.unwrapped.pallets[pID], Finger, -1, enableCollision=0)
                        p.setCollisionFilterPair(env.unwrapped.robot_id, env.unwrapped.pallets[pID], Finger1, -1, enableCollision=0)
                        p.setCollisionFilterPair(env.unwrapped.robot_id, env.unwrapped.pallets[pID], Finger2, -1, enableCollision=0)
                        p2f = attach_object_to_finger(env.unwrapped.robot_id, Finger-1, env.unwrapped.pallets[pID])
                        step = 4
            elif step == 4:  # move back up
                Move_To_Position(Joint_Approach_Grip)
                error = Check_Error(Approach_Grip)
                CloseGripper()
                action = Joint_Approach_Grip
                action[6] = action[7] = action[8] = CloseGripperStroke
                current_pick_location = [0,0,0]
                current_place_location = Place
                print(error)
                if error < errorTolerance:
                    step = 5
            elif step == 5:  # move back up
                Move_To_Position(Joints_Above_Pick)
                error = Check_Error(Above_Pick)
                CloseGripper()
                action = Joints_Above_Pick
                action[6] = action[7] = action[8] = CloseGripperStroke
                current_pick_location = [0,0,0]
                current_place_location = Place
                print(error)
                if error < .1:
                    step = 6

            elif step == 6:  # Moving to intermediate point
                Move_To_Position(Joint_Place_PT)
                error = Check_Error(Place_PT)
                action = Joint_Place_PT
                action[6] = action[7] = action[8] = CloseGripperStroke
                current_pick_location = [0,0,0]
                current_place_location = Place
                print(error)
                if error < errorTolerance:
                    step = 7
            elif step == 7:  # Moving to abovr place point
                Move_To_Position(Joint_Approach_Place)
                error = Check_Error(Approach_Place)
                action = Joint_Approach_Place
                action[6] = action[7] = action[8] = CloseGripperStroke
                current_pick_location = [0,0,0]
                current_place_location = Place
                print(error)
                if error < .05:
                    step = 8
            elif step == 8:  # Moving to apparch place
                Move_To_Position(Joint_Above_Place)
                error = Check_Error(Above_Place)
                action = Joint_Above_Place
                action[6] = action[7] = action[8] = CloseGripperStroke
                current_pick_location = [0,0,0]
                current_place_location = Place
                print(error)
                if error < errorTolerance:
                    step = 9
            elif step == 9:  # Moving to Place
                Move_To_Position(Joint_Place)
                error = Check_Error(Place)
                Joint_Above_Place_Old = Joint_Above_Place[:]
                Above_Place_Previous = Above_Place[:]
                action = Joint_Place
                action[6] = action[7] = action[8] = CloseGripperStroke
                current_pick_location = [0,0,0]
                current_place_location = Place
                print(error)
                if error < errorTolerance:
                    if (Ons[1] == 0):
                        link_state = p.getLinkState(env.unwrapped.robot_id, Finger, computeForwardKinematics=True)
                        current_pos = link_state[4]
                        StartTime = time.time()
                        Ons[1] = 1

                    p.removeConstraint(p2f)

                    OpenGripper()
                    action = Joint_Place
                    action[6] = action[7] = action[8] = OpenGripperStroke
                    if time.time() - StartTime > 1.5:
                        print('Opening Gripper')
                        updatePoints = updatePoints + 1
                        step = 10
            elif step == 10:  # Moving up from Place
                Move_To_Position(Joint_Above_Place_Old)
                error = Check_Error(Above_Place_Previous)
                OpenGripper()
                action = Joint_Above_Place
                action[6] = action[7] = action[8] = OpenGripperStroke
                current_pick_location = Grip
                current_place_location = Place

                print(error)
                if error < .005:
                    step = 11
            elif step == 11:
                Move_To_Position(Joint_Place_PT)
                error = Check_Error(Place_PT)
                OpenGripper()
                action = Joint_Place_PT
                action[6] = action[7] = action[8] = OpenGripperStroke
                current_pick_location = Grip
                current_place_location = Place
                print(error)
                if error < .005:
                    # ---- EPISODE BOUNDARY HERE ----
                    traj = save_as_trajectory(current_episode)
                    expert_trajectories.append(traj)
                    current_episode = []  # clear buffer for next episode
                    episode_idx += 1
                    print(f"Episode {episode_idx} saved with {len(traj.acts)} steps")

                    # New Episode
                    step = 0
            elif step >= 100: # used to go home
                Move_To_Position(Joints_Above_Pick)
                action = Joints_Above_Pick
                action[6] = action[7] = action[8] = OpenGripperStroke
                current_pick_location = [0,0,0]
                current_place_location = [0,0,0]
                error = Check_Error(Above_Pick)
                print(error)
                if error < .01:
                    # --- Save expert trajectories ---
                    if expert_trajectories:
                        with open("expert_trajs.pkl", "wb") as f:
                            pickle.dump(expert_trajectories, f)
                        print(f"Wrote {len(expert_trajectories)} episodes to expert_trajs.pkl")
                    step = 99

        #draw_contact_points(Pallet17, PalletBase2)
        p.stepSimulation()
        #time.sleep(1.0 / 240) #240
        if (mode > 0) and (mode < 1):
            timestep = timestep+1
            if timestep >20500:
                print(timestep)
        log_expert(action)
        if len(current_episode) == 1:
            print("obs length:", len(current_episode[0][0]),
          "act length:", len(current_episode[0][1]),
          "env obs space:", env.unwrapped.observation_space.shape,
          "env act space:", env.unwrapped.action_space.shape)

except Exception as e:
    print("Exception")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {e}")

    # Print full traceback
    print("\n--- Full Traceback ---")
    traceback.print_exc()

    print("\n--- Local Debug Info ---")

    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals
    for var_name, value in local_vars.items():
        try:
            print(f"{var_name} = {value}")
        except:
            print(f"{var_name} = <unprintable>")


