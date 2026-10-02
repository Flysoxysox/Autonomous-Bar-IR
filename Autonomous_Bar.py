import time

from math import pi

import numpy as np

import swift

from roboticstoolbox import jtraj

from spatialgeometry import Cuboid, Cylinder

from spatialmath import SE3

from ir_support.robots import UR3e

from ir_support_extra_parts.parts import part_mesh


#traj and sim setting used to create a smooth movement for the robots and the mobile robot also can adjust the swift frame time
arm_steps = 30
frame_time = 0.02


#cartesian clearances used to aproach lift and hand the bottle over
approach_height = 0.12
lift_height = 0.18
table_height = 0.62


#bottle offsets so and other data that allows for the ik solver to target the bottle and ensure it is downards and at a suitable distance from the end effector of the robot
bottle_height = 0.242
ur3e_tool_gap = 0.015
ur3e_bottle_offset = bottle_height + ur3e_tool_gap
bottle_grab_height = 0.19


#main locations of the bottle and the ur3e base
handover_spot = np.array([0.68, 0.18, table_height])
final_spot = np.array([1.192, -0.288, table_height-0.1])
ur3e_base_position = np.array([0.95, 0.00, table_height])


#position of the robots joint angles at home
ur3e_home = np.array([0, -pi / 2, pi / 2, -pi / 2, -pi / 2, 0])


#a fixed pos to ensure the movement inbetween pickup and handover is more repeatable and reliable to prevent the solver from colliding the bottle into the robot or the table. 
ur3e_transfer_pose = np.array([0,-pi / 2,0,-pi / 2,-pi / 2,0,])


# the func to add boxes to the scene
def add_box(env, size, position, color):
    box = Cuboid(
        size,
        #3d transform at the center of the box it has no rotation
        pose=SE3.Trans(position[0], position[1], position[2]),
        color=color,
    )
    env.add(box)
    return box


#where the scene is created by adding boxes to the environment.
def build_scene(env):

    table_length = 1.15
    table_width = 0.88
    table_centre_x = 0.98

    add_box(
        env,
        [table_length, table_width, 0.05],
        [table_centre_x, 0.00, table_height - 0.025],
        "gray",
    )

    leg_height = table_height - 0.05
    #adds four legs using a nested loop so bassically 2 times 2 equaks four using the table dimensions
    for x in [0.47, 1.49]:
        for y in [-0.38, 0.38]:
            add_box(
                env,
                [0.06, 0.06, leg_height],
                [x, y, leg_height / 2],
                "dimgray",
            )


#adds the coaster and imports the beer glass model from the ir package
def create_final_table_props(env):
    x, y, z = final_spot

    coaster = Cylinder(
        0.045, #radius
        0.006, #height
        pose=SE3.Trans(x, y, z + 0.003), # the center is set to the final position
        color="steelblue",
    )
    env.add(coaster)

    beer_glass = part_mesh("BeerGlass")
    beer_glass.T = SE3.Trans(
        handover_spot[0],
        handover_spot[1],
        handover_spot[2],
    ).A

    env.add(beer_glass)

    return coaster, beer_glass


# I use this to check that the calc joint numbers are within the robots joint limits
def check_joint_limits(robot, joints):
    #places the joints into a numpy array and reshapes it to be a 1d array
    joints = np.asarray(joints, dtype=float).reshape(-1)

    #if nothing dont check
    if getattr(robot, "qlim", None) is None:
        return True

    limits = np.asarray(robot.qlim, dtype=float)

    #ensure the number of limit data is the same as the number of joints defined
    if limits.shape != (2, len(joints)):
        return True

    #check if any of the joints are below or above their limits, if so return false
    below = np.isfinite(limits[0]) & (joints < limits[0])
    above = np.isfinite(limits[1]) & (joints > limits[1])

    if np.any(below) or np.any(above):
        return False

    return True


#this ensures the caluclated joint angles can be kept close to the robots current joint angles, so it doesnt have to move a lot to get to the new position bassically just makes the movement simpler
def stabilise_joint_angles(robot, joints, reference_joints):
    #converts both the joints and reference joints into numpy arrays and reshapes them to be 1d arrays
    joints = np.asarray(joints, dtype=float).reshape(-1).copy()
    reference_joints = np.asarray(reference_joints, dtype=float).reshape(-1)

    #return the joints without change if the configs are not the same size for some reason
    if len(joints) != len(reference_joints):
        return joints

    #grab the joint limits
    limits = getattr(robot, "qlim", None)

    if limits is not None:
        limits = np.asarray(limits, dtype=float)

        #wrong shape ignore the limits
        if limits.shape != (2, len(joints)):
            limits = None

    #check each joint sequentially and generate a list of physically equivalent angles for each joint, then choose the one that is closest to the reference joint angle
    for joint in range(len(joints)):
        candidates = [
            joints[joint] + 2 * pi * turn
            for turn in range(-2, 3)
        ]

        #remove any limits that are outside the joint limits if they exist
        if limits is not None:
            lower = limits[0, joint]
            upper = limits[1, joint]

            valid = []

            for candidate in candidates:
                lower_ok = (not np.isfinite(lower)) or candidate >= lower
                upper_ok = (not np.isfinite(upper)) or candidate <= upper

                if lower_ok and upper_ok:
                    valid.append(candidate)

            if valid:
                candidates = valid

        #this bassicaly choosese the angle closest to the robots current joint angle
        joints[joint] = min(
            candidates,
            key=lambda candidate: abs(candidate - reference_joints[joint]),
        )

    return joints


#solves the inverse kin for a target xyz by returning the joint angles!
def solve_position_ik(robot, position, start_joints, label):

    #convert the target pos and the intial joint angles into numpy arrays and reshape them to be 1d arrays
    position = np.asarray(position, dtype=float)
    start_joints = np.asarray(start_joints, dtype=float).reshape(-1)

    #this is the actual target for the end effector transform from the required xyz this has no rotation 
    target = SE3.Trans(position[0], position[1], position[2])

    #use the ikine function to caluclate the required joint angles with the ik solver to reach the target pos
    solution = robot.ikine_LM(
        target,
        q0=start_joints,
        mask=[1, 1, 1, 0, 0, 0], #the mask only controls the xyz and so the orientation is not considered in this case
    )

    #stop if there is no solution found and raise an error
    if not getattr(solution, "success", True):
        raise RuntimeError(
            f"IK failed for {label} at {np.round(position, 3)}"
        )

    # grab the angles calculated by the ik solver and reshape them toan array then use the stabilise joint function to ensure the angles are as close as possible to the robots current joint angles
    joints = np.asarray(solution.q, dtype=float).reshape(-1)
    joints = stabilise_joint_angles(
        robot,
        joints,
        start_joints,
    )

    #final check to see if the angles are within the limits of the robot raise an error if found so
    if not check_joint_limits(robot, joints):
        raise RuntimeError(f"{label} violates joint limits.")

    return joints


#i chose to create another ik function to ensure the bottle is remained downards to avoid the bottle from collding with other objects or even the sb6 its self
def solve_downward_tool_ik(robot, position, start_joints, label):
    position = np.asarray(position, dtype=float)
    start_joints = np.asarray(start_joints, dtype=float).reshape(-1)

    #same as before except this time the tool is rotated downards by 180 degrees along the x axis 
    target = (
        SE3.Trans(position[0], position[1], position[2])
        * SE3.Rx(pi)
    )

    #solve the ik with the same mask as before 
    solution = robot.ikine_LM(
        target,
        q0=start_joints,
        mask=[1, 1, 1, 1, 1, 0],
    )

    #if the first attmept at a solution fails the function will try to solve the ik again but this time it will use the position only ik function to get a better guess at a solution
    if not getattr(solution, "success", True):
        position_joints = solve_position_ik(
            robot,
            position,
            start_joints,
            label + " (position seed)",
        )

        #retry the ik solver with the new guess at a solution and this time it will also consider the orientation of the tool to ensure it is downards
        solution = robot.ikine_LM(
            target,
            q0=position_joints,
            mask=[1, 1, 1, 1, 1, 0],
        )

    #halt if still cant find a solution and raise an error
    if not getattr(solution, "success", True):
        raise RuntimeError(
            f"Orientation IK failed for {label} at {np.round(position, 3)}"
        )

    #Extract the solution and once again keep the angles as cloe to the current pos
    joints = np.asarray(solution.q, dtype=float).reshape(-1)
    joints = stabilise_joint_angles(
        robot,
        joints,
        start_joints,
    )

    #check the joint limits and raise an error 
    if not check_joint_limits(robot, joints):
        raise RuntimeError(f"{label} violates joint limits.")

    return joints


#keep the bottles position attached to the robots end effector while it mopves around the scene
def bottle_follow_robot(robot, bottle, bottle_offset=bottle_grab_height):
    #use fk to use the joint angles to get the pose of the end effector 
    tool_pose = robot.fkine(robot.q)

    #uses the bottles offset and a rotation around x to ensure the bottle is downards and at a suitable ditance from the modelled end of the robot. 
    bottle_pose = (
        tool_pose
        * SE3.Trans(0, 0, bottle_offset)
        * SE3.Rx(pi)
    )

    #constantly update the bottle so it follows the new calculated pose
    bottle.T = bottle_pose.A


#move robot functon that allows the robot to move from the current to target joint angles.
def move_robot(
    env,
    robot,
    goal_joints,
    bottle=None,
    holding=False,
    bottle_offset=bottle_grab_height,
):
    #grab the current joint angles and reshape them to be 1d arrays
    start_joints = np.asarray(robot.q, dtype=float).reshape(-1)
    goal_joints = np.asarray(goal_joints, dtype=float).reshape(-1)

    #adjustr the target angles so that the robot chooses the closest angle to move to
    goal_joints = stabilise_joint_angles(
        robot,
        goal_joints,
        start_joints,
    )

    #generates a joint trajectory between the intial and target joint angles with a number of defined steps the trajectory must follow.
    path = jtraj(start_joints, goal_joints, arm_steps).q

    #go through each step of the trajectory and update the robot joint angles to follow the path
    for joints in path:
        robot.q = joints

        #if the robot is holding the bottle then the bottle will follow the robot as it moves
        if holding:
            bottle_follow_robot(
                robot,
                bottle,
                bottle_offset=bottle_offset,
            )

        #updates the sim
        env.step(frame_time)


# where all the code runs
def main():

    #ENVIRONMENT SETUP CODE

    #launch the swift environment 
    env = swift.Swift()
    env.launch(realtime=True)

    #build the environent and also add the table props into the scene
    build_scene(env)
    coaster, bottle = create_final_table_props(env)

    #load the ur3e model and place it on the tavle, also set all the joint angles to the home position
    ur3e = UR3e()
    ur3e.base = SE3.Trans(
        ur3e_base_position[0],
        ur3e_base_position[1],
        ur3e_base_position[2],
    ).A
    ur3e.q = ur3e_home.copy()

    #add the model into the scene
    if hasattr(ur3e, "add_to_env"):
        ur3e.add_to_env(env)
    else:
        env.add(ur3e)

    #set the inital camera position and orientation
    env.set_camera_pose(
        [0, -1.8, 1.8],
        [0, 0, 0],
    )

    env.step(0.1)
    time.sleep(1)

    input("\nPress Enter to start the demonstration...\n")

    #RUN CODE

    # ur3e pickup position is the handover spot with the offset for the bottle
    ur3e_pickup_position = (
        handover_spot
        + [0, 0, ur3e_bottle_offset]
    )

    #define the approach and lift positions for the ur3e to pick up the bottle
    ur3e_approach_position = ur3e_pickup_position + [0, 0, approach_height]
    ur3e_lift_position = ur3e_pickup_position + [0, 0, lift_height]

    #solve the ik for the ur3e to move to the approach position then on to the pickup position and finally lift the bottle up
    ur3e_approach_joints = solve_downward_tool_ik(
        ur3e,
        ur3e_approach_position,
        ur3e.q,
        "ur3e handover approach",
    )

    ur3e_pickup_joints = solve_downward_tool_ik(
        ur3e,
        ur3e_pickup_position,
        ur3e_approach_joints,
        "ur3e bottle pickup",
    )

    ur3e_lift_joints = solve_downward_tool_ik(
        ur3e,
        ur3e_lift_position,
        ur3e_pickup_joints,
        "ur3e lift",
    )

    #move the ur3e from aproach down to the pickup pos 
    move_robot(env, ur3e, ur3e_approach_joints)
    move_robot(env, ur3e, ur3e_pickup_joints)

    #attach the bottle
    bottle_follow_robot(
        ur3e,
        bottle,
        bottle_offset=ur3e_bottle_offset,
    )

    #lift up the bottle from the handover pad
    move_robot(
        env,
        ur3e,
        ur3e_lift_joints,
        bottle,
        True,
        bottle_offset=ur3e_bottle_offset,
    )

    # I had to add a step here in the middle of the handover to deposit positions to ensre the movement is repeatable as often the solver would find a solution that colldied with something
    move_robot(
        env,
        ur3e,
        ur3e_transfer_pose,
        bottle,
        True,
        bottle_offset=ur3e_bottle_offset,
    )

    # ur3e calculation of final position and aproach pos
    final_tool_position = final_spot + [0, 0, ur3e_bottle_offset]
    final_approach_position = final_tool_position + [0, 0, approach_height + 0.05]

    #ik solvers to find the aproach the placement and the final joint angles for the ur3e 
    final_approach_joints = solve_downward_tool_ik(
        ur3e,
        final_approach_position,
        ur3e.q,
        "ur3e final approach",
    )

    final_place_joints = solve_downward_tool_ik(
        ur3e,
        final_tool_position,
        final_approach_joints,
        "ur3e final placement",
    )

    #tracjectory to move the ur3e to the final approach and then on to the final placement of the bottle
    move_robot(
        env,
        ur3e,
        final_approach_joints,
        bottle,
        True,
        bottle_offset=ur3e_bottle_offset,
    )

    move_robot(
        env,
        ur3e,
        final_place_joints,
        bottle,
        True,
        bottle_offset=ur3e_bottle_offset,
    )

    #detach the bottle 
    bottle.T = SE3.Trans(
        final_spot[0],
        final_spot[1],
        final_spot[2],
    ).A
    env.step(frame_time)

    #move the ur3e back to the final approach position to finalise the movement 
    move_robot(env, ur3e, final_approach_joints)

    input("\nPress Enter to close Swift...\n")
    env.close()


if __name__ == "__main__":
    main()
