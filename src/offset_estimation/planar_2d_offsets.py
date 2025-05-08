import numpy as np
import plotly.graph_objects as go
from python.runfiles import Runfiles
from roboticstoolbox import Robot
from scipy.optimize import minimize


def plot_cost_contours(robot, joint_angles, observed_tip_positions):
    """
    Plots a contour of the cost function over a 2D grid of joint offsets.

    Parameters:
        robot (Robot): The robot model.
        joint_angles (np.ndarray): The commanded joint angles (n_samples, n_joints).
        observed_tip_positions (np.ndarray): The measured tip positions corresponding to each joint angle set.
    """
    offset_range = np.linspace(-np.pi, np.pi, 50)  # Range of offsets to evaluate
    X, Y = np.meshgrid(offset_range, offset_range)
    Z = np.zeros_like(X)

    # Evaluate the cost function over the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            test_offset = np.array([X[i, j], Y[i, j]])
            Z[i, j] = cost(robot, joint_angles, test_offset, observed_tip_positions)

    # Plot the cost contours
    fig = go.Figure(
        data=go.Contour(
            z=Z,
            x=offset_range,
            y=offset_range,
            colorscale="Viridis",
            contours=dict(showlabels=True),
            colorbar=dict(title="Cost"),
        )
    )

    fig.update_layout(
        title="Cost Function Contour over Joint Offsets",
        xaxis_title="Offset Joint 1",
        yaxis_title="Offset Joint 2",
        width=700,
        height=600,
    )

    fig.show()


def cost(
    robot: Robot,
    joint_angles: np.ndarray,
    joint_offsets: np.ndarray,
    observed_tip_positions: np.ndarray,
) -> float:
    """
    Computes the sum of Euclidean distances between observed and theoretical tip positions.

    Parameters:
        robot (Robot): The robot model.
        joint_angles (np.ndarray): Commanded joint angles (n_samples, n_joints).
        joint_offsets (np.ndarray): Estimated mastering offsets (n_joints,).
        observed_tip_positions (np.ndarray): Measured tip positions (n_samples, 2).

    Returns:
        float: Sum of Euclidean errors.
    """
    theoretical_tip_positions = get_tip_positions(robot, joint_angles, joint_offsets)
    diff = observed_tip_positions - theoretical_tip_positions
    return np.linalg.norm(diff, axis=1).sum()


def solve_for_offsets(
    robot: Robot, executed_joint_angles: np.ndarray, observed_tip_positions: np.ndarray
) -> np.ndarray:
    """
    Solves for joint mastering offsets that minimize the difference between observed and theoretical tip positions.

    Parameters:
        robot (Robot): The robot model.
        executed_joint_angles (np.ndarray): Commanded joint angles (n_samples, n_joints).
        observed_tip_positions (np.ndarray): Measured tip positions (n_samples, 2).

    Returns:
        np.ndarray: Estimated mastering offsets (n_joints,).
    """
    x0 = np.zeros(executed_joint_angles.shape[1])  # Initial guess: zero offsets

    solution = minimize(
        lambda x: cost(
            robot=robot,
            joint_angles=executed_joint_angles,
            joint_offsets=x,
            observed_tip_positions=observed_tip_positions,
        ),
        x0=x0,
    )

    # Show optimization result details
    print(solution)

    return solution.x


def get_tip_positions(
    robot: Robot, joint_angles: np.ndarray, joint_offsets: np.ndarray
) -> np.ndarray:
    """
    Computes the 2D tip positions of the robot given joint angles and mastering offsets.

    Parameters:
        robot (Robot): The robot model.
        joint_angles (np.ndarray): Commanded joint angles (n_samples, n_joints).
        joint_offsets (np.ndarray): Mastering offsets to apply (n_joints,).

    Returns:
        np.ndarray: Calculated 2D tip positions (n_samples, 2).
    """
    executed_joint_angles = joint_angles + joint_offsets
    tip_positions = np.zeros((len(executed_joint_angles), 2))

    for indx, q in enumerate(executed_joint_angles):
        # IMPORTANT: Ensure this uses the tip link!
        world_H_tip = robot.fkine(q, end="tip")  # 'tip' must match URDF tip link name
        tip_positions[indx, :] = world_H_tip.t[0:2]

    return tip_positions


def main():
    # Load URDF using Bazel runfiles and build the robot
    r = Runfiles.Create()
    urdf_path = r.Rlocation("playground_robotic/src/urdf/planar_rr_arm.urdf")

    robot = Robot.URDF(urdf_path)

    print(f"Loaded robot: {robot.name}")
    print("Number of joints:", robot.n)

    # Number of random samples to generate (number of calibration acquires)
    n_samples = 100

    # Generate random joint angles in [-pi, pi] (calibration joint angles)
    sample_joint_angles = 2 * np.pi * np.random.rand(n_samples, 2) - np.pi

    # Ground truth mastering offsets
    mastering_offsets = np.array([0.2, -0.1])

    # Simulate observed positions using true offsets
    observed_tip_positions = get_tip_positions(robot, sample_joint_angles, mastering_offsets)

    # Plot cost surface to verify observability and gradient shape
    plot_cost_contours(robot, sample_joint_angles, observed_tip_positions)

    # Solve for the unknown offsets using optimization
    calculated_offsets = solve_for_offsets(robot, sample_joint_angles, observed_tip_positions)

    # Print results
    print(f"\nAssumed mastering offsets: {mastering_offsets}")
    print(f"Calculated offsets:        {calculated_offsets}")
    print(f"Difference norm:           {np.linalg.norm(mastering_offsets - calculated_offsets)}")


if __name__ == "__main__":
    main()
