from roboticstoolbox import Robot
from python.runfiles import Runfiles


def main():
    r = Runfiles.Create()
    urdf_path = r.Rlocation("playground_robotic/src/urdf/planar_rr_arm.urdf")

    print(f"{urdf_path = }")

    robot = Robot.URDF(urdf_path)

    print(f"Loaded robot: {robot.name}")
    print("Number of joints:", robot.n)

    q = [0.5, 0.3]
    T = robot.fkine(q)
    print("End-effector pose:\n", T)


if __name__ == "__main__":
    main()
