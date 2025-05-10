import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Grasp Scene")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import (
    GroundPlaneCfg,
    UsdFileCfg,
)
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import (
    subtract_frame_transforms,
    convert_quat,
    quat_conjugate,
    quat_mul,
)
from isaaclab_assets import FRANKA_PANDA_CFG
from multiprocessing import Process, Manager
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sensors import TiledCameraCfg, TiledCamera
import numpy as np
from PIL import Image


class ROS2_listener(Node):
    def __init__(self):
        super().__init__("isaaclab_listener")
        self.initial_sub = self.create_subscription(
            TransformStamped, "/unity/initial_hand_pose", self.initial_callback, 10
        )
        self.current_sub = self.create_subscription(
            TransformStamped, "/unity/current_hand_pose", self.current_callback, 10
        )
        self.initial_hand_pos = None
        self.initial_hand_rot = None
        self.current_hand_pos = None
        self.current_hand_rot = None

    def initial_callback(self, msg):
        self.initial_hand_pos = msg.transform.translation
        self.initial_hand_rot = msg.transform.rotation

    def current_callback(self, msg):
        self.current_hand_pos = msg.transform.translation
        self.current_hand_rot = msg.transform.rotation


class HandEyeMapper:
    def __init__(self, initial_hand_rot, robot_initial_rot):

        self.h0 = torch.tensor(
            [
                initial_hand_rot.w,
                initial_hand_rot.x,
                initial_hand_rot.y,
                initial_hand_rot.z,
            ],
            dtype=torch.float32,
            device="cuda:0",
        )
        self.r0 = torch.tensor(
            robot_initial_rot, dtype=torch.float32, device="cuda:0"
        ).view(-1)

        self.h0_inv = quat_conjugate(self.h0)

    def map(self, current_hand_rot):
        h_cur = torch.tensor(
            [
                current_hand_rot.w,
                current_hand_rot.x,
                current_hand_rot.y,
                current_hand_rot.z,
            ],
            dtype=torch.float32,
            device="cuda:0",
        )
        h_rel = quat_mul(h_cur, self.h0_inv)
        r_target = quat_mul(h_rel, self.r0)
        return r_target / torch.norm(r_target)


def design_scene():
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))

    cube1_cfg = RigidObjectCfg(
        prim_path="/World/cube1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-10, -0.11587828, 0.02599183]),
        spawn=UsdFileCfg(
            usd_path="/home/roboticlab/Desktop/IsaacLab/urdf_usd/Cube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            scale=(0.05, 0.05, 0.05),
        ),
    )
    cube2_cfg = RigidObjectCfg(
        prim_path="/World/cube2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-10, 0.11261413, 0.02684178]),
        spawn=UsdFileCfg(
            usd_path="/home/roboticlab/Desktop/IsaacLab/urdf_usd/Cube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            scale=(0.05, 0.05, 0.05),
        ),
    )
    cube3_cfg = RigidObjectCfg(
        prim_path="/World/cube3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-10, 0, 0]),
        spawn=UsdFileCfg(
            usd_path="/home/roboticlab/Desktop/IsaacLab/urdf_usd/Cube.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            scale=(0.05, 0.05, 0.15),
        ),
    )
    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
    plane_cfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.7]),
        spawn=GroundPlaneCfg(),
    )
    cam_cfg = TiledCameraCfg(
        prim_path="/World/robot/panda_hand/panda_hand_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        update_period=1.0 / 20.0,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=224,
        height=224,
    )
    cube1_cfg.spawn.func(
        cube1_cfg.prim_path,
        cube1_cfg.spawn,
        translation=cube1_cfg.init_state.pos,
    )
    cube2_cfg.spawn.func(
        cube2_cfg.prim_path,
        cube2_cfg.spawn,
        translation=cube2_cfg.init_state.pos,
    )
    cube3_cfg.spawn.func(
        cube3_cfg.prim_path,
        cube3_cfg.spawn,
        translation=cube3_cfg.init_state.pos,
    )
    robot = Articulation(cfg=robot_cfg)

    plane_cfg.spawn.func(
        plane_cfg.prim_path,
        plane_cfg.spawn,
        translation=plane_cfg.init_state.pos,
    )
    light_cfg.func("/World/Light", light_cfg)

    table_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
    )
    table_cfg.func("/World/Origin4/Table", table_cfg, translation=(0.0, 0.0, 0.0))

    cube1 = RigidObject(cube1_cfg)
    cube2 = RigidObject(cube2_cfg)
    cube3 = RigidObject(cube3_cfg)
    wrist_camera = TiledCamera(cam_cfg)
    scene_entities = {
        "robot": robot,
        "cube1": cube1,
        "cube2": cube2,
        "cube3": cube3,
        "wrist_camera": wrist_camera,
    }
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene, listener, camera_name):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    cube1 = scene["cube1"]
    cube2 = scene["cube2"]
    cube3 = scene["cube3"]
    wrist_camera = scene["wrist_camera"]
    prev_pos = None
    robot: Articulation = scene["robot"]

    # initial joint pos
    default_joint_pos = robot.data.default_joint_pos
    default_joint_vel = torch.zeros_like(default_joint_pos, device="cuda:0")
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    os.makedirs("wrist_cam_images", exist_ok=True)
    ik_controller = DifferentialIKController(ik_controller_cfg, 1, device="cuda:0")
    robot_entity_cfg = SceneEntityCfg(
        "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
    )
    robot_entity_cfg.resolve(scene)
    robot_initial_pos = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:3]
    robot_initial_rot = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 3:7]

    sim_count = 0
    # Simulation loop
    while simulation_app.is_running():
        rclpy.spin_once(listener)
        if listener.initial_hand_pos != None and listener.current_hand_pos != None:
            ik_controller.reset()
            hand_eye_map = HandEyeMapper(
                listener.initial_hand_rot,
                robot_initial_rot,
            )
            gripper_pos = torch.tensor(
                [
                    robot_initial_pos[0][0]
                    + listener.current_hand_pos.x
                    - listener.initial_hand_pos.x,
                    robot_initial_pos[0][1]
                    + listener.current_hand_pos.y
                    - listener.initial_hand_pos.y,
                    robot_initial_pos[0][2]
                    + listener.current_hand_pos.z
                    - listener.initial_hand_pos.z,
                ],
                device="cuda:0",
            )
            gripper_rot = torch.tensor(
                hand_eye_map.map(listener.current_hand_rot), device="cuda:0"
            )
            print("initial_rot:", robot_initial_rot)
            print("current_rot:", gripper_rot)
            # ik_controller.reset()
            # gripper_pos = torch.tensor([0.36, 0.2, 0.1], device="cuda:0")
            # gripper_rot = torch.tensor(
            #     robot.data.body_quat_w[0, robot_entity_cfg.body_ids[0], :],
            #     dtype=torch.float32,
            #     device="cuda:0",
            # )
            gripper_command = torch.cat((gripper_pos, gripper_rot), dim=-1)
            ik_controller.set_command(gripper_command)
            cur_ee_pos = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pos = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos[:, 0:3],
                root_pos[:, 3:7],
                cur_ee_pos[:, 0:3],
                cur_ee_pos[:, 3:7],
            )
            jacobians_mtx = robot.root_physx_view.get_jacobians()[
                :, robot_entity_cfg.body_ids[0] - 1, :, robot_entity_cfg.joint_ids
            ]
            target_joint_pos = ik_controller.compute(
                ee_pos_b.cuda(),
                ee_quat_b.cuda(),
                jacobians_mtx.cuda(),
                joint_pos.cuda(),
            )
            print(target_joint_pos)
            print(robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:3])
            robot.set_joint_position_target(
                target_joint_pos.cuda(), joint_ids=robot_entity_cfg.joint_ids
            )
            robot.write_data_to_sim()
            robot.update(sim_dt)
            img = wrist_camera.data.output["rgb"]
            img_cpu = img[0].cpu()
            if img_cpu.dtype == torch.float32:
                img_cpu = (
                    (img_cpu * 255).clamp(0, 255).to(torch.uint8)
                )  # :contentReference[oaicite:0]{index=0}
            arr = img_cpu.numpy()
            # Image.fromarray(arr).save(f"wrist_cam_images/frame_{sim_count:05d}.png")
            sim_count += 1
            sim.step()


def main():
    """Main function."""
    rclpy.init()
    listener = ROS2_listener()
    sim_cfg = SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene, listener, "panda_hand_wrist_camera")
    listener.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
