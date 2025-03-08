import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

##
# configuration
##

"""Configuration for a simple cone marker"""

# prim_path= ""
CONE1_CFG = RigidObjectCfg(
    prim_path="/World/Origin.*/Cone",
    spawn=sim_utils.ConeCfg(
        radius=0.1,
        height=0.2,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
)

"""Configuration for a simple traffic cone marker"""
# The Cones TrafficCone_A01_16cm_PR_V_NVD_01.usd and TrafficCone_A_PR_V_NVD_01.usd are the same and are too small.
# The Cone TrafficCone_A02_30cm_PR_V_NVD_01.usd seems ideal for the track
# omniverse://localhost/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/
# omniverse://localhost/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/TrafficCone_A01_16cm_PR_V_NVD_01.usd
# omniverse://localhost/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/TrafficCone_A_PR_V_NVD_01.usd
# omniverse://localhost/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/TrafficCone_A02_30cm_PR_V_NVD_01.usd
# f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
# Downloading you dont get the Materials
# usd_path=f"/home/goat/Documents/GitHub/renanmb/IsaacLab/source/assets/robots/TrafficCone_A02_30cm_PR_V_NVD_01.usd"
# Try using MultiAssetSpawnerCfg vs UsdFileCfg
CONE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{NVIDIA_NUCLEUS_DIR}/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/TrafficCone_A02_30cm_PR_V_NVD_01.usd",
        scale=(0.006,0.006,0.006),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)
# The traffic Cones are Spawning of Scale for some reason
# usd_path=f"/home/goat/Documents/GitHub/renanmb/IsaacLab/source/assets/robots/Cones/TrafficCone_A02_30cm_PR_V_NVD_01.usd"
"""Configuration for a collection of 10 traffic cones with unique paths"""
CONES_CFG = RigidObjectCollectionCfg(
    rigid_objects={
        f"Cone_{i}": RigidObjectCfg(
            prim_path = f"/World/envs/env_.*/Cone_{i}",
            spawn = sim_utils.UsdFileCfg(
                # usd_path = f"{NVIDIA_NUCLEUS_DIR}/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/TrafficCone_A02_30cm_PR_V_NVD_01.usd",
                usd_path=f"/home/goat/Documents/GitHub/renanmb/IsaacLab/source/assets/robots/Cones/TrafficCone_A02_30cm_PR_V_NVD_01.usd",
                scale = (0.008,0.008,0.008), # make sure the assets have the proper scale
                # mass_props=sim_utils.MassPropertiesCfg(mass=1.0), # Adding mass is causing issues
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props = sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=100.0,
                    enable_gyroscopic_forces=True,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(i, 0.0, 1.0)),  # Default position
        )
        for i in range(20)  # Create 10 cones at different positions
    }
)

"""Configuration for a collection of 10 red cones with unique paths"""
CONE_COLLECTION_CFG = RigidObjectCollectionCfg(
    rigid_objects={
        f"Cone_{i}": RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Cone_{i}",
            spawn=sim_utils.ConeCfg(
                radius=0.15,
                height=0.5,
                # rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red cones
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4, solver_velocity_iteration_count=0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(i, 10.0, 10.0)),  # Default position
        )
        for i in range(5)  # Create 10 cones at different positions
    }
)

