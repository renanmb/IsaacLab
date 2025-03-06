import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

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
# omniverse://localhost/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/
# omniverse://localhost/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/TrafficCone_A_PR_V_NVD_01.usd
# f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
CONE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Assets/DigitalTwin/Assets/Warehouse/Safety/Cones/Traffic/TrafficCone_A_PR_V_NVD_01.usd",
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