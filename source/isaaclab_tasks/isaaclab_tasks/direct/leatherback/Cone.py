import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

##
# configuration
##

CONE_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Cones",
    markers={
        "marker1": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker2": VisualizationMarkersCfg.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    }
)
"""Configuration for a simple traffic cone marker"""

REW_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Cones",
    markers={
        "marker0": sim_utils.SphereCfg(
            radius=-0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker1": VisualizationMarkersCfg.SphereCfg(
            radius=-0.09,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker2": VisualizationMarkersCfg.SphereCfg(
            radius=-0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker3": VisualizationMarkersCfg.SphereCfg(
            radius=-0.07,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker4": VisualizationMarkersCfg.SphereCfg(
            radius=-0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker5": VisualizationMarkersCfg.SphereCfg(
            radius=-0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker6": VisualizationMarkersCfg.SphereCfg(
            radius=-0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker7": VisualizationMarkersCfg.SphereCfg(
            radius=-0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker8": VisualizationMarkersCfg.SphereCfg(
            radius=-0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker9": VisualizationMarkersCfg.SphereCfg(
            radius=-0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "marker10": VisualizationMarkersCfg.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker11": VisualizationMarkersCfg.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker12": VisualizationMarkersCfg.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker13": VisualizationMarkersCfg.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker14": VisualizationMarkersCfg.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker15": VisualizationMarkersCfg.SphereCfg(
            radius=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker16": VisualizationMarkersCfg.SphereCfg(
            radius=0.07,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker17": VisualizationMarkersCfg.SphereCfg(
            radius=0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker18": VisualizationMarkersCfg.SphereCfg(
            radius=0.09,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        "marker19": VisualizationMarkersCfg.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    }
)