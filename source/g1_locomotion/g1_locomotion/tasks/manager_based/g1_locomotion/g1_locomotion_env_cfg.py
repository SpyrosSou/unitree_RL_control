# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29dof unified stand+walk velocity-tracking task.

``G1LocomotionEnvCfg``/``G1LocomotionEnvCfg_PLAY`` below are ported near-verbatim from
``unitree_rl_lab`` (``source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/
29dof/velocity_env_cfg.py``), 2026-07-21 — see ``29dof_implementation_plan.md`` for the
full rationale. Only change from the upstream version: the robot asset import points at
this repo's own ``g1_locomotion.assets.robots.unitree.UNITREE_G1_29DOF_CFG`` (a local
copy, not a runtime dependency on the ``unitree_rl_lab`` package — see the plan doc's
"what gets ported" section for why).

``G1LocomotionArmDisturbanceEnvCfg``/``_PLAY`` add the arm-motion disturbance curriculum
(``mdp.ArmMotionDisturbance``) on top of the base recipe — kept as its own gym task id
rather than folded into ``G1LocomotionEnvCfg`` directly, so the pure-ported-recipe
baseline stays available for comparison (per the plan's Phase 1.4 sanity-check goal).

Replaces the 23dof-era ``g1_locomotion_env_cfg.py`` that used to live at this exact path
(many ``G1LocomotionStandingFlat*``/``G1LocomotionFlat*`` variants, 1-DOF waist, 5-DOF
arms) — preserved on the ``23_dof`` git branch, not carried forward here; this file is a
full replacement, not an addition alongside it.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from g1_locomotion.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

from . import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # Actuator domain randomization (2026-07-21, added on top of Unitree's ported recipe
    # — see 29dof_implementation_plan.md's sim2real notes: their own leg/waist recipe has
    # no actuator gain/friction/armature randomization at all, only the arm task in this
    # repo's 23dof-era g1_arm_env.py did. Applied here repo-wide (all joints, not just
    # arms) — same distribution ranges as that precedent (gain scale 0.8-1.2x, friction
    # 0-0.03, armature scale 0.8-1.2x), since this task now needs to be robust to gain
    # uncertainty across the whole body, not just the arms.
    #
    # Briefly removed 2026-07-23 as a hypothesis for why trained checkpoints showed
    # near-zero leg motion at some test speeds — restored after further investigation
    # (checking leg motion across a range of commanded speeds, not just one) showed
    # every checkpoint, randomization on or off, actually produces real leg motion and
    # displacement at speeds within what lin_vel_cmd_levels had actually reached during
    # its training. The real cause was that curriculum being stuck at its starting tier
    # (fixed the same day — see mdp/curriculums.py's own note), not this randomization.
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.0, 0.03),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "abs",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # REVERTED 2026-07-23 back to Unitree's original std=0.5 — briefly tightened to 0.3
    # earlier today as a guess, but a direct comparison against this repo's own 23dof
    # branch (isaaclab_tasks' stock G1FlatEnvCfg, confirmed to produce real walking in
    # ~1500 iterations) showed that recipe ALSO uses std=0.5 — so this wasn't the actual
    # gap and just added an untested confound. See the termination_penalty/curriculum/
    # action_rate changes in the Ablation*EnvCfg classes below for the properly evidenced
    # candidates, each tested in isolation against this same, original Unitree baseline.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # CHANGED 2026-07-24: -0.05 (Unitree original) -> -0.005 (this repo's own 23dof-branch
    # stock G1 recipe's value). Confirmed via a 4-way, 2000-iteration ablation (see
    # Ablation*EnvCfg classes below) that this was the one change, of three candidates
    # tested in isolation, that actually broke the "never falls, never walks" local
    # optimum — the only config that produced real forward velocity in eval instead of
    # standing still (forward_medium track err dropped from ~0.6, i.e. ~0 achieved
    # velocity, to 0.173). Not yet clean/controlled walking (real heading/lateral drift
    # in that eval), but the first checkpoint in this project's 29dof history to move at
    # all under a nonzero command. The other two candidates (termination_penalty,
    # curriculum loosening — see Ablation*EnvCfg) were no-ops at this same budget.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    # CHANGED 2026-07-24: -1.0 (Unitree original) -> -0.1 (23dof-branch stock G1 recipe's
    # value) — see action_rate's own comment above for the ablation evidence; these two
    # changes were tested together as one candidate and are the pair confirmed to matter.
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.78})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class G1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Self-collision — already the asset's own default (see
        # g1_locomotion.assets.robots.unitree.UnitreeUsdFileCfg), restated explicitly
        # here for parity with the arm task's own explicit line (g1_arm_env.py) and so
        # this doesn't silently depend on an un-stated asset default (2026-07-21, user
        # asked to confirm self-collision is actually implemented — it was, this just
        # makes it defensive/explicit here too).
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1LocomotionEnvCfg_PLAY(G1LocomotionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# ---------------------------------------------------------------------------
# Walking-fix ablation configs (2026-07-23)
#
# Byte-identical-to-Unitree checkpoints (base and arm_disturbance recipes) were found to
# converge to near-zero real forward velocity — falls rarely, but essentially never
# walks either, confirmed via direct root_pos_w/joint_vel measurement, not just reward
# metrics. Diffing against this repo's own 23dof branch (which used isaaclab_tasks'
# stock G1FlatEnvCfg, NOT the Unitree-ported recipe, and reportedly converges to real
# walking in ~1500 iterations) surfaced several concrete structural differences. Each is
# tested here IN ISOLATION against the same, unmodified Unitree baseline (G1LocomotionEnvCfg
# itself, std=0.5/action_rate=-0.05 as ported), so a short (~1k iteration) run of each can
# be compared directly to attribute any improvement to a specific change rather than a
# bundle. `G1LocomotionAblationAllEnvCfg` combines all three for comparison against the
# individual results. push_robot is deliberately NOT touched by any of these — it's
# plausibly load-bearing for this project's own standing-under-disturbance robustness
# (unlike the 23dof stock recipe, which has no standing requirement at all), so disabling
# it isn't a safe thing to blindly port even though the stock recipe does.
class G1LocomotionAblationTermPenaltyEnvCfg(G1LocomotionEnvCfg):
    """Ablation 1/3: add an explicit, large penalty for the termination event itself
    (falling), on top of the unmodified Unitree baseline. The Unitree-ported recipe has
    NO such term — falling only costs the `alive`/tracking reward it would have earned
    for the rest of the episode, the same soft cost as e.g. a mistimed step. The 23dof
    branch's stock G1FlatEnvCfg has `termination_penalty = RewTerm(func=is_terminated,
    weight=-200.0)` — a large, one-time, unambiguous cost specifically for falling,
    regardless of whether the fall happened while standing-commanded or walking-commanded."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


class G1LocomotionAblationCurriculumEnvCfg(G1LocomotionEnvCfg):
    """Ablation 2/3: disable the slow-growing lin_vel_cmd_levels curriculum and train on
    a wide command range from iteration 0, matching the 23dof stock recipe's approach
    (it has no equivalent curriculum at all — trains on ~its full target range
    immediately). Our own curriculum starts at (-0.1, 0.1) and only grows when reward
    exceeds a threshold; even after fixing its normalization bug earlier this session it
    still spends a lot of early training in a near-stationary regime where standing
    still is nearly free reward-wise."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)


class G1LocomotionAblationRewardWeightsEnvCfg(G1LocomotionEnvCfg):
    """Ablation 3/3: lighten action_rate and joint_deviation_legs to match the 23dof
    stock recipe's actual values (-0.005 and -0.1 respectively, vs. our -0.05 and -1.0).
    action_rate was the single largest-magnitude reward contribution in every walking
    run examined; joint_deviation_legs penalizes hip roll/yaw deviation an order of
    magnitude harder than the recipe that's known to produce real walking, which could
    be suppressing the hip motion real bipedal gait needs."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate.weight = -0.005
        self.rewards.joint_deviation_legs.weight = -0.1


class G1LocomotionAblationAllEnvCfg(G1LocomotionEnvCfg):
    """All three ablations combined, for comparison against each isolated above."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
        self.curriculum.lin_vel_cmd_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.rewards.action_rate.weight = -0.005
        self.rewards.joint_deviation_legs.weight = -0.1


@configclass
class G1LocomotionArmDisturbanceEnvCfg(G1LocomotionEnvCfg):
    """``G1LocomotionEnvCfg`` (Unitree's ported base recipe) plus the arm-motion disturbance
    curriculum (``mdp.ArmMotionDisturbance``) — see ``29dof_implementation_plan.md``'s
    Phase-0.3 resolution: this is not optional/late-stage, a unified stand+walk policy
    trained with zero arm disturbance is expected to fail the same way the 23dof phase's
    early standing checkpoints did the first time a real reaching arm was introduced.

    Kept as its own gym task id (not folded into ``G1LocomotionEnvCfg`` directly) so the pure-
    ported-recipe baseline stays available for the Phase 1.4 sanity-check comparison —
    same layered-config pattern the 23dof-era ``g1_locomotion_env_cfg.py`` used throughout.
    """

    def __post_init__(self):
        super().__post_init__()

        if hasattr(self.actions, "JointPositionAction"):
            self.actions.JointPositionAction.class_type = mdp.ArmDisturbanceBlendJointPositionAction

        self.events.arm_motion_disturbance = EventTerm(
            func=mdp.ArmMotionDisturbance,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={"asset_cfg": SceneEntityCfg("robot")},
            is_global_time=False,
        )
        self.events.arm_motion_reset = EventTerm(
            func=mdp.reset_arm_motion_targets_to_default,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class G1LocomotionArmDisturbanceEnvCfg_PLAY(G1LocomotionArmDisturbanceEnvCfg):
    """Play/eval variant — same PLAY overrides as ``G1LocomotionEnvCfg_PLAY``, plus starting the
    arm-disturbance phase further along so it's visible quickly (same pattern the 23dof-era
    ``G1LocomotionStandingFlatEnvCfg_PLAY`` used)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        # Force every env to be "standing" for this play session — mdp.ArmMotionDisturbance
        # only disturbs standing-commanded envs (base recipe's rel_standing_envs=0.02 is
        # untouched by the override above), so with a handful of --num_envs the odds any
        # of them land in that 2% slice are low; without this, arms silently just relax
        # to default the whole session instead of visibly disturbing (found 2026-07-22
        # when a --num_envs 4 play session showed no arm motion at all).
        self.commands.base_velocity.rel_standing_envs = 1.0

        if hasattr(self.events, "arm_motion_disturbance"):
            # Re-tuned 2026-07-22 alongside mdp.ArmMotionDisturbance's own re-tune (3
            # non-zero tiers now, not 5 — the old (20, 80, 180, 320, 500) 5-boundary
            # tuple would index out of range against the new, shorter per-phase arrays).
            # offset=0 (not the old scheme's 140) so a play.py session starts at true
            # phase 0 and shows the full 0->1->2->3 progression in order, ~2s/4s/6s dwell
            # per phase before settling at max — for visually verifying the curriculum,
            # not for skipping ahead to "interesting" behavior immediately.
            self.events.arm_motion_disturbance.params["phase_step_boundaries"] = (100, 300, 600)
            self.events.arm_motion_disturbance.params["phase_step_offset"] = 0


@configclass
class G1LocomotionArmDisturbanceStandingFocusEnvCfg(G1LocomotionArmDisturbanceEnvCfg):
    """``G1LocomotionArmDisturbanceEnvCfg`` with a much higher fraction of envs commanded
    to stand (``rel_standing_envs`` 0.02 -> 0.5), for a dedicated fine-tuning phase
    stacked on top of a normal run via ``--resume`` (2026-07-22, user request).

    Motivation: ``mdp.ArmMotionDisturbance`` is gated to only disturb envs currently
    commanded to (near-)stand (see that class's docstring) — a deliberate, correct
    restriction ("arms while walking" is a future feature, not trained toward yet) —
    but it means the disturbance curriculum only ever gets to act on however many
    env-steps happen to be near-zero-commanded, which at the base recipe's own
    ``rel_standing_envs=0.02`` is a small slice of total training. This config doesn't
    change that gate — it changes how much of training happens inside the regime the
    gate lets through, by making standing itself far more common.

    Deliberately NOT 100% standing: the general (non-standing) command range is left
    completely untouched, so whatever fraction of envs aren't in the dedicated standing
    slice still sample the same full walking distribution as before — the intent is
    "spend more time in the regime that matters most for arm-disturbance robustness,
    without starving general walking competency of practice entirely," not a full
    switch to a standing-only task. 0.5 is a first-pass, reasoned starting point (half
    standing, half full walking distribution), not tuned — if a resumed run shows clear
    walking-quality regression (tracking error/foot slip getting worse, not just
    noisier) vs. the pre-resume checkpoint, that's a sign this fraction (or the
    iteration count spent here) was too aggressive; lower it or shorten this phase
    rather than assuming the regression is unrelated.

    Usage — resume an existing arm-disturbance run under this task for additional
    iterations (see overnight_train.sh's Step 1b for the concrete overnight version):

        python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-ArmDisturbance-StandingFocus-v0 \\
            --headless --resume --checkpoint <path/to/model_5999.pt> --max_iterations 9000
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.5


@configclass
class G1LocomotionArmDisturbanceStandingFocusEnvCfg_PLAY(G1LocomotionArmDisturbanceStandingFocusEnvCfg):
    """Eval variant, same pattern as G1LocomotionArmDisturbanceEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        # Force every env to be "standing" for this play session — mdp.ArmMotionDisturbance
        # only disturbs standing-commanded envs (base recipe's rel_standing_envs=0.02 is
        # untouched by the override above), so with a handful of --num_envs the odds any
        # of them land in that 2% slice are low; without this, arms silently just relax
        # to default the whole session instead of visibly disturbing (found 2026-07-22
        # when a --num_envs 4 play session showed no arm motion at all).
        self.commands.base_velocity.rel_standing_envs = 1.0

        if hasattr(self.events, "arm_motion_disturbance"):
            # Re-tuned 2026-07-22 alongside mdp.ArmMotionDisturbance's own re-tune (3
            # non-zero tiers now, not 5 — the old (20, 80, 180, 320, 500) 5-boundary
            # tuple would index out of range against the new, shorter per-phase arrays).
            # offset=0 (not the old scheme's 140) so a play.py session starts at true
            # phase 0 and shows the full 0->1->2->3 progression in order, ~2s/4s/6s dwell
            # per phase before settling at max — for visually verifying the curriculum,
            # not for skipping ahead to "interesting" behavior immediately.
            self.events.arm_motion_disturbance.params["phase_step_boundaries"] = (100, 300, 600)
            self.events.arm_motion_disturbance.params["phase_step_offset"] = 0


@configclass
class G1LocomotionArmPolicyDisturbanceEnvCfg(G1LocomotionEnvCfg):
    """``G1LocomotionEnvCfg`` (Unitree's ported base recipe) plus the arm-motion
    disturbance curriculum driven by the actual trained 7-DOF arm RL policy
    (``mdp.ArmPolicyDisturbance``) instead of scripted random joint deltas.

    **Not currently used by any registered training run** (2026-07-22, user request) —
    built and ready once the currently-training ``G1-Locomotion-Velocity-ArmDisturbance-
    v0`` (scripted disturbance) run's results are in. See ``ArmPolicyDisturbance``'s own
    docstring in ``mdp/events.py`` for the full design/rationale and the "not yet
    runtime-verified" caveat.

    ``arm_checkpoint`` defaults to the 2026-07-22 left-arm checkpoint (iteration 2999,
    99.95%+ eval success rate) — override via this class's ``__post_init__`` or by
    editing the path below once a better checkpoint exists.
    """

    _DEFAULT_ARM_CHECKPOINT = (
        "logs/rsl_rl/arms/left/2026-07-22_06-20-55/model_2999.pt"
    )

    def __post_init__(self):
        super().__post_init__()

        if hasattr(self.actions, "JointPositionAction"):
            self.actions.JointPositionAction.class_type = mdp.ArmDisturbanceBlendJointPositionAction

        self.events.arm_motion_disturbance = EventTerm(
            func=mdp.ArmPolicyDisturbance,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={"asset_cfg": SceneEntityCfg("robot"), "arm_checkpoint": self._DEFAULT_ARM_CHECKPOINT},
            is_global_time=False,
        )


@configclass
class G1LocomotionArmPolicyDisturbanceEnvCfg_PLAY(G1LocomotionArmPolicyDisturbanceEnvCfg):
    """Play/eval variant — same PLAY overrides as ``G1LocomotionArmDisturbanceEnvCfg_PLAY``."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_velocity.rel_standing_envs = 1.0

        if hasattr(self.events, "arm_motion_disturbance"):
            self.events.arm_motion_disturbance.params["enable_step"] = 0
            self.events.arm_motion_disturbance.params["ramp_full_step"] = 200
            self.events.arm_motion_disturbance.params["start_fraction"] = 1.0
