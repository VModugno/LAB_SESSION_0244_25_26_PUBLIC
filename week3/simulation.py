import numpy as np
import dartpy as dart
import copy
from utils import *
import os
import inverse_dynamics as id
from logger import Logger

divergence_detected = False

def set_body_friction(body_node, friction_coeff):
    """
    Iterates through all ShapeNodes of a BodyNode and sets the friction
    coefficient on those that have a CollisionAspect.
    """
    if body_node is None:
        print("Warning: BodyNode is None, cannot set friction.")
        return

    # Iterate over all shapes attached to this body
    for i in range(body_node.getNumShapeNodes()):
        shape_node = body_node.getShapeNode(i)
        
        # We only care about shapes involved in collision
        if shape_node.hasCollisionAspect():
            # Friction is stored in the DynamicsAspect
            dyn_aspect = shape_node.getDynamicsAspect()
            
            # If the aspect doesn't exist (rare), create it
            if dyn_aspect is None:
                dyn_aspect = shape_node.createDynamicsAspect()
                
            dyn_aspect.setFrictionCoeff(friction_coeff)




class Hrp4Controller(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, hrp4):
        super(Hrp4Controller, self).__init__(world)
        self.world = world
        self.hrp4 = hrp4
        self.time = 0
        self.params = {
            'g': 9.81,
            'h': 0.75,
            'foot_size': 0.1,
            'world_time_step': world.getTimeStep(),
            'µ': 0.5,
            'dof': self.hrp4.getNumDofs(),
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])

        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')

        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()

            # set floating base to passive, everything else to torque
            if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
            elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

        # set initial configuration
        self.hrp4.setPosition(0, -50. * np.pi / 180.)
        initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
                                 'R_HIP_Y': 0., 'R_HIP_R': -53., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
                                 'L_HIP_Y': 0., 'L_HIP_R':  53., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -90., 'R_SHOULDER_Y': 0., 'R_ELBOW_P': -25., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  90., 'L_SHOULDER_Y': 0., 'L_ELBOW_P': -25.}

        for joint_name, value in initial_configuration.items():
            self.hrp4.setPosition(self.hrp4.getDof(joint_name).getIndexInSkeleton(), value * np.pi / 180.)

        # position the robot on the ground
        lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        self.hrp4.setPosition(3, - lsole_pos[0])
        self.hrp4.setPosition(4, - lsole_pos[1])
        self.hrp4.setPosition(5, - lsole_pos[2])

        # initialize state
        self.initial = self.retrieve_state()
        self.desired = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [
            "NECK_Y", "NECK_P",
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P",
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P",

            # add the non-contact (swing) leg joints so they are stabilized by posture regularization
            "R_HIP_Y", "R_HIP_R", "R_HIP_P", "R_KNEE_P", "R_ANKLE_P", "R_ANKLE_R",
        ]

        # initialize inverse dynamics (QP-based WBC)
        self.id = id.InverseDynamics(
            self.hrp4,
            redundant_dofs,
            foot_size=self.params['foot_size'],
            µ=self.params['µ'],
        )

        # initialize logger and plots
        self.logger = Logger(self.initial)
        self.logger.initialize_plot(frequency=10)

        # Which foot is the stance contact in single-support.
        # NOTE: the "balancing" init shifts the base to place the *left* sole on the ground.
        self.stance_foot = 'lfoot'
        self.swing_foot = 'rfoot'

    def customPreStep(self):
        global divergence_detected
        # create current and desired states
        self.current = self.retrieve_state()

        # Desired tasks (single-support balancing):
        # - keep COM at nominal height
        # - keep stance foot fixed (world)
        # - do NOT track swing foot pose (let it be free)
        # - keep torso/base orientations close to initial
        # - redundant joint posture oscillation
        self.desired['com']['pos'] = np.array([
            self.initial['com']['pos'][0],
            self.initial['com']['pos'][1],
            self.params['h'],
        ])
        self.desired['com']['vel'] = np.zeros(3)
        self.desired['com']['acc'] = np.zeros(3)

        # keep stance foot fixed (world)
        stance = self.stance_foot
        self.desired[stance]['pos'] = self.initial[stance]['pos'].copy()
        self.desired[stance]['vel'] = np.zeros(6)
        self.desired[stance]['acc'] = np.zeros(6)

        # let swing foot be unconstrained by task tracking (but keep dict entries valid)
        swing = self.swing_foot
        self.desired[swing]['pos'] = self.current[swing]['pos'].copy()
        self.desired[swing]['vel'] = self.current[swing]['vel'].copy()
        self.desired[swing]['acc'] = np.zeros(6)

        # keep torso/base orientations close to initial
        for link in ['torso', 'base']:
            self.desired[link]['pos'] = self.initial[link]['pos'].copy()
            self.desired[link]['vel'] = np.zeros(3)
            self.desired[link]['acc'] = np.zeros(3)

        # joint posture task + oscillation (only affects redundant dofs via selection matrix in ID)
        self.desired['joint']['pos'] = self.initial['joint']['pos'].copy()
        self.desired['joint']['vel'] = np.zeros_like(self.desired['joint']['pos'])
        self.desired['joint']['acc'] = np.zeros_like(self.desired['joint']['pos'])

        oscillation = 1.0 * np.sin(0.003 * self.time)
        idx_L_SHOULDER_R = self.hrp4.getDof('L_SHOULDER_R').getIndexInSkeleton()
        idx_R_SHOULDER_R = self.hrp4.getDof('R_SHOULDER_R').getIndexInSkeleton()
        idx_R_HIP_P = self.hrp4.getDof('R_HIP_P').getIndexInSkeleton()
        idx_R_KNEE_P = self.hrp4.getDof('R_KNEE_P').getIndexInSkeleton()

        self.desired['joint']['pos'][idx_L_SHOULDER_R] += oscillation
        self.desired['joint']['pos'][idx_R_SHOULDER_R] -= oscillation
        self.desired['joint']['pos'][idx_R_HIP_P]      -= oscillation
        self.desired['joint']['pos'][idx_R_KNEE_P]     += oscillation

        # single-support contact in the QP constraints
        contact = self.stance_foot

        # get torque commands using inverse dynamics
        if not divergence_detected:
            commands = self.id.get_joint_torques(self.desired, self.current, contact)
        else:
            commands = np.zeros(self.params['dof'] - 6)

        # --- SAFETY INTERVENTION: CHECK CONTROL OUTPUT ---
        # If the QP solver exploded, 'commands' will contain NaN or Inf.
        # If the physics is becoming unstable, torques might spike to 10,000+ Nm.
        if not divergence_detected:
            is_nan_inf = np.isnan(commands).any() or np.isinf(commands).any()
            is_explosion = np.max(np.abs(commands)) > 200.0 # Threshold: 500 Nm (way above HRP4 limits)
        else:
            is_nan_inf = True
            is_explosion = True

        if divergence_detected or is_nan_inf or is_explosion:
            print("!!! CONTROLLER DIVERGENCE DETECTED !!!")
            print(f"Status: NaN/Inf={is_nan_inf}, MaxTorque={np.max(np.abs(commands))}")
            print("Shutting down controller to prevent simulation crash.")
            divergence_detected = True
            # OVERRIDE: Send zero torques (Passive/Ragdoll mode)
            commands = np.zeros_like(commands)
        # --------------------------------------------------

        # set actuator commands
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])

        # log and plot
        self.logger.log_data(self.current, self.desired)
        # self.logger.update_plot(self.time)

        self.time += 1

    def retrieve_state(self):
        # com and torso pose (orientation and position)
        com_position = self.hrp4.getCOM()
        torso_orientation = get_rotvec(self.hrp4.getBodyNode('torso').getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())
        base_orientation  = get_rotvec(self.hrp4.getBodyNode('body' ).getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())

        # feet poses (orientation and position)
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))
        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_orientation, r_foot_position))

        # velocities
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_angular_velocity = self.hrp4.getBodyNode('torso').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_angular_velocity = self.hrp4.getBodyNode('body').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())

        # compute total contact force
        force = np.zeros(3)
        for contact in world.getLastCollisionResult().getContacts():
            force += contact.force

        # compute zmp
        zmp = np.zeros(3)
        zmp[2] = com_position[2] - force[2] / (self.hrp4.getMass() * self.params['g'] / self.params['h'])
        for contact in world.getLastCollisionResult().getContacts():
            if contact.force[2] <= 0.1: continue
            zmp[0] += (contact.point[0] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[0] / force[2])
            zmp[1] += (contact.point[1] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[1] / force[2])

        if force[2] <= 0.1: # threshold for when we lose contact
            zmp = np.array([0., 0., 0.]) # FIXME: this should return previous measurement
        else:
            # sometimes we get contact points that dont make sense, so we clip the ZMP close to the robot
            midpoint = (l_foot_position + l_foot_position) / 2.
            zmp[0] = np.clip(zmp[0], midpoint[0] - 0.3, midpoint[0] + 0.3)
            zmp[1] = np.clip(zmp[1], midpoint[1] - 0.3, midpoint[1] + 0.3)
            zmp[2] = np.clip(zmp[2], midpoint[2] - 0.3, midpoint[2] + 0.3)

        # create state dict
        return {
            'lfoot': {'pos': left_foot_pose,
                      'vel': l_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'rfoot': {'pos': right_foot_pose,
                      'vel': r_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'com'  : {'pos': com_position,
                      'vel': com_velocity,
                      'acc': np.zeros(3)},
            'torso': {'pos': torso_orientation,
                      'vel': torso_angular_velocity,
                      'acc': np.zeros(3)},
            'base' : {'pos': base_orientation,
                      'vel': base_angular_velocity,
                      'acc': np.zeros(3)},
            'joint': {'pos': self.hrp4.getPositions(),
                      'vel': self.hrp4.getVelocities(),
                      'acc': np.zeros(self.params['dof'])},
            'zmp'  : {'pos': zmp,
                      'vel': np.zeros(3),
                      'acc': np.zeros(3)}
        }

if __name__ == "__main__":
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))

    # --- PHYSICS FRICTION ---
    # Set friction of the ground (e.g., 0.1 for ice, 1.0 for rubber)
    ground_mu = 1.0 
    set_body_friction(ground.getBodyNode("ground_link"), ground_mu)
    #ground.getBodyNode("ground_link").setFrictionCoeff(ground_mu)

    # Ideally, set the feet as well, though DART usually takes the min() of the two.
    # We set them high so the ground dominates the interaction.
    foot_mu = 1.0 
    #hrp4.getBodyNode("l_sole").setFrictionCoeff(foot_mu)
    #hrp4.getBodyNode("r_sole").setFrictionCoeff(foot_mu)
    set_body_friction(hrp4.getBodyNode("l_sole"), foot_mu)
    set_body_friction(hrp4.getBodyNode("r_sole"), foot_mu)
    # ----------------------------------------------------


    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    world.setGravity([0, 0, -9.81])
    world.setTimeStep(0.01)

    # set default inertia
    default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
    for body in hrp4.getBodyNodes():
        if body.getMass() == 0.0:
            body.setMass(1e-8)
            body.setInertia(default_inertia)

    node = Hrp4Controller(world, hrp4)

    # create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    node.setTargetRealTimeFactor(10) # speed up the visualization by 10x
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    #viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    viewer.run()
