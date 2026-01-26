import dartpy as dart
import numpy as np
from utils import *

class InverseDynamics:
    def __init__(self, robot, redundant_dofs, foot_size=0.1, µ=0.5):
        self.robot = robot
        self.dofs = self.robot.getNumDofs()
        self.d = foot_size / 2.
        self.µ = µ

        # left foot is the stance contact, right foot is swing.
        self.contact_l = True
        self.contact_r = False

        # define sizes for QP solver
        self.num_contacts = 2
        self.num_contact_dims = self.num_contacts * 6
        self.n_vars = 2 * self.dofs + self.num_contact_dims

        self.n_eq_constraints = self.dofs
        self.n_ineq_constraints = 8 * self.num_contacts

        # initialize QP solver
        self.qp_solver = QPSolver(self.n_vars, self.n_eq_constraints, self.n_ineq_constraints)

        # selection matrix for redundant dofs
        self.joint_selection = np.zeros((self.dofs, self.dofs))
        for i in range(self.dofs):
            joint_name = self.robot.getDof(i).getName()
            if joint_name in redundant_dofs:
                self.joint_selection[i, i] = 1

    def get_joint_torques(self, desired, current, contact=None):
        contact_l = self.contact_l
        contact_r = self.contact_r

        # robot parameters
        lsole = self.robot.getBodyNode('l_sole')
        rsole = self.robot.getBodyNode('r_sole')
        torso = self.robot.getBodyNode('torso')
        base  = self.robot.getBodyNode('body')

        # tasks: keep only what we need for single-support balancing.
        tasks = ['lfoot', 'com', 'torso', 'base', 'joints']

        # weights and gains
        weights   = {'lfoot':  1., 'com':  1., 'torso': 1., 'base': 1., 'joints': 1.e-2}
        pos_gains = {'lfoot': 10., 'com':  5., 'torso': 1., 'base': 1., 'joints': 10.  }
        vel_gains = {'lfoot': 10., 'com': 10., 'torso': 2., 'base': 2., 'joints': 1.   }

        # jacobians
        J = {'lfoot' : self.robot.getJacobian(lsole,        inCoordinatesOf=dart.dynamics.Frame.World()),
             'com'   : self.robot.getCOMLinearJacobian(     inCoordinatesOf=dart.dynamics.Frame.World()),
             'torso' : self.robot.getAngularJacobian(torso, inCoordinatesOf=dart.dynamics.Frame.World()),
             'base'  : self.robot.getAngularJacobian(base,  inCoordinatesOf=dart.dynamics.Frame.World()),
             'joints': self.joint_selection}

        # jacobians derivatives
        Jdot = {'lfoot' : self.robot.getJacobianClassicDeriv(lsole, inCoordinatesOf=dart.dynamics.Frame.World()),
                'com'   : self.robot.getCOMLinearJacobianDeriv(     inCoordinatesOf=dart.dynamics.Frame.World()),
                'torso' : self.robot.getAngularJacobianDeriv(torso, inCoordinatesOf=dart.dynamics.Frame.World()),
                'base'  : self.robot.getAngularJacobianDeriv(base,  inCoordinatesOf=dart.dynamics.Frame.World()),
                'joints': np.zeros((self.dofs, self.dofs))}

        # feedforward terms
        ff = {'lfoot' : desired['lfoot']['acc'],
              'com'   : desired['com']['acc'],
              'torso' : desired['torso']['acc'],
              'base'  : desired['base']['acc'],
              'joints': desired['joint']['acc']}

        # error vectors
        pos_error = {'lfoot' : pose_difference(desired['lfoot']['pos'] , current['lfoot']['pos'] ),
                     'com'   : desired['com']['pos'] - current['com']['pos'],
                     'torso' : rotation_vector_difference(desired['torso']['pos'], current['torso']['pos']),
                     'base'  : rotation_vector_difference(desired['base']['pos'] , current['base']['pos'] ),
                     'joints': desired['joint']['pos'] - current['joint']['pos']}

        # velocity error vectors
        vel_error = {'lfoot' : desired['lfoot']['vel'] - current['lfoot']['vel'],
                     'com'   : desired['com']['vel']   - current['com']['vel'],
                     'torso' : desired['torso']['vel'] - current['torso']['vel'],
                     'base'  : desired['base']['vel']  - current['base']['vel'],
                     'joints': desired['joint']['vel'] - current['joint']['vel']}

        # cost function
        H = np.zeros((self.n_vars, self.n_vars))
        F = np.zeros(self.n_vars)
        q_ddot_indices = np.arange(self.dofs)
        tau_indices = np.arange(self.dofs, 2 * self.dofs)
        f_c_indices = np.arange(2 * self.dofs, self.n_vars)

        for task in tasks:
            H_task =   weights[task] * J[task].T @ J[task]
            F_task = - weights[task] * J[task].T @ (ff[task]
                                                    + vel_gains[task] * vel_error[task]
                                                    + pos_gains[task] * pos_error[task]
                                                    - Jdot[task] @ current['joint']['vel'])

            H[np.ix_(q_ddot_indices, q_ddot_indices)] += H_task
            F[q_ddot_indices] += F_task

        # regularization term for contact forces
        H[np.ix_(f_c_indices, f_c_indices)] += np.eye(len(f_c_indices)) * 1e-6

        # dynamics constraints: M * q_ddot + C - J_c^T * f_c = tau
        inertia_matrix = self.robot.getMassMatrix()
        actuation_matrix = block_diag(np.zeros((6, 6)), np.eye(self.dofs - 6))

        # contact wrench ordering in f_c is [lfoot(6), rfoot(6)]
        J_l = self.robot.getJacobian(lsole, inCoordinatesOf=dart.dynamics.Frame.World())
        J_r = self.robot.getJacobian(rsole, inCoordinatesOf=dart.dynamics.Frame.World())
        contact_jacobian = np.vstack((contact_l * J_l, contact_r * J_r))

        A_eq = np.hstack((inertia_matrix, - actuation_matrix, - contact_jacobian.T))
        b_eq = - self.robot.getCoriolisAndGravityForces()

        # inequality constraints
        A_ineq = np.zeros((self.n_ineq_constraints, self.n_vars))
        b_ineq = np.zeros(self.n_ineq_constraints)
        A = np.array([[ 1, 0, 0, 0, 0, -self.d],
                      [-1, 0, 0, 0, 0, -self.d],
                      [0,  1, 0, 0, 0, -self.d],
                      [0, -1, 0, 0, 0, -self.d],
                      [0, 0, 0,  1, 0, -self.µ],
                      [0, 0, 0, -1, 0, -self.µ],
                      [0, 0, 0, 0,  1, -self.µ],
                      [0, 0, 0, 0, -1, -self.µ]])

        ineq_left_rows = slice(0, 8)
        ineq_right_rows = slice(8, 16)

        # Map each foot's 6D wrench into the full 12D (lfoot+rfoot) wrench vector.
        # Each assignment must be (8 x 12).
        left_block = np.hstack((A, np.zeros((8, 6))))
        right_block = np.hstack((np.zeros((8, 6)), A))

        A_ineq[ineq_left_rows,  f_c_indices] = (1.0 if contact_l else 0.0) * left_block
        A_ineq[ineq_right_rows, f_c_indices] = (1.0 if contact_r else 0.0) * right_block

        # solve the QP, compute torques and return them
        self.qp_solver.set_values(H, F, A_eq, b_eq, A_ineq, b_ineq)
        solution = self.qp_solver.solve()
        tau = solution[tau_indices]
        forces_contact = solution[f_c_indices]
        
        return tau[6:]