import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from footstep_planner import FootstepPlanner

class LIPCOMPlanner:
    """
    Linear Inverted Pendulum (LIP) model for COM trajectory generation.
    """
    
    def __init__(self, param):
        self.z_com = param['h']
        self.g = param['g']
        self.dt = param['world_time_step']
        
        # ---------------------------------------------------------
        # TODO: Calculate the natural frequency (omega) of the LIP model
        # Hint: omega = sqrt(g / z_com)
        self.omega = 0.0 
        # ---------------------------------------------------------

    def interpolate_zmp_trajectory(self, footstep_plan, total_time):
        """
        Creates ZMP trajectory by interpolating through footstep positions.
        """
        t_array = np.arange(0, total_time, self.dt)
        zmp_traj = np.zeros((2, len(t_array)))
        
        t = 0
        for step_idx, step in enumerate(footstep_plan):
            ss_duration = step['ss_duration']
            ds_duration = step['ds_duration']
            
            # Single support phase: ZMP at support foot
            ss_end = t + ss_duration
            ss_indices = np.where((t_array >= t) & (t_array < ss_end))[0]
            if len(ss_indices) > 0:
                zmp_traj[:, ss_indices] = step['pos'][:2, np.newaxis]
            t = ss_end
            
            # Double support phase: Interpolate
            ds_end = t + ds_duration
            if step_idx + 1 < len(footstep_plan):
                next_pos = footstep_plan[step_idx + 1]['pos'][:2]
                curr_pos = step['pos'][:2]
                ds_indices = np.where((t_array >= t) & (t_array < ds_end))[0]
                if len(ds_indices) > 0:
                    alpha = np.linspace(0, 1, len(ds_indices))
                    zmp_traj[:, ds_indices] = (curr_pos[:, None] * (1 - alpha) + next_pos[:, None] * alpha)
            else:
                ds_indices = np.where((t_array >= t) & (t_array < ds_end))[0]
                if len(ds_indices) > 0:
                    zmp_traj[:, ds_indices] = step['pos'][:2, np.newaxis]
            t = ds_end
        
        return t_array, zmp_traj

    def solve_lip_dynamics(self, zmp_traj, t_array, x0_com, v0_com):
        """
        Task: Solve LIP dynamics using forward integration.
        Equation: d²x/dt² = ω² * (x - x_zmp)
        """
        n_timesteps = zmp_traj.shape[1]
        
        com_traj = np.zeros((2, n_timesteps))
        com_vel = np.zeros((2, n_timesteps))
        com_acc = np.zeros((2, n_timesteps))
        
        # Initial conditions
        com_traj[:, 0] = x0_com
        com_vel[:, 0] = v0_com
        
        omega_sq = self.omega ** 2
        
        for i in range(n_timesteps - 1):
            # 1. Get current state (x, v) and current ZMP (p)
            x = com_traj[:, i]
            v = com_vel[:, i]
            x_zmp = zmp_traj[:, i]
            
            # ---------------------------------------------------------
            # TODO: STUDENTS IMPLEMENT DYNAMICS HERE
            # 1. Calculate acceleration 'a' based on LIP dynamics
            #    a = omega^2 * (x - x_zmp)
            # 2. Integrate to find next velocity 'v_next' using Forward Euler
            # 3. Integrate to find next position 'x_next' using Forward Euler
            
            a = np.zeros(2) # Replace with LIP equation
            
            # Euler Integration
            com_vel[:, i + 1] = v # + ...
            com_traj[:, i + 1] = x # + ...
            # ---------------------------------------------------------
            
            com_acc[:, i] = a

        # Compute final acceleration for completeness
        com_acc[:, -1] = omega_sq * (com_traj[:, -1] - zmp_traj[:, -1])
        
        return com_traj, com_vel, com_acc

    def plan_com_trajectory(self, footstep_plan, initial_com_pos, initial_com_vel=None):
        if initial_com_vel is None:
            initial_com_vel = np.array([0.0, 0.0])
        
        total_time = sum(step['ss_duration'] + step['ds_duration'] for step in footstep_plan)
        t_array, zmp_traj = self.interpolate_zmp_trajectory(footstep_plan, total_time)
        
        com_pos, com_vel, com_acc = self.solve_lip_dynamics(
            zmp_traj, t_array, initial_com_pos, initial_com_vel
        )
        
        return {'t': t_array, 'com_pos': com_pos, 'zmp': zmp_traj}

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Parameters
    params = {
        'g': 9.81,
        'h': 0.72,
        'world_time_step': 0.01,
        'ss_duration': 80,
        'ds_duration': 20,
        'first_swing': 'rfoot',
        'foot_size': 0.05,
        # Visualization params
        'vis_foot_width': 0.1, 
        'vis_foot_length': 0.2 
    }
    
    # Initial foot positions [ang_x, ang_y, ang_z, x, y, z]
    initial_lfoot = np.array([0., 0.1, 0., 0., 0.1, 0.]) 
    initial_rfoot = np.array([0., -0.1, 0., 0., -0.1, 0.])
    
    # Velocity reference sequence
    vref = np.array([
        [0.0, 0.0, 0.0],   # Step 1: Start
        [0.0, 0.0, 0.0],   # Step 2
        [0.2, 0.0, 0.0],   # Step 3: Move forward
        [0.2, 0.0, 0.0],   # Step 4
        [0.0, 0.0, 0.0],   # Step 5: Stop
    ])
    
    print("Generating footsteps...")
    planner = FootstepPlanner(vref, initial_lfoot, initial_rfoot, params)
    
    print("\nGenerating COM trajectory using LIP...")
    lip_planner = LIPCOMPlanner(params)
    
    # Start COM between the feet
    initial_com = (initial_lfoot[3:5] + initial_rfoot[3:5]) / 2.0
    
    traj = lip_planner.plan_com_trajectory(
        planner.plan, 
        initial_com,
        initial_com_vel=np.array([0.0, 0.0])
    )

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plot Footsteps (rectangles)
    for step in planner.plan:
        x, y = step['pos'][0], step['pos'][1]
        # Use simple visualization dimensions
        w = params['vis_foot_length']
        h = params['vis_foot_width']
        
        rect = patches.Rectangle(
            (x - w/2, y - h/2), w, h,
            linewidth=1, edgecolor='r', facecolor='none', label='Footstep'
        )
        ax.add_patch(rect)
        
    # 2. Plot ZMP
    ax.plot(traj['zmp'][0, :], traj['zmp'][1, :], 'g--', label='ZMP Ref', linewidth=2)
    
    # 3. Plot COM
    ax.plot(traj['com_pos'][0, :], traj['com_pos'][1, :], 'b-', label='CoM Trajectory', linewidth=2)
    
    # Fix legend duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_title("LIP Model: Forward Integration Results")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis('equal')
    ax.grid(True)
    plt.show()
