'''hw6p5.py

   This is the skeleton code for HW6 Problem 5.

   This explores the singularity handing while following a circle
   outside the workspace.

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from hw6code.GeneratorNode     import GeneratorNode
from hw6code.KinematicChain    import KinematicChain
from hw5code.TransformHelpers  import *


def spline(t, T, p0, pf):
    p = p0 + (pf-p0) * (3*t**2/T**2 - 3*t**3/T**3)
    v =      (pf-p0) * (6*t   /T**2 - 6*t**2/T**3)
    return (p, v)

class Q:
    def __init__(self, joint_names) -> None:
        self.joint_names = joint_names
        self.joint_values = dict([(joint_name, 0) for joint_name in joint_names])
        
    def setAll(self, value):
        self.joint_values = dict([(joint_name, value) for joint_name in self.joint_names])

    def setSome(self, joints, values):
        if isinstance(values, np.ndarray):
            values = values.flatten()

        for (i, joint) in enumerate(joints):
            if (joint not in self.joint_values):
                raise IndexError("Invalid Jointname pass into set")
            self.joint_values[joint] = values[i]

    def retSome(self, joints):
        return np.array([self.joint_values[joint] for joint in joints]).reshape((-1, 1))

    def retAll(self):
        return self.retSome(self.joint_names)

    def __len__(self):
        return len(self.joint_names)

class Jacobian():
    def __init__(self, joints, chain) -> None:

        Jv = chain.Jv()
        Jw = chain.Jw()
        J = np.vstack((Jv, Jw))
        print("Jacobian Shape", J.shape, "Length", len(joints))
        self.Jacobian = J
        self.joints = joints
        self.columns = dict([(joint, J[:, i]) for (i, joint) in enumerate(self.joints)])

    def merge(Js, joints):
        columns = len(joints)
        rows = sum([J.Jacobian.shape[0] for J in Js])
        mergedJ = np.zeros((rows, columns))

        rowInput = 0
        for J in Js:
            size = J.Jacobian.shape[0]
            for (columnNo, joint) in enumerate(joints):
                if joint in J.columns:
                    mergedJ[rowInput:rowInput + size, columnNo] = J.columns[joint]
                
            rowInput += size
                
        return mergedJ

class X():
    def __init__(self, chains) -> None:
        # Chains should just be a dictionary of chains
        self.chains = chains

    def calculateError(self, desiredValues):
        # desiredValues = [(chain1p, chain1r), (chain2p, chain2r), ...]
        
        tipPoses = [(self.chains[i].ptip(), self.chains[i].rtip()) for (i, (desP, desR)) in enumerate(desiredValues)]
        errors = [(ep(tipP, desP), eR(tipR, desR)) for ((tipP, tipR), (desP, desR)) in zip(tipPoses, desiredValues)]
        
        flatErrors = [item for sublist in errors for item in sublist]
        return np.vstack(tuple(flatErrors))

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        
        self.chain_world_pelvis = KinematicChain(node, 'world', 'pelvis', self.jointnames('world_pelvis'))
        
        self.chain_left_arm = KinematicChain(node, 'world', 'l_lfarm', self.jointnames('left_arm'))
        self.chain_right_arm = KinematicChain(node, 'world', 'r_lfarm', self.jointnames('right_arm'))
        self.chain_left_leg = KinematicChain(node, 'world', 'l_foot', self.jointnames('left_leg'))
        self.chain_right_leg = KinematicChain(node, 'world', 'r_foot', self.jointnames('right_leg'))
        

        # Initialize the current joint position and chain data.
        # TODO Initialize all the chains 

        self.Q = Q(self.jointnames())
        self.Qdot = Q(self.jointnames())
        
        self.Qdot.setAll(0)
        self.Q.setAll(0)
        self.Q.setSome(['r_arm_shx', 'l_arm_shx', 'r_arm_shz', 'l_arm_shz', 'rotate_y', 'mov_z'], np.array([0.25, -0.25, np.pi/2, -np.pi/2, 0.95, 0.51]))
        
        self.chain_left_arm.setjoints(self.Q.retSome(self.jointnames('left_arm')))
        self.chain_right_arm.setjoints(self.Q.retSome(self.jointnames('right_arm')))
        self.chain_left_leg.setjoints(self.Q.retSome(self.jointnames('left_leg')))
        self.chain_right_leg.setjoints(self.Q.retSome(self.jointnames('right_leg')))
        
        self.chain_world_pelvis.setjoints(self.Q.retSome(self.jointnames('world_pelvis')))

        # Also zero the task error.
        self.err = np.zeros((24, 1))

        # Pick the convergence bandwidth.
        self.lam = 30


    # Declare the joint names.
    def jointnames(self, which_chain='all'):
        # Return a list of joint names FOR THE EXPECTED URDF!
        # TODO Implement better method to find the jointnames
        
        joints = {
        
        'left_arm':['back_bkz', 'back_bky', 'back_bkx', 'l_arm_shz','l_arm_shx', 'l_arm_ely', 'l_arm_elx', 'l_arm_wry', 'l_arm_wrx'],
        
         'right_arm': ['back_bkz', 'back_bky', 'back_bkx', 'r_arm_shz','r_arm_shx', 'r_arm_ely', 'r_arm_elx', 'r_arm_wry', 'r_arm_wrx'],
         
         'left_leg': ['l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky', 'l_leg_akx'] , 
         
         'right_leg':['r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx'], 
         
         'world_pelvis':['mov_x', 'mov_y', 'mov_z', 'rotate_x', 'rotate_y', 'rotate_z']
         
        }
         
        if which_chain == 'all':
            return joints['world_pelvis'] + joints['left_arm'] + joints['right_arm'][3:] + joints['left_leg'] + joints['right_leg']
         
        if which_chain == 'world_pelvis':
            return joints['world_pelvis']
             
        return joints['world_pelvis'] + joints[which_chain]

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):
    
        # Cyclic (sinusoidal) movements, after the first 3s.
        s    =               np.cos(np.pi/2.5 * (t-3))
        sdot = - np.pi/2.5 * np.sin(np.pi/2.5 * (t-3))

        # Use the path variables to compute the position trajectory.
        pd = np.array([-0.3*s    , 0.5, 0.75-0.6*s**2  ]).reshape((3,1))
        vd = np.array([-0.3*sdot , 0.0,     -1.2*s*sdot]).reshape((3,1))

        pd = np.zeros((len(self.Q), 1))
        vd = np.zeros((len(self.Q), 1))
        
        Rd = Reye()
        wd = np.zeros((3,1))
        
        # Grab the last joint value and task error.
        
        q = self.Q.retAll()
        err = self.err
        
        # Compute the inverse kinematics

        J_left_arm = Jacobian(self.jointnames('left_arm'), self.chain_left_arm)
        J_right_arm = Jacobian(self.jointnames('right_arm'), self.chain_right_arm)
        J_left_leg = Jacobian(self.jointnames('left_leg'), self.chain_left_leg)
        J_right_leg = Jacobian(self.jointnames('right_leg'), self.chain_left_leg)
        JMerged = Jacobian.merge([J_left_arm, J_right_arm, J_left_leg, J_right_leg], self.jointnames())

        print(f"JMerged Shape: {JMerged.shape}")

        # TODO
        xdot = np.zeros((24, 1))
        print(f"xdot {xdot.shape}, error {err.shape}" )

        qdot = np.linalg.pinv(JMerged) @ (xdot + self.lam * err)

        # Integrate the joint position and update the kin chain data.
        q = q + dt * qdot
        
        print(q.shape)
        print(len(self.Q))

        self.Q.setSome(self.jointnames(), q)
        self.Qdot.setAll(qdot)

        print(self.Q.retSome(self.jointnames('left_arm')).shape)

        self.chain_left_arm.setjoints(self.Q.retSome(self.jointnames('left_arm')))
        self.chain_right_arm.setjoints(self.Q.retSome(self.jointnames('right_arm')))
        self.chain_left_leg.setjoints(self.Q.retSome(self.jointnames('left_leg')))
        self.chain_right_leg.setjoints(self.Q.retSome(self.jointnames('right_leg')))
        

        # Compute the resulting task error (to be used next cycle).
        # err  = np.vstack((ep(pd, self.chain_left_arm.ptip()), eR(Rd, self.chain_left_arm.Rtip())))
        err = np.zeros((24, 1))
        # Save the joint value and task error for the next cycle.
        self.err = err

        # Return the position and velocity as python lists.
        # return (q.flatten().tolist(), qdot.flatten().tolist())       
        # test code
        q_all = self.Q.retAll()

        qdot_all = self.Qdot.retAll()

        self.Q2 = Q(self.jointnames())
        self.Qdot2 = Q(self.jointnames())
        
        self.Qdot2.setAll(0.0)
        self.Q2.setAll(0.0)
        self.Q2.setSome(['r_arm_shx', 'l_arm_shx', 'r_arm_shz', 'l_arm_shz', 'rotate_y', 'mov_z'], np.array([0.25, -0.25, np.pi/2, -np.pi/2, 0.95, 0.51]))
        
        q_all = self.Q2.retAll()
        qdot_all = self.Qdot2.retAll()
        
        return (q_all.flatten().tolist(), qdot_all.flatten().tolist())

        


#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the generator node (100Hz) for the Trajectory.
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

