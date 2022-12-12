import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#Fixing random state for reproducibility
np.random.seed(2172081)


class differential_drive():

    # Constructor
    def __init__(self, x, y, theta, v, w, dt):
        self.x = x          # x position
        self.y = y          # y position
        self.theta = theta  # orientation
        self.v = v          # linear velocity
        self.w = w          # angular velocity
        self.dt = dt        # time step

        # State Estimates with EKF
        self.x_est = x          # x position estimate
        self.y_est = y          # y position estimate
        self.theta_est = theta  # orientation estimate

        # State Estimates without accounting for noise
        self.x_bad_est = x          # x position estimate
        self.y_bad_est = y          # y position estimate
        self.theta_bad_est = theta  # orientation estimate

    # Euler Discretization of the differential drive model
    def update(self):
        # Motion Model (Noise Free)
        self.x = self.x + self.v * np.cos(self.theta) * self.dt # x_k+1
        self.y = self.y + self.v * np.sin(self.theta) * self.dt # y_k+1
        self.theta = self.theta + self.w * self.dt              # theta_k+1

    # Update the state estimates without correcting for noise
    def update_bad_est(self):
        # Motion Model (With Process Noise)
        self.x_bad_est = self.x_bad_est + self.v * np.cos(self.theta_bad_est) * self.dt + np.random.normal(0, 0.03)   # x_hat_k+1
        self.y_bad_est = self.y_bad_est + self.v * np.sin(self.theta_bad_est) * self.dt + np.random.normal(0, 0.03)   # y_hat_k+1
        self.theta_bad_est = self.theta_bad_est + self.w * self.dt + np.random.normal(0, 0.01)                        # theta_hat_k+1

        # Observation Model (With Measurement Noise)
        self.x_bad_est = self.x_bad_est + np.random.normal(0, 0.015)           # h_k+1
        self.y_bad_est = self.y_bad_est + np.random.normal(0, 0.015)           # h_k+1
        self.theta_bad_est = self.theta_bad_est + np.random.normal(0, 0.005)   # h_k+1
    
    # Initialize the Extended Kalman Filter Parameters
    def EKF_initialize(self):

        # Prediction Covariance Matrix - Initial
        self.P = np.diag([0.001, 0.001, 0.001])

        # Process Noise Covariance Matrix
        self.Q = np.diag([0.03**2, 0.03**2, 0.01**2])

        # Measurement Noise Covariance Matrix
        self.R = np.diag([0.015**2, 0.015**2, 0.005**2])
    
    def EKF_predict(self):

        # Jacobian of the Motion Model
        self.F = np.array([[1.0, 0, -self.v * np.sin(self.theta_est) * self.dt],
                      [0, 1.0, self.v * np.cos(self.theta_est) * self.dt],
                      [0, 0, 1.0]])

        # Estimate Next State Based on Last Known State
        self.x_est = self.x_est + self.v * np.cos(self.theta_est) * self.dt
        self.y_est = self.y_est + self.v * np.sin(self.theta_est) * self.dt
        self.theta_est = self.theta_est + self.w * self.dt

        # Predict Covariance Matrix
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q

    def EKF_update(self):
 
        # Jacobian of the Observation Model
        self.H = np.array([[1.0, 0, 0],
                        [0, 1.0, 0],
                        [0, 0, 1.0]])       

        # Innovation (Residual) - Difference between the actual and predicted measurement
        y = np.array([self.x, self.y, self.theta]) - np.array([self.x_est, self.y_est, self.theta_est])

        # Innovation Covariance
        S = np.dot(self.H, self.P).dot(self.H.T) + self.R

        # Kalman Gain
        K = np.dot(self.P, self.H.T).dot(inv(S))

        # Update State Estimate
        self.x_est = self.x_est + np.dot(K, y)[0]
        self.y_est = self.y_est + np.dot(K, y)[1]
        self.theta_est = self.theta_est + np.dot(K, y)[2]
        
        # Update Covariance Matrix
        self.P = self.P - np.dot(K, self.H).dot(self.P)


    def get_state(self):
        return self.x, self.y, self.theta

    def get_bad_state_est(self):
        return self.x_bad_est, self.y_bad_est, self.theta_bad_est

    def get_state_est(self):
        return self.x_est, self.y_est, self.theta_est

    def set_state(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def set_input(self, v, w):
        self.v = v
        self.w = w

    def get_input(self):
        return self.v, self.w

    def get_time(self):
        return self.dt

    def set_time(self, dt):
        self.dt = dt

    def get_position(self):
        return self.x, self.y

    def get_position_est(self):
        return self.x_est, self.y_est

    def get_bad_position_est(self):
        return self.x_bad_est, self.y_bad_est

    def get_orientation(self):
        return self.theta

    def get_orientation_est(self):
        return self.theta_est

    def get_bad_orientation_est(self):
        return self.theta_bad_est


if __name__ == "__main__":

    # Initialize the robot with x = 0, y = 0, theta = 0, v = 0, w = 0, dt = 0.1
    robot = differential_drive(0, 0, 0, 1, 0, 0.1)

    # Initial Conditions
    v_input = 1
    w_input = 0

    # Limits on the input (+/-)
    v_lim = 2
    w_lim = 1

    # Update rate
    update_rate = 1

    # Perform 1000 iterations of the robot motion
    for k in range(0, 1000):

        # Print Time Step
        #print("Time Step: ", k)

        # create random normally distributed changes in the input 
        v_input += np.random.normal(0, 0.05)
        w_input += np.random.normal(0, 0.01)

        # limit the input
        if abs(v_input) > v_lim:
            v_input = v_lim*np.sign(v_input)
        if abs(w_input) > w_lim:
            w_input = w_lim*np.sign(w_input)

        # set the input to the robot
        robot.set_input(v_input, w_input)

        # Use EKF to estimate the robot state
        if k == 0:
            # Initialize the EKF on step 1
            #print("EKF Initialized")
            robot.EKF_initialize()
        else:
            # Predict the state estimate using linearized motion model for k > 0 always predict based on last state estimate
            #print("Prediction Step")
            robot.EKF_predict()

        if k % update_rate == 0 and k != 0:
            # Update the state estimate using linearized observation model for k > 0 and after certain number of steps
            #print("Update Step")
            robot.EKF_update()

        # Update the robot state and state estimates using motion model then take observations
        robot.update()

        # Update robot state in presence of noise without correction
        robot.update_bad_est()

        # Debug Statements
        #print("Robot State (x,y,theta): ", robot.get_state())
        #print("Robot Input (v,w): ", robot.get_input())

        # Record the robot trajectory and error
        if k == 0:
            robot_trajectory = np.array(robot.get_state())
            robot_trajectory_bad_est = np.array(robot.get_bad_state_est())
            robot_trajectory_est = np.array(robot.get_state_est())
            error_without_ekf = np.array(robot.get_state()) - np.array(robot.get_bad_state_est())
            error_with_ekf = np.array(robot.get_state()) - np.array(robot.get_state_est())
        else:
            robot_trajectory = np.vstack((robot_trajectory, robot.get_state())) 
            robot_trajectory_bad_est = np.vstack((robot_trajectory_bad_est, robot.get_bad_state_est()))
            robot_trajectory_est = np.vstack((robot_trajectory_est, robot.get_state_est()))
            error_without_ekf = np.vstack((error_without_ekf, np.array(robot.get_state()) - np.array(robot.get_bad_state_est())))
            error_with_ekf = np.vstack((error_with_ekf, np.array(robot.get_state()) - np.array(robot.get_state_est())))

    plt.figure(1)

    # plot the true robot trajectory
    plt.plot(robot_trajectory[:,0], robot_trajectory[:,1], 'b-')

    # plot the badly estimated robot trajectory
    plt.plot(robot_trajectory_bad_est[:,0], robot_trajectory_bad_est[:,1], 'm-')

    # plot the estimated robot trajectory using EKF
    plt.plot(robot_trajectory_est[:,0], robot_trajectory_est[:,1], 'r-')

    # Plot the initial location as a red x
    plt.plot(0, 0, 'rx')

     # Plot the end location as a green x
    plt.plot(robot.get_position()[0], robot.get_position()[1], 'gx')

    # Add title and axis labels
    plt.title('Robot Trajectory & Estimates (Update Rate = %d)' % update_rate)
    plt.xlabel('x-position')
    plt.ylabel('y-position')

    # Legend
    plt.legend(['True Trajectory', 'Dead-Reckoned Trajectory', 'Estimated Trajectory', 'End Location', 'Start Location'])

    # Show Figure 1
    plt.show()
    plt.close()
    
    plt.figure(2)

    plt.plot(error_without_ekf[:,0], 'b-')
    plt.plot(error_without_ekf[:,1], 'r-')
    plt.plot(error_without_ekf[:,2], 'g-')
    plt.axhline(y = 0, color = 'k', linestyle = 'dashed')

    # Add title and axis labels
    plt.title('Error - Dead Reckoning')
    plt.xlabel('Time Step (k)')
    plt.ylabel('Error')

    # Legend
    plt.legend(['x-error', 'y-error', 'theta-error'])

    # Show Figure 2
    plt.show()
    plt.close()

    # EKF Error
    plt.figure(3)

    plt.plot(error_with_ekf[:,0], 'b-')
    plt.plot(error_with_ekf[:,1], 'r-')
    plt.plot(error_with_ekf[:,2], 'g-')
    plt.axhline(y = 0, color = 'k', linestyle = 'dashed')

    # Add title and axis labels
    plt.title('Error - Extended Kalman Filter (Update Rate = %d)' % update_rate)
    plt.xlabel('Time Step (k)')
    plt.ylabel('Error')

    # Legend
    plt.legend(['x-error', 'y-error', 'theta-error'])

    # Show Figure 3
    plt.show()
    plt.close()

    # Error in x comparison
    plt.figure(4)

    plt.plot(error_without_ekf[:,0], 'b-')
    plt.plot(error_with_ekf[:,0], 'r-')
    plt.axhline(y = 0, color = 'k', linestyle = 'dashed')

    # Add title and axis labels
    plt.title('Error in x (Update Rate = %d)' % update_rate)
    plt.xlabel('Time Step (k)')
    plt.ylabel('Error')

    # Legend
    plt.legend(['Dead Reckoning', 'Extended Kalman Filter'])

    # Show Figure 4
    plt.show()
    plt.close()

    # Error in y comparison
    plt.figure(5)

    plt.plot(error_without_ekf[:,1], 'b-')
    plt.plot(error_with_ekf[:,1], 'r-')
    plt.axhline(y = 0, color = 'k', linestyle = 'dashed')

    # Add title and axis labels
    plt.title('Error in y (Update Rate = %d)' % update_rate)
    plt.xlabel('Time Step (k)')
    plt.ylabel('Error')

    # Legend
    plt.legend(['Dead Reckoning', 'Extended Kalman Filter'])

    # Show Figure 5
    plt.show()
    plt.close()

    # Error in theta comparison
    plt.figure(6)

    plt.plot(error_without_ekf[:,2], 'b-')
    plt.plot(error_with_ekf[:,2], 'r-')
    plt.axhline(y = 0, color = 'k', linestyle = 'dashed')

    # Add title and axis labels
    plt.title('Error in Theta (Update Rate = %d)' % update_rate)
    plt.xlabel('Time Step (k)')
    plt.ylabel('Error')
    
    # Legend
    plt.legend(['Dead Reckoning', 'Extended Kalman Filter'])

    # Show Figure 6
    plt.show()
    plt.close()





