import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, error_dim=12):
        # Initialize the PID controller with provided gains and error dimension
        # Validate that the lengths of kp, ki, kd arrays match the specified error dimension
        if len(kp) != error_dim or len(ki) != error_dim or len(kd) != error_dim:
            raise ValueError(f"Size of kp, ki, and kd must be {error_dim} for a {error_dim}-dimensional error input.")

        # Set PID gains
        self.kp = np.array(kp)  # Proportional gains
        self.ki = np.array(ki)  # Integral gains
        self.kd = np.array(kd)  # Derivative gains

        # Initialize error and integral terms
        self.last_errors = np.zeros(error_dim)  # Previous error values, for derivative term calculation
        self.integrals = np.zeros(error_dim)    # Integral term accumulators

    def compute_control_output(self, errors, dt):
        # Compute the control output based on current errors and time delta

        # Check if the size of errors array matches the PID dimensionality
        if len(errors) != len(self.kp):
            raise ValueError(f"Size of errors must be {len(self.kp)} for a {len(self.kp)}-dimensional error input.")
        
        # Update integrals (accumulated error over time)
        self.integrals += errors * dt

        # Calculate PID terms
        proportional_term = self.kp * errors  # Proportional term
        integral_term = self.ki * self.integrals  # Integral term
        derivative_term = self.kd * (self.last_errors - errors) / dt  # Derivative term

        # Combine PID terms to form the control output
        control_output = proportional_term + integral_term + derivative_term#[1:3] - derivative_term[4:6]

        # Update last error values for next derivative calculation
        self.last_errors = errors

        return control_output
