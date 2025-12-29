import numpy as np

from .PIControl_base import PIControl_base

DELIVERABLE = "3"  # Change to "3" or"4"

class PIControl:

	PI_x: PIControl_base = None
	PI_y: PIControl_base = None
	PI_z: PIControl_base = None

	if DELIVERABLE == "3": 

		def __init__(self, pos_target: np.ndarray) -> None:
			self.PI_x = PIControl_base(Kp=0.2, Ki=0.0, Ts=0.05, xs=pos_target[0], u_min=-10.0, u_max=10.0)
			self.PI_y = PIControl_base(Kp=0.2, Ki=0.0, Ts=0.05, xs=pos_target[1], u_min=-10.0, u_max=10.0)
			self.PI_z = PIControl_base(Kp=0.2, Ki=0.0, Ts=0.05, xs=pos_target[2], u_min=-10.0, u_max=10.0)
	else:
		def __init__(self, pos_target: np.ndarray) -> None:
			self.PI_x = PIControl_base(Kp=0.1, Ki=0.0, Ts=0.05, xs=pos_target[0], u_min=-10.0, u_max=10.0)
			self.PI_y = PIControl_base(Kp=0.1, Ki=0.0, Ts=0.05, xs=pos_target[1], u_min=-10.0, u_max=10.0)
			self.PI_z = PIControl_base(Kp=0.15, Ki=0.0, Ts=0.05, xs=pos_target[2], u_min=-10.0, u_max=10.0)

	def get_u(self, x: np.ndarray) -> np.ndarray:
		u = np.zeros(3)
		u[0] = self.PI_x.compute_control(x[0])
		u[1] = self.PI_y.compute_control(x[1])
		u[2] = self.PI_z.compute_control(x[2])
		return u