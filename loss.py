import numpy as np
class HuberPoseLoss:
    def __init__(self,
                 ang_weight: float = 8.0,
                 delta_xyz:  float = 0.055,
                 delta_ang:  float = 0.16):
        self.ang_weight = ang_weight
        self.delta_xyz  = delta_xyz
        self.delta_ang  = delta_ang

    def _huber(self, e: np.ndarray, delta: float) -> float:
        abs_e = np.abs(e)
        loss  = np.where(
            abs_e <= delta,
            0.5 * e ** 2,
            delta * (abs_e - 0.5 * delta)
        )
        return float(loss.mean())

    def __call__(self,
                 pred:   np.ndarray,
                 target: np.ndarray):
        loss_xyz = self._huber(pred[:, :3] - target[:, :3], self.delta_xyz)
        loss_ang = self._huber(pred[:, 3:] - target[:, 3:], self.delta_ang)
        total    = loss_xyz + self.ang_weight * loss_ang
        return total, loss_xyz, loss_ang