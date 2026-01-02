import numpy as np

class CustomSimulator:
    """手写 RK4 仿真器"""
    def __init__(self, num: list, den: list):
        scale = den[-1]
        self.num = [c / scale for c in num]
        self.den = [c / scale for c in den]
        self.n = len(den) - 1
        
        # 补齐分子
        if len(self.num) < len(self.den):
            self.num += [0.0] * (len(self.den) - len(self.num))

        # 构建能控标准型
        self.A = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n, 1))
        self.C = np.zeros((1, self.n))
        
        for i in range(self.n - 1): self.A[i, i+1] = 1.0
        self.A[self.n-1, :] = -np.array(self.den[:-1])
        self.B[self.n-1, 0] = 1.0
        
        self.D = self.num[-1]
        for i in range(self.n):
            self.C[0, i] = self.num[i] - self.den[i] * self.D
        
        self.state = np.zeros((self.n, 1))

    def step(self, u, dt):
        def dyn(x): return self.A @ x + self.B * u
        k1 = dyn(self.state)
        k2 = dyn(self.state + 0.5*dt*k1)
        k3 = dyn(self.state + 0.5*dt*k2)
        k4 = dyn(self.state + dt*k3)
        self.state += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        return float(self.C @ self.state + self.D * u)

class PerformanceAnalyzer:
    """性能指标计算"""
    def __init__(self, t, y, target):
        self.t = t
        self.y = y
        self.target = target
        self.y_final = np.mean(y[-50:])

    def get_metrics(self, ts_tol=0.02):
        y_max = np.max(self.y)
        overshoot = (y_max - self.y_final)/self.y_final * 100 if self.y_final!=0 else 0
        
        # 调节时间
        ts = 0
        upper, lower = self.y_final*(1+ts_tol), self.y_final*(1-ts_tol)
        for i in range(len(self.y)-1, 0, -1):
            if self.y[i] > upper or self.y[i] < lower:
                ts = self.t[i+1] if i+1 < len(self.t) else self.t[-1]
                break
        
        return {
            "steady_val": self.y_final,
            "error": abs(self.target - self.y_final),
            "overshoot": overshoot,
            "ts": ts
        }