import numpy as np

class CustomSimulator:
    """
    通用 SISO 线性系统仿真器 (改进版：分离输出与状态更新)
    适用于传递函数 G(s) = Num(s) / Den(s)
    """
    def __init__(self, num: list, den: list):
        # 因果性检查
        if len(num) > len(den):
             raise ValueError(f"物理不可实现：分子阶次({len(num)-1})高于分母阶次({len(den)-1})")

        scale = den[-1]
        if abs(scale) < 1e-12:
             raise ValueError("分母最高次系数不能为0")

        # 归一化并转为 float
        self.num = [float(c) / scale for c in num]
        self.den = [float(c) / scale for c in den]
        self.n = len(den) - 1
        
        # 补齐分子 (使其长度等于分母)
        if len(self.num) < len(self.den):
            self.num += [0.0] * (len(self.den) - len(self.num))

        # 构建能控标准型 (Control Canonical Form)
        self.A = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n, 1))
        self.C = np.zeros((1, self.n))
        
        # A 矩阵构造
        if self.n > 0:
            for i in range(self.n - 1): self.A[i, i+1] = 1.0
            self.A[self.n-1, :] = -np.array(self.den[:-1])
            self.B[self.n-1, 0] = 1.0
        
        # C, D 矩阵构造
        self.D = self.num[-1]
        for i in range(self.n):
            self.C[0, i] = self.num[i] - self.den[i] * self.D
        
        self.state = np.zeros((self.n, 1))

    def compute_output(self, u_in):
        """
        计算当前时刻的输出 y(t) = C*x(t) + D*u(t)
        不更新状态，无副作用。
        """
        u = float(u_in)
        return float(self.C @ self.state + self.D * u)

    def update_state(self, u_in, dt):
        """
        更新状态 x(t) -> x(t+dt) 使用 RK4 算法
        """
        u = float(u_in)
        def dyn(x): return self.A @ x + self.B * u
        
        if self.n > 0:
            k1 = dyn(self.state)
            k2 = dyn(self.state + 0.5*dt*k1)
            k3 = dyn(self.state + 0.5*dt*k2)
            k4 = dyn(self.state + dt*k3)
            self.state += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

class PerformanceAnalyzer:
    """性能指标计算"""
    def __init__(self, t, y, target):
        self.t = t
        self.y = y
        self.target = target
        # 取最后 10% 数据计算稳态
        lookback = max(1, int(len(y) * 0.1))
        self.y_final = np.mean(y[-lookback:])

    def get_metrics(self, ts_tol=0.02):
        y_max = np.max(self.y)
        if abs(self.y_final) > 1e-9:
            overshoot = (y_max - self.y_final)/self.y_final * 100 
        else:
            overshoot = 0.0

        ts = 0
        upper, lower = self.y_final*(1+ts_tol), self.y_final*(1-ts_tol)
        in_band = True
        
        for i in range(len(self.y)-1, -1, -1):
            if self.y[i] > upper or self.y[i] < lower:
                ts = self.t[i+1] if i+1 < len(self.t) else self.t[-1]
                in_band = False
                break
        
        if in_band: ts = 0 
        
        return {
            "steady_val": self.y_final,
            "error": abs(self.target - self.y_final),
            "overshoot": overshoot,
            "ts": ts
        }