import numpy as np

class CustomSimulator:
    """手写 RK4 仿真器"""
    def __init__(self, num: list, den: list):
        # 【FIX 3】因果性检查
        # 物理可实现系统要求：分母阶次 >= 分子阶次
        # den阶次 = len(den) - 1, num阶次 = len(num) - 1
        if len(num) > len(den):
             raise ValueError(f"物理不可实现：分子阶次({len(num)-1})高于分母阶次({len(den)-1})")

        scale = den[-1]
        if abs(scale) < 1e-12:
             raise ValueError("分母最高次系数不能为0")

        self.num = [c / scale for c in num]
        self.den = [c / scale for c in den]
        self.n = len(den) - 1
        
        # 补齐分子 (使其长度等于分母，方便算法处理)
        if len(self.num) < len(self.den):
            self.num += [0.0] * (len(self.den) - len(self.num))

        # 构建能控标准型
        self.A = np.zeros((self.n, self.n))
        self.B = np.zeros((self.n, 1))
        self.C = np.zeros((1, self.n))
        
        # 构造A矩阵 (Bottom companion form)
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
        
        # 【FIX 3】更健壮的稳态值计算
        # 取最后 10% 的数据点计算平均值，而非固定的50个点
        # 避免仿真点数很少时报错，或点数极多时范围太小
        lookback = max(1, int(len(y) * 0.1))
        self.y_final = np.mean(y[-lookback:])

    def get_metrics(self, ts_tol=0.02):
        y_max = np.max(self.y)
        
        # 避免除以零
        if abs(self.y_final) > 1e-9:
            overshoot = (y_max - self.y_final)/self.y_final * 100 
        else:
            overshoot = 0.0

        # 调节时间计算
        ts = 0
        upper, lower = self.y_final*(1+ts_tol), self.y_final*(1-ts_tol)
        
        # 从后向前扫描，找到第一个超出误差带的点
        in_band = True
        for i in range(len(self.y)-1, -1, -1):
            if self.y[i] > upper or self.y[i] < lower:
                ts = self.t[i+1] if i+1 < len(self.t) else self.t[-1]
                in_band = False
                break
        
        # 如果从未进入误差带（发散或一直在外），Ts设为仿真结束时间
        if in_band: # 如果一开始就在误差带内（不太可能，除非初始值就在目标值）
             ts = 0 
        
        return {
            "steady_val": self.y_final,
            "error": abs(self.target - self.y_final),
            "overshoot": overshoot,
            "ts": ts
        }