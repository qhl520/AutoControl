import numpy as np
import math
from math_core import PolynomialUtils, PoleUtils

def count_integrators(den: list) -> int:
    """【智能检测】被控对象已有的积分环节数量"""
    cnt = 0
    for c in den:
        if abs(c) < 1e-6: cnt += 1
        else: break
    return cnt

def design_controller(num, den, mp, ts, input_type='step'):
    """Diophantine 方程求解器"""
    # 1. 期望极点计算
    ln_mp = math.log(mp/100) if mp > 0 else -999
    zeta = -ln_mp / math.sqrt(math.pi**2 + ln_mp**2)
    wn = 4.0 / (zeta * ts)
    
    # 构造复数主极点
    p_real = -zeta * wn
    p_imag = wn * math.sqrt(1 - zeta**2)
    desired_pole = complex(p_real, p_imag)
    
    # 2. 智能积分补偿
    req_type = 2 if input_type == 'ramp' else 1
    cur_type = count_integrators(den)
    r_add = max(0, req_type - cur_type)
    
    # 构造广义对象分母
    s_term = [1.0]
    for _ in range(r_add):
        s_term = PolynomialUtils.multiply(s_term, [0.0, 1.0])
    Dp_ext = PolynomialUtils.multiply(den, s_term)
    
    # 3. 构造闭环特征多项式 A_cl
    n_ext = len(Dp_ext) - 1
    deg_ctrl = n_ext - 1
    total_order = n_ext + deg_ctrl
    
    # 【核心】使用 PoleUtils 正确处理共轭极点
    complex_poles = PoleUtils.conjugate_pair([desired_pole])
    A_cl = [1.0]
    for p in complex_poles:
        A_cl = PolynomialUtils.multiply(A_cl, [-p, 1.0]) # (s-p)
    A_cl = [c.real for c in A_cl] # 转实数

    # 补全远极点
    while len(A_cl) - 1 < total_order:
        A_cl = PolynomialUtils.multiply(A_cl, [10.0, 1.0])
        
    # 4. 求解 Sylvester 矩阵
    num_vars = (deg_ctrl + 1) * 2
    M = np.zeros((num_vars, num_vars))
    b_vec = np.zeros(num_vars)
    
    for i in range(min(len(A_cl), num_vars)): b_vec[i] = A_cl[i]
    
    # 填充 Dp_ext 列
    for j in range(deg_ctrl + 1):
        for k in range(len(Dp_ext)):
            if j+k < num_vars: M[j+k, j] = Dp_ext[k]
            
    # 填充 Np 列
    off = deg_ctrl + 1
    for j in range(deg_ctrl + 1):
        for k in range(len(num)):
            if j+k < num_vars: M[j+k, off+j] = num[k]
            
    try:
        x = np.linalg.solve(M, b_vec)
    except np.linalg.LinAlgError:
        raise ValueError("设计失败：对象不可控")

    A_prime = x[:deg_ctrl+1].tolist()
    B_final = x[deg_ctrl+1:].tolist()
    A_final = PolynomialUtils.multiply(A_prime, s_term)
    
    return B_final, A_final, r_add, zeta, wn