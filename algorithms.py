import numpy as np
import math
from math_core import PolynomialUtils, PoleUtils

def count_integrators(den: list) -> int:
    cnt = 0
    for c in den:
        if abs(c) < 1e-6: cnt += 1
        else: break
    return cnt

def design_controller(num, den, mp, ts, input_type='step'):
    """Diophantine 方程求解器 (修正版：极点分散与鲁棒求解)"""
    ln_mp = math.log(mp/100) if mp > 0 else -999
    zeta = -ln_mp / math.sqrt(math.pi**2 + ln_mp**2)
    wn = 4.0 / (zeta * ts)
    
    p_real = -zeta * wn
    p_imag = wn * math.sqrt(1 - zeta**2)
    desired_pole = complex(p_real, p_imag)
    
    req_type = 2 if input_type == 'ramp' else 1
    cur_type = count_integrators(den)
    r_add = max(0, req_type - cur_type)
    
    s_term = [1.0]
    for _ in range(r_add):
        s_term = PolynomialUtils.multiply(s_term, [0.0, 1.0])
    Dp_ext = PolynomialUtils.multiply(den, s_term)
    
    n_ext = len(Dp_ext) - 1
    deg_ctrl = n_ext - 1
    total_order = n_ext + deg_ctrl
    
    complex_poles = PoleUtils.conjugate_pair([desired_pole])
    A_cl = [1.0]
    for p in complex_poles:
        A_cl = PolynomialUtils.multiply(A_cl, [-p, 1.0])
    A_cl = [c.real for c in A_cl]

    dom_real_abs = abs(zeta * wn)
    far_pole_base = max(10.0, 10.0 * dom_real_abs)
    
    # 修正：分散远极点，避免重根导致的数值奇异
    idx = 0
    while len(A_cl) - 1 < total_order:
        # 每个新增极点位置偏移 10%，防止形成Jordan块
        curr_pole = far_pole_base * (1.0 + idx * 0.1)
        A_cl = PolynomialUtils.multiply(A_cl, [curr_pole, 1.0])
        idx += 1
        
    num_vars = (deg_ctrl + 1) * 2
    M = np.zeros((num_vars, num_vars))
    b_vec = np.zeros(num_vars)
    
    for i in range(min(len(A_cl), num_vars)): b_vec[i] = A_cl[i]
    
    for j in range(deg_ctrl + 1):
        for k in range(len(Dp_ext)):
            if j+k < num_vars: M[j+k, j] = Dp_ext[k]
            
    off = deg_ctrl + 1
    for j in range(deg_ctrl + 1):
        for k in range(len(num)):
            if j+k < num_vars: M[j+k, off+j] = num[k]
            
    try:
        x = np.linalg.solve(M, b_vec)
    except np.linalg.LinAlgError:
        # 修正：更具体的错误提示
        raise ValueError("设计失败：Sylvester矩阵奇异。\n原因可能是：\n1. 被控对象存在零极点对消（如 s/(s^2+s)）\n2. 系统不可控或不可观")

    A_prime = x[:deg_ctrl+1].tolist()
    B_final = x[deg_ctrl+1:].tolist()
    A_final = PolynomialUtils.multiply(A_prime, s_term)
    
    return B_final, A_final, r_add, zeta, wn