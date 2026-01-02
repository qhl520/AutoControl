import numpy as np
from typing import List, Union

class PolynomialUtils:
    """多项式基础运算工具 (升幂排列: a0 + a1*s + a2*s^2 ...)"""
    @staticmethod
    def multiply(p1: List[float], p2: List[float]) -> List[float]:
        n = len(p1) + len(p2) - 1
        res = [0.0] * n
        for i in range(len(p1)):
            for j in range(len(p2)):
                res[i + j] += p1[i] * p2[j]
        return PolynomialUtils.filter_small_coeffs(res)

    @staticmethod
    def add(p1: List[float], p2: List[float]) -> List[float]:
        n = max(len(p1), len(p2))
        res = [0.0] * n
        for i in range(len(p1)): res[i] += p1[i]
        for i in range(len(p2)): res[i] += p2[i]
        return PolynomialUtils.filter_small_coeffs(res)

    @staticmethod
    def filter_small_coeffs(poly: List[float], eps: float = 1e-9) -> List[float]:
        return [0.0 if abs(c) < eps else c for c in poly]

    @staticmethod
    def to_str(poly: List[float], var: str = 's') -> str:
        terms = []
        for i, c in enumerate(poly):
            if abs(c) < 1e-6: continue
            coeff = f"{c:.4f}".rstrip('0').rstrip('.')
            if coeff.endswith('.'): coeff += '0' # Fix "12." case
            
            if i == 0: terms.append(coeff)
            elif i == 1: terms.append(f"{coeff}{var}")
            else: terms.append(f"{coeff}{var}^{i}")
        return " + ".join(terms) if terms else "0"

class RouthStability:
    @staticmethod
    def check(coeff_asc: List[float]) -> bool:
        """
        劳斯判据稳定性检查
        输入: 升幂排列的系数 [a0, a1, a2...] -> a0 + a1*s + ...
        """
        # 转换为降幂排列: a_n s^n + ... + a_0 (标准Routh表习惯)
        coeff = coeff_asc[::-1]
        
        # 去除高阶零系数
        while len(coeff) > 0 and abs(coeff[0]) < 1e-9:
            coeff.pop(0)
            
        if not coeff: return True # 空多项式视为稳定? 或者抛错
            
        n = len(coeff)
        cols = (n + 1) // 2
        R = np.zeros((n, cols))

        # 填充前两行
        R[0, :len(coeff[0::2])] = coeff[0::2]
        R[1, :len(coeff[1::2])] = coeff[1::2]
        
        # 【FIX 4】统一 Epsilon 阈值
        EPS = 1e-9

        for i in range(2, n):
            # 检查上一行首元素是否为0
            if abs(R[i-1, 0]) < EPS:
                 # TODO: 完整处理全零行需要对辅助方程求导
                 # 这里仅做简单的 Epsilon 替换以避免除零崩溃
                 # 这对于工程作业通常足够，但对于临界稳定/共轭虚根情况仍是不完全的
                 R[i-1, 0] = 1e-6 

            for j in range(cols - 1):
                a, b = R[i-2, 0], R[i-2, j+1]
                c, d = R[i-1, 0], R[i-1, j+1]
                R[i, j] = (c * b - a * d) / c

        # 判据：第一列元素符号相同（即全为正，假设首项>0）
        # 且不能出现NaN
        first_col = R[:, 0]
        if np.any(np.isnan(first_col)): return False
        
        return np.all(first_col > -EPS)

class PoleUtils:
    @staticmethod
    def conjugate_pair(poles: List[Union[float, complex]]) -> List[Union[float, complex]]:
        """自动配对共轭极点，防止数学错误"""
        paired = []
        for p in poles:
            if isinstance(p, complex) and abs(p.imag) > 1e-6:
                paired.append(p)
                paired.append(p.conjugate())
            else:
                paired.append(p.real)
        return paired