import numpy as np
from typing import List, Union

class PolynomialUtils:
    """多项式基础运算工具 (升幂排列)"""
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
    def filter_small_coeffs(poly: List[float], eps: float = 1e-8) -> List[float]:
        return [0.0 if abs(c) < eps else c for c in poly]

    @staticmethod
    def to_str(poly: List[float], var: str = 's') -> str:
        terms = []
        for i, c in enumerate(poly):
            if abs(c) < 1e-6: continue
            coeff = f"{c:.4f}".rstrip('0').rstrip('.')
            if i == 0: terms.append(coeff)
            elif i == 1: terms.append(f"{coeff}{var}")
            else: terms.append(f"{coeff}{var}^{i}")
        return " + ".join(terms) if terms else "0"

class RouthStability:
    @staticmethod
    def check(coeff_asc: List[float]) -> bool:
        # 转换为降幂排列: a_n s^n + ... + a_0
        coeff = coeff_asc[::-1]
        n = len(coeff)
        cols = (n + 1) // 2
        R = np.zeros((n, cols))

        # 填充前两行
        R[0, :len(coeff[0::2])] = coeff[0::2]
        R[1, :len(coeff[1::2])] = coeff[1::2]

        for i in range(2, n):
            for j in range(cols - 1):
                a, b = R[i-2, 0], R[i-2, j+1]
                c, d = R[i-1, 0], R[i-1, j+1]
                if abs(c) < 1e-10: c = 1e-6 # 避免除零
                R[i, j] = (c * b - a * d) / c

        # 第一列元素全为正即稳定 (假设首项>0)
        return np.all(R[:, 0] > -1e-10)

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