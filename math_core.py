import numpy as np
from typing import List, Union

class PolynomialUtils:
    """多项式基础运算工具"""
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
    def derivative(poly: List[float]) -> List[float]:
        """多项式求导"""
        n = len(poly)
        if n <= 1: return [0.0]
        res = []
        for i in range(1, n):
            res.append(poly[i] * i)
        return res

    @staticmethod
    def filter_small_coeffs(poly: List[float], eps: float = 1e-9) -> List[float]:
        return [0.0 if abs(c) < eps else c for c in poly]

    @staticmethod
    def to_str(poly: List[float], var: str = 's') -> str:
        """美化版多项式转字符串"""
        superscript_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
        }
        
        def to_super(n):
            return "".join(superscript_map.get(c, c) for c in str(n))

        n = len(poly)
        terms = []
        
        for i in range(n - 1, -1, -1):
            c = poly[i]
            if abs(c) < 1e-6: continue 
            
            is_negative = c < 0
            abs_c = abs(c)
            
            val_str = f"{abs_c:.4f}".rstrip('0').rstrip('.')
            if val_str == '': val_str = '0'
            
            if i == 0: coeff_str = val_str
            else:
                if abs(abs_c - 1.0) < 1e-6: coeff_str = ""
                else: coeff_str = val_str
            
            if i == 0: var_str = ""
            elif i == 1: var_str = var
            else: var_str = f"{var}{to_super(i)}"
                
            term_str = f"{coeff_str}{var_str}"
            terms.append((is_negative, term_str))
            
        if not terms: return "0"
        
        res = []
        for idx, (is_neg, s) in enumerate(terms):
            if idx == 0: prefix = "-" if is_neg else ""
            else: prefix = " - " if is_neg else " + "
            res.append(f"{prefix}{s}")
            
        return "".join(res)

class RouthStability:
    @staticmethod
    def check(coeff_asc: List[float]) -> bool:
        """劳斯判据稳定性检查 (完整版)"""
        coeff = coeff_asc[::-1]
        while len(coeff) > 0 and abs(coeff[0]) < 1e-9: coeff.pop(0)
        if not coeff: return True
            
        n = len(coeff)
        cols = (n + 1) // 2
        R = np.zeros((n, cols))

        R[0, :len(coeff[0::2])] = coeff[0::2]
        R[1, :len(coeff[1::2])] = coeff[1::2]
        EPS = 1e-9

        for i in range(2, n):
            if np.all(np.abs(R[i-1, :]) < EPS): # 全零行处理
                for j in range(cols):
                    power = (n - 1 - (i - 2)) - 2 * j 
                    if power < 0: continue
                    R[i-1, j] = R[i-2, j] * power 

            if abs(R[i-1, 0]) < EPS: R[i-1, 0] = 1e-6 

            for j in range(cols - 1):
                a, b = R[i-2, 0], R[i-2, j+1]
                c, d = R[i-1, 0], R[i-1, j+1]
                R[i, j] = (c * b - a * d) / c

        first_col = R[:, 0]
        if np.any(np.isnan(first_col)): return False
        return np.all(first_col > -EPS)

class PoleUtils:
    @staticmethod
    def conjugate_pair(poles: List[Union[float, complex]]) -> List[Union[float, complex]]:
        paired = []
        for p in poles:
            if isinstance(p, complex) and abs(p.imag) > 1e-6:
                paired.append(p)
                paired.append(p.conjugate())
            else:
                paired.append(p.real)
        return paired