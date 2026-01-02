import tkinter as tk
from tkinter import messagebox, scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

# 导入核心逻辑
from math_core import PolynomialUtils, RouthStability
from algorithms import design_controller
from simulator import CustomSimulator, PerformanceAnalyzer

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

class AutoControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SISO 自动控制系统设计平台 Pro v2.1 (Fix)")
        self.root.geometry("1300x850")
        self.root.minsize(1200, 800)
        
        # ========== 全局样式配置 ==========
        self.style = ttk.Style()
        self.style.configure('.', font=('微软雅黑', 9), padding=3)  
        self.style.configure('TButton', font=('微软雅黑', 9, 'bold'), padding=5)
        self.style.configure('Labelframe.Label', font=('微软雅黑', 10, 'bold'), 
                           foreground='#2c3e50', padding=5)
        
        # ========== 主布局 ==========
        self.main_container = ttk.Frame(root, padding=8)
        self.main_container.pack(fill=BOTH, expand=YES)
        
        # 左侧面板
        self.left_panel = ttk.Labelframe(self.main_container)
        self.left_panel.pack(side=LEFT, fill=BOTH, padx=(5, 10), pady=5, expand=False)
        
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=RIGHT, fill=BOTH, expand=YES, padx=5, pady=5)
        
        # 构建组件
        self.create_sidebar()
        self.create_plot_area()

    def create_sidebar(self):
        """左侧面板：极致紧凑参数区 + 最大化日志区"""
        # 标题栏
        title_frame = ttk.Frame(self.left_panel, padding=(5, 8))
        title_frame.pack(fill=X, pady=(0, 5))
        ttk.Label(
            title_frame, 
            text="⚡ SISO自动控制器设计平台 v2.1", 
            font=("微软雅黑", 14, "bold"), 
            foreground='#2980b9'
        ).pack(side=LEFT)

        # ========== 1. 被控对象模型 ==========
        group_plant = ttk.Labelframe(self.left_panel, text="🏭 被控对象模型", padding=8)
        group_plant.pack(fill=X, pady=(0, 6))
        self.entry_num = self.create_labeled_entry(group_plant, "分子系数[升幂]", "10", "例：0 1 1 → s²+s")
        self.entry_den = self.create_labeled_entry(group_plant, "分母系数[升幂]", "0 1 1", "例：1 2 3 → 3s²+2s+1")

        # ========== 2. 性能指标 ==========
        group_specs = ttk.Labelframe(self.left_panel, text="🎯 性能指标", padding=8)
        group_specs.pack(fill=X, pady=(0, 6))
        self.entry_mp = self.create_labeled_entry(group_specs, "超调量MP(%)", "10", "5-20%")
        self.entry_ts = self.create_labeled_entry(group_specs, "调节时间Ts(s)", "2", "系统稳态时间")

        # ========== 3. 仿真设置 ==========
        group_sim = ttk.Labelframe(self.left_panel, text="⚙️ 仿真设置", padding=8)
        group_sim.pack(fill=X, pady=(0, 6))
        self.var_input = tk.StringVar(value="step")
        input_frame = ttk.Frame(group_sim)
        input_frame.pack(fill=X)
        ttk.Radiobutton(input_frame, text="阶跃", variable=self.var_input, value="step").pack(side=LEFT, padx=5)
        ttk.Radiobutton(input_frame, text="斜坡", variable=self.var_input, value="ramp").pack(side=LEFT, padx=5)

        # ========== 4. 核心按钮 ==========
        btn_frame = ttk.Frame(self.left_panel, padding=3)
        btn_frame.pack(fill=X, pady=(0, 6))
        self.btn_run = ttk.Button(btn_frame, text="🚀 开始设计", command=self.run_design, bootstyle="success")
        self.btn_run.pack(fill=X, ipady=3)
        self.btn_run.bind("<Enter>", lambda e: self.btn_run.config(bootstyle="success,outline"))
        self.btn_run.bind("<Leave>", lambda e: self.btn_run.config(bootstyle="success"))

        # ========== 5. 控制器参数 ==========
        result_frame = ttk.Labelframe(self.left_panel, text="📊 控制器参数", padding=5)
        result_frame.pack(fill=X, pady=(0, 6))
        self.controller_info = ttk.Label(
            result_frame, text="设计完成后显示参数...", font=("Consolas", 8),
            justify=LEFT, wraplength=350
        )
        self.controller_info.pack(anchor=W, fill=X)

        # ========== 6. 日志输出区 ==========
        log_frame = ttk.Labelframe(self.left_panel, text="📝 设计日志", padding=8)
        log_frame.pack(fill=BOTH, expand=YES, pady=(5, 0))
        self.txt_log = scrolledtext.ScrolledText(
            log_frame, font=("Consolas", 9), wrap=tk.WORD,
            relief=tk.FLAT, bg="#f8f9fa", bd=0
        )
        self.txt_log.pack(fill=BOTH, expand=YES)

    def create_labeled_entry(self, parent, label_text, default_val, hint_text=""):
        container = ttk.Frame(parent)
        container.pack(fill=X, pady=(0, 4))
        ttk.Label(container, text=label_text, font=("微软雅黑", 9), foreground="#34495e").pack(anchor=W)
        
        entry = ttk.Entry(container, font=("微软雅黑", 9))
        entry.insert(0, default_val)
        entry.pack(fill=X, pady=(1, 0))
        entry.bind("<FocusIn>", lambda e: entry.config(bootstyle="primary"))
        entry.bind("<FocusOut>", lambda e: entry.config(bootstyle=""))
        
        if hint_text:
            ttk.Label(container, text=hint_text, font=("微软雅黑", 7), foreground="gray").pack(anchor=W)
        return entry

    def create_plot_area(self):
        plot_container = ttk.Labelframe(self.right_panel, text="📈 系统响应曲线", padding=10)
        plot_container.pack(fill=BOTH, expand=YES)
        
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor='#ffffff')
        self.ax = self.fig.add_subplot(111)
        self.setup_plot_style("等待设计结果...")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=YES)
        
        toolbar_frame = ttk.Frame(plot_container)
        toolbar_frame.pack(fill=X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def setup_plot_style(self, title):
        self.ax.clear()
        self.ax.set_title(title, fontsize=13, fontweight='bold', color='#2c3e50', pad=15)
        self.ax.grid(True, linestyle=':', alpha=0.7, color='#bdc3c7')
        self.ax.set_facecolor('#f8f9fa')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#95a5a6')
        self.ax.spines['bottom'].set_color('#95a5a6')
        self.ax.tick_params(axis='both', colors='#7f8c8d')

    def log(self, msg, level="info"):
        color_map = {"info":"#2c3e50", "success":"#27ae60", "warning":"#f39c12", "error":"#e74c3c"}
        self.txt_log.tag_config(level, foreground=color_map.get(level, "#2c3e50"))
        self.txt_log.insert(tk.END, f"> {msg}\n", level)
        self.txt_log.see(tk.END)

    def update_controller_info(self, Bc, Ac, r_added, zeta, wn):
        info = (
            f"Gc(s) = B(s)/A(s) | ζ={zeta:.3f} | ωn={wn:.2f}\n"
            f"积分补偿+{r_added}个 | 稳态误差=0\n"
            f"B(s)={PolynomialUtils.to_str(Bc)} | A(s)={PolynomialUtils.to_str(Ac)}"
        )
        self.controller_info.config(text=info)

    def run_design(self):
        self.txt_log.delete(1.0, tk.END)
        self.btn_run.configure(state=DISABLED, text="⏳ 计算中...")
        self.root.update()

        try:
            # 获取参数
            num = [float(x) for x in self.entry_num.get().replace(',',' ').split()]
            den = [float(x) for x in self.entry_den.get().replace(',',' ').split()]
            mp = float(self.entry_mp.get())
            ts = float(self.entry_ts.get())
            in_type = self.var_input.get()

            self.log("✅ 启动控制器自动化设计流程...", "info")
            self.log(f"被控对象 G(s) = {PolynomialUtils.to_str(num)} / {PolynomialUtils.to_str(den)}")

            # 控制器设计
            self.log("🔍 求解Diophantine方程，配置闭环极点...", "info")
            Bc, Ac, r_added, zeta, wn = design_controller(num, den, mp, ts, in_type)
            self.update_controller_info(Bc, Ac, r_added, zeta, wn)
            self.log(f"✅ 控制器设计完成！ζ={zeta:.3f}, ωn={wn:.2f}", "success")
            self.log(f"✅ 积分补偿{r_added}个，稳态误差归零", "success")

            # 稳定性校验
            T_num = PolynomialUtils.multiply(Bc, num)
            T_den = PolynomialUtils.add(PolynomialUtils.multiply(Ac, den), T_num)
            is_stable = RouthStability.check(T_den)
            status = "稳定" if is_stable else "不稳定"
            self.log(f"🔒 劳斯稳定性校验：{status}", "success" if is_stable else "warning")

            # RK4仿真 - 【FIX 1】动态计算步长
            self.log("⚙️ 启动RK4仿真引擎，计算响应曲线...", "info")
            
            # 安全检查：如果不稳定，提示但不强制退出仿真（方便看发散波形）
            if not is_stable:
                self.log("⚠️ 警告：闭环系统不稳定，仿真结果可能发散！", "warning")

            sim = CustomSimulator(T_num, T_den)
            
            # 动态步长策略：保证调节时间内至少有200个点，且dt不超过0.01s
            # 这解决了快速系统（Ts小）被欠采样的问题
            calc_dt = ts / 200.0
            dt = min(0.01, calc_dt)  
            t_end = ts * 4.0
            
            self.log(f"ℹ️ 仿真参数自动调优：dt={dt:.5f}s (Ts={ts}s)", "info")
            
            t_data = np.arange(0, t_end, dt)
            
            if in_type == 'ramp':
                y_data = np.array([sim.step(t, dt) for t in t_data])
                target_curve = t_data
                target_val = t_data[-1]
            else:
                y_data = np.array([sim.step(1.0, dt) for _ in t_data])
                target_curve = np.ones_like(t_data)
                target_val = 1.0

            # 性能指标计算
            analyzer = PerformanceAnalyzer(t_data, y_data, target_val)
            metrics = analyzer.get_metrics()
            if in_type == 'step':
                self.log(f"📊 超调量：{metrics['overshoot']:.2f}% | 调节时间：{metrics['ts']:.2f}s", "info")
                self.log(f"📊 稳态误差：{metrics['error']:.2e}", "info")

            # 绘图
            self.setup_plot_style(f"闭环系统响应曲线 (劳斯判据：{status})")
            self.ax.plot(t_data, target_curve, color='#e74c3c', linestyle='--', linewidth=1.5, label='参考输入')
            self.ax.plot(t_data, y_data, color='#3498db', linewidth=2.5, label='系统输出', alpha=0.9)
            
            if in_type == 'step':
                self.ax.fill_between(t_data, 0.98, 1.02, color='#2ecc71', alpha=0.1, label='2%误差带')
                info_text = f"超调量：{metrics['overshoot']:.1f}%\n调节时间：{metrics['ts']:.2f}s\n稳态误差：{metrics['error']:.1e}"
                # 调整文本位置，防止遮挡
                text_x = t_end * 0.5
                self.ax.text(text_x, 0.5, info_text, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bdc3c7", alpha=0.9), fontsize=9)
            
            self.ax.legend(loc='best', frameon=True, framealpha=0.8)
            self.ax.set_xlabel("时间 (s)", fontsize=10)
            self.ax.set_ylabel("幅值", fontsize=10)
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.log(f"❌ 运行错误：{str(e)}", "error")
            import traceback
            traceback.print_exc() # 打印堆栈到控制台方便调试
            messagebox.showerror("运算异常", f"程序执行出错：\n{str(e)}")
        finally:
            self.btn_run.configure(state=NORMAL, text="🚀 开始设计")

# ========== 程序入口 ==========
if __name__ == "__main__":
    root = ttk.Window(themename="flatly")
    app = AutoControlApp(root)
    root.mainloop()