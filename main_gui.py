import tkinter as tk
from tkinter import messagebox, scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from math_core import PolynomialUtils, RouthStability
from algorithms import design_controller
from simulator import CustomSimulator, PerformanceAnalyzer

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

class AutoControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SISO 自动控制系统设计平台 Pro v3.8 (Thesis Edition)") 
        self.root.geometry("1300x900")
        self.root.minsize(1200, 800)
        
        self.style = ttk.Style()
        self.style.configure('.', font=('微软雅黑', 9), padding=3)  
        self.style.configure('TButton', font=('微软雅黑', 9, 'bold'), padding=5)
        self.style.configure('Labelframe.Label', font=('微软雅黑', 10, 'bold'), foreground='#2c3e50', padding=5)
        
        self.main_container = ttk.Frame(root, padding=8)
        self.main_container.pack(fill=BOTH, expand=YES)
        
        self.left_panel = ttk.Labelframe(self.main_container)
        self.left_panel.pack(side=LEFT, fill=BOTH, padx=(5, 10), pady=5, expand=False)
        
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=RIGHT, fill=BOTH, expand=YES, padx=5, pady=5)
        
        self.create_sidebar()
        self.create_plot_area()

    def create_sidebar(self):
        title_frame = ttk.Frame(self.left_panel, padding=(5, 8))
        title_frame.pack(fill=X, pady=(0, 5))
        ttk.Label(title_frame, text="⚡ SISO设计平台 v3.8", font=("微软雅黑", 14, "bold"), foreground='#2980b9').pack(side=LEFT)

        # 1. 被控对象
        group_plant = ttk.Labelframe(self.left_panel, text="🏭 被控对象模型", padding=8)
        group_plant.pack(fill=X, pady=(0, 6))
        self.entry_num = self.create_labeled_entry(group_plant, "分子系数[升幂]", "10", "例：0 1 1 → s²+s")
        self.entry_den = self.create_labeled_entry(group_plant, "分母系数[升幂]", "0 1 1", "例：1 2 3 → 3s²+2s+1")

        # 2. 性能指标
        group_specs = ttk.Labelframe(self.left_panel, text="🎯 性能指标", padding=8)
        group_specs.pack(fill=X, pady=(0, 6))
        self.entry_mp = self.create_labeled_entry(group_specs, "超调量MP(%)", "10", "5-20%")
        self.entry_ts = self.create_labeled_entry(group_specs, "调节时间Ts(s)", "2", "系统稳态时间")

        # 3. 仿真设置
        group_sim = ttk.Labelframe(self.left_panel, text="⚙️ 仿真设置", padding=8)
        group_sim.pack(fill=X, pady=(0, 6))
        self.var_input = tk.StringVar(value="step")
        input_frame = ttk.Frame(group_sim)
        input_frame.pack(fill=X)
        ttk.Radiobutton(input_frame, text="阶跃", variable=self.var_input, value="step").pack(side=LEFT, padx=5)
        ttk.Radiobutton(input_frame, text="斜坡", variable=self.var_input, value="ramp").pack(side=LEFT, padx=5)
        self.entry_ulim = self.create_labeled_entry(group_sim, "控制量限幅", "1000", "执行器最大输出")

        # 4. 按钮
        btn_frame = ttk.Frame(self.left_panel, padding=3)
        btn_frame.pack(fill=X, pady=(0, 6))
        self.btn_run = ttk.Button(btn_frame, text="🚀 开始设计", command=self.run_design, bootstyle="success")
        self.btn_run.pack(fill=X, ipady=3)

        # 5. 参数显示
        result_frame = ttk.Labelframe(self.left_panel, text="📊 控制器参数", padding=5)
        result_frame.pack(fill=X, pady=(0, 6))
        self.controller_info = ttk.Label(result_frame, text="...", font=("Consolas", 10), justify=LEFT, wraplength=1000)
        self.controller_info.pack(anchor=W, fill=X)

        # 6. 日志
        log_frame = ttk.Labelframe(self.left_panel, text="📝 设计日志", padding=8)
        log_frame.pack(fill=BOTH, expand=YES, pady=(5, 0))
        self.txt_log = scrolledtext.ScrolledText(log_frame, font=("Consolas", 9), wrap=tk.WORD, relief=tk.FLAT, bg="#f8f9fa", bd=0)
        self.txt_log.pack(fill=BOTH, expand=YES)

    def create_labeled_entry(self, parent, label_text, default_val, hint_text=""):
        container = ttk.Frame(parent)
        container.pack(fill=X, pady=(0, 4))
        ttk.Label(container, text=label_text, font=("微软雅黑", 9), foreground="#34495e").pack(anchor=W)
        entry = ttk.Entry(container, font=("微软雅黑", 9))
        entry.insert(0, default_val)
        entry.pack(fill=X, pady=(1, 0))
        if hint_text: ttk.Label(container, text=hint_text, font=("微软雅黑", 7), foreground="gray").pack(anchor=W)
        return entry

    def create_plot_area(self):
        plot_container = ttk.Labelframe(self.right_panel, text="📈 系统响应与控制量", padding=10)
        plot_container.pack(fill=BOTH, expand=YES)
        
        self.fig = Figure(figsize=(7, 6), dpi=100, facecolor='#ffffff')
        self.ax1 = self.fig.add_subplot(211) 
        self.ax2 = self.fig.add_subplot(212) 
        self.fig.subplots_adjust(hspace=0.3) 

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=YES)
        
        toolbar_frame = ttk.Frame(plot_container)
        toolbar_frame.pack(fill=X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def setup_plot_style(self, title, ax):
        ax.clear()
        ax.set_title(title, fontsize=11, fontweight='bold', color='#2c3e50')
        ax.grid(True, linestyle=':', alpha=0.7, color='#bdc3c7')
        ax.set_facecolor('#f8f9fa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def log(self, msg, level="info"):
        color_map = {"info":"#2c3e50", "success":"#27ae60", "warning":"#f39c12", "error":"#e74c3c"}
        self.txt_log.tag_config(level, foreground=color_map.get(level, "#2c3e50"))
        self.txt_log.insert(tk.END, f"{msg}\n", level)
        self.txt_log.see(tk.END)

    def log_transfer_function(self, name, num, den):
        """在日志中打印漂亮的分数形式传递函数"""
        s_num = PolynomialUtils.to_str(num)
        s_den = PolynomialUtils.to_str(den)
        len_num = len(s_num)
        len_den = len(s_den)
        width = max(len_num, len_den) + 4
        
        divider = "-" * width
        fmt_num = s_num.center(width)
        fmt_den = s_den.center(width)
        
        self.log(f"💠 {name}:")
        self.log(f"{fmt_num}")
        self.log(f"{divider}")
        self.log(f"{fmt_den}\n")

    def update_controller_info(self, Bc, Ac, r_added, zeta, wn):
        info = (
            f"Gc(s) = B(s)/A(s) | ζ={zeta:.3f} | ωn={wn:.2f}\n"
            f"积分补偿+{r_added}个\n"
            f"B(s)={PolynomialUtils.to_str(Bc)} | A(s)={PolynomialUtils.to_str(Ac)}"
        )
        self.controller_info.config(text=info)

    def run_design(self):
        self.txt_log.delete(1.0, tk.END)
        self.btn_run.configure(state=DISABLED, text="⏳ 计算中...")
        self.root.update()

        try:
            num = [float(x) for x in self.entry_num.get().replace(',',' ').split()]
            den = [float(x) for x in self.entry_den.get().replace(',',' ').split()]
            mp = float(self.entry_mp.get())
            ts = float(self.entry_ts.get())
            ulim = float(self.entry_ulim.get())
            in_type = self.var_input.get()

            self.log(f"✅ 对象: {PolynomialUtils.to_str(num)} / {PolynomialUtils.to_str(den)}")

            # 1. 设计控制器
            Bc, Ac, r_added, zeta, wn, desired_poly = design_controller(num, den, mp, ts, in_type)
            self.update_controller_info(Bc, Ac, r_added, zeta, wn)
            self.log(f"> 设计目标：ζ={zeta:.3f}, ωn={wn:.2f}", "success")

            # 2. 丢番图方程验证 (LHS vs RHS)
            self.log("-" * 55)
            self.log("🔍 验证环节：丢番图方程求解 (LHS vs RHS)")
            
            # 计算实际闭环多项式
            LHS_part1 = PolynomialUtils.multiply(den, Ac)
            LHS_part2 = PolynomialUtils.multiply(num, Bc)
            actual_poly = PolynomialUtils.add(LHS_part1, LHS_part2)
            
            # 对齐长度以便打印
            len_max = max(len(actual_poly), len(desired_poly))
            act_pad = [0.0]*(len_max - len(actual_poly)) + actual_poly
            des_pad = [0.0]*(len_max - len(desired_poly)) + desired_poly
            
            header = f"{'阶次':<6} {'实际系数(LHS)':<15} {'期望系数(RHS)':<15} {'误差':<12}"
            self.log(header)
            self.log("-" * 55)
            
            # 倒序打印
            for i in range(len_max - 1, -1, -1):
                idx = len_max - 1 - i
                val_act = act_pad[idx]
                val_des = des_pad[idx]
                err = abs(val_act - val_des)
                if abs(val_act) > 1e-9 or abs(val_des) > 1e-9:
                    row_str = f"s^{i:<5} {val_act:<15.5f} {val_des:<15.5f} {err:<12.1e}"
                    self.log(row_str)
            self.log("-" * 55)

            # 3. 打印传递函数
            self.log("🧮 系统传递函数形式:")
            self.log_transfer_function("控制器 C(s)", Bc, Ac)
            
            # 闭环传递函数 T(s) = (Num*Bc) / Characteristic_Poly
            CL_num = PolynomialUtils.multiply(num, Bc)
            CL_den = actual_poly 
            self.log_transfer_function("闭环系统 T(s)", CL_num, CL_den)
            self.log("-" * 55)

            # 4. 稳定性校验
            is_stable = RouthStability.check(actual_poly)
            status = "稳定" if is_stable else "不稳定"
            self.log(f"🔒 劳斯稳定性检查：{status}", "success" if is_stable else "warning")
            if not is_stable: self.log("⚠️ 警告：闭环理论不稳定！", "warning")

            # 5. 时域仿真
            sim_ctrl = CustomSimulator(Bc, Ac)
            sim_plant = CustomSimulator(num, den)

            calc_dt = ts / 200.0
            dt = max(1e-5, min(0.01, calc_dt))
            t_end = ts * 4.0
            t_data = np.arange(0, t_end, dt)
            
            y_list = []
            u_list = []
            y_curr = sim_plant.compute_output(0.0)
            
            self.log(f"⚙️ 启动抗饱和高精度仿真 (dt={dt:.1e}s)...", "info")
            
            for t in t_data:
                r_val = t if in_type == 'ramp' else 1.0
                error = r_val - y_curr
                
                u_raw = sim_ctrl.compute_output(error)
                
                in_saturation = False
                if u_raw > ulim: 
                    u_act = ulim
                    in_saturation = True
                elif u_raw < -ulim: 
                    u_act = -ulim
                    in_saturation = True
                else: 
                    u_act = u_raw
                
                y_list.append(y_curr)
                u_list.append(u_act)
                
                ctrl_input = error
                if in_saturation:
                    if (u_act > 0 and error > 0) or (u_act < 0 and error < 0):
                        ctrl_input = 0.0

                sim_ctrl.update_state(ctrl_input, dt)
                sim_plant.update_state(u_act, dt)
                y_curr = sim_plant.compute_output(u_act)

            y_data = np.array(y_list)
            u_data = np.array(u_list)
            
            if in_type == 'ramp':
                target_curve = t_data
                target_val = t_data[-1]
            else:
                target_curve = np.ones_like(t_data)
                target_val = 1.0

            # 绘图
            self.setup_plot_style("系统响应 y(t)", self.ax1)
            self.ax1.plot(t_data, target_curve, 'r--', label='参考输入')
            self.ax1.plot(t_data, y_data, 'b', linewidth=2, label='系统输出')
            self.ax1.legend(prop={'size': 9})
            
            self.setup_plot_style("控制量 u(t) [含抗饱和]", self.ax2)
            self.ax2.plot(t_data, u_data, 'g', linewidth=1.5, label='控制量')
            self.ax2.axhline(ulim, color='k', linestyle=':', alpha=0.3, label='限幅值')
            self.ax2.axhline(-ulim, color='k', linestyle=':', alpha=0.3)
            self.ax2.legend(prop={'size': 9})

            # 性能指标与可视化标注
            analyzer = PerformanceAnalyzer(t_data, y_data, target_val)
            metrics = analyzer.get_metrics()
            
            if in_type == 'step':
                self.log(f"📊 仿真结果: MP={metrics['overshoot']:.2f}% | Ts={metrics['ts']:.2f}s | Tp={metrics['tp']:.2f}s")
                
                # --- 绘制辅助线和标注 ---
                # 1. 峰值时间 Tp (绿色虚线 + 红点)
                tp = metrics['tp']
                peak_val = y_data[np.argmax(y_data)]
                self.ax1.axvline(x=tp, color='green', linestyle='--', alpha=0.6, linewidth=1)
                self.ax1.plot(tp, peak_val, 'ro', markersize=4)
                self.ax1.text(tp, peak_val*1.02, f"Tp:{tp:.2f}s", color='green', fontsize=8, ha='center')

                # 2. 调节时间 Ts (品红虚线)
                ts = metrics['ts']
                if ts > 0:
                    self.ax1.axvline(x=ts, color='magenta', linestyle='--', alpha=0.6, linewidth=1)
                    self.ax1.text(ts, target_val*0.9, f"Ts:{ts:.2f}s", color='magenta', fontsize=8, ha='right')

                info = f"OS: {metrics['overshoot']:.1f}%\nTs: {metrics['ts']:.2f}s"
                self.ax1.text(t_end*0.75, target_val*0.2, info, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

            self.canvas.draw()

        except Exception as e:
            self.log(f"❌ 错误：{str(e)}", "error")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_run.configure(state=NORMAL, text="🚀 开始设计")

if __name__ == "__main__":
    root = ttk.Window(themename="flatly")
    app = AutoControlApp(root)
    root.mainloop()