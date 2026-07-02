import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ====================== 顶层可调参数区 仅修改此处切换场景与变量 ======================
SCENE = "dynasty"  # 可选：star / dynasty / capital / company
# 基础三核心参数 DR-TOE统一变量
E0 = 100.0       # 初始核心势能/等效核心质量
P0 = 20.0        # 初始外层压力/外层物质总量
k_decay = 0.00188# 梯度耗散系数 单位/年

# 开放吸积参数（外部补给，0代表完全封闭系统）
ac_rate = 0.000  # 周期性吸积强度（变法、技术、并购、星际物质）
ac_cycle = 50    # 吸积重置周期

# 外部扰动参数（天灾、战乱、行业危机，0无冲击）
sigma_noise = 0.0
disturb_year = 120   # 扰动发生年份
disturb_E_loss = 30  # 扰动损失核心势能数值

# 仿真总时长上限
sim_max_year = 600
# ==============================================================================

# 系统守恒总维度量 全局恒定不变
U_total = E0 + P0

# 通用耗散微分方程组
def dyn_eq(t, state, k, ac, ac_cyc):
    E, P = state
    # 周期性外部吸积补给项
    cycle_term = ac * np.sin(2 * np.pi * t / ac_cycle) if ac > 0 else 0
    dE_dt = -k * E + cycle_term
    dP_dt = k * E - cycle_term
    return [dE_dt, dP_dt]

# 六大演化终局自动判定函数
def judge_end_type(E_final, P_final, E_peak, ac):
    core_ratio = E_final / (E_final + 1e-8)
    M_eq = E_peak
    end_type = ""
    # 基础质量分层判定
    if M_eq < 30:
        end_type = "MODEL1 包层剥离·白矮星稳态型"
    elif 30 <= M_eq < 80:
        end_type = "MODEL2 爆发坍缩·中子星重构型"
    elif M_eq >= 80:
        end_type = "MODEL3 极致坍缩·黑洞吞噬型"
    # 极低核心，系统彻底溃散
    if E_peak < 15 and P_final > E_final * 1.8:
        end_type = "MODEL4 彻底崩解·无核消散型"
    # 存在持续外部吸积，判定为重生周期系统
    if ac > 0.0001:
        end_type = "MODEL5 吸积重生·周期续命型"
    # 合并跃升模型预留接口（后续扩展）
    return end_type

# 数值求解演化全过程
t_span = (0, sim_max_year)
init_state = [E0, P0]
sol = solve_ivp(fun=lambda t,s: dyn_eq(t,s,k_decay,ac_rate,ac_cycle),
                t_span=t_span,
                y0=init_state,
                t_eval=np.linspace(0, sim_max_year, 2000),
                max_step=1)

t_arr = sol.t
E_arr = sol.y[0]
P_arr = sol.y[1]

# 查找崩溃临界点 P >= E 年份
collapse_mask = P_arr >= E_arr
collapse_idx = np.where(collapse_mask)[0]
collapse_year = t_arr[collapse_idx[0]] if len(collapse_idx) > 0 else sim_max_year
E_peak = np.max(E_arr)
end_label = judge_end_type(E_arr[-1], P_arr[-1], E_peak, ac_rate)

# 改革延寿测算函数：降低耗散系数，计算系统延长寿命
def calc_reform_extend(new_k):
    sol_reform = solve_ivp(lambda t,s: dyn_eq(t,s,new_k,ac_rate,ac_cycle),
                           t_span=(0,sim_max_year), y0=[E0], t_eval=t_arr)
    Er = sol_reform.y[0]
    Pr = U_total - Er
    cr = np.where(Pr >= Er)[0]
    yr = t_arr[cr[0]] if len(cr) > 0 else sim_max_year
    return yr - collapse

# 示例：改革使耗散系数下降10%，测算延寿年限
reform_ext_year = calc_reform_extend(k_decay * 0.9)

# 绘图输出演化曲线
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(12,6))
plt.plot(t_arr, E_arr, label="核心势能 E(t)", color="#c0392b", linewidth=2)
plt.plot(t_arr, P_arr, label="外层压力 P(t)", color="#2980b9", linewidth=2)
plt.axvline(x=collapse_year, linestyle="--", color="#27ae6", label=f"崩溃临界年份:{collapse_year:.1f}")
plt.grid(alpha=0.3)
plt.xlabel("时间 / 年")
plt.ylabel("势能-压力 无量纲数值")
plt.title(f"DR-TOE跨尺度层级耗散演化仿真 | 场景:{SCENE} | 终局分类:{end_label}")
plt.legend()
plt.show()

# 控制台打印完整定量分析结果
print("="*72)
print("【DR-TOE 跨尺度耗散动力学仿真结果输出】")
print(f"仿真场景：{SCENE}")
print(f"系统演化终局分类：{end_label}")
print(f"系统完整临界寿命：{collapse_year:.2f} 年")
print(f"系统峰值核心强度 E_max = {E_peak:.2f}")
print(f"政策测算：耗散系数降低10%，可延长系统存续 {reform_ext_year:.2f} 年")
print("="*72)
print("参数操作说明：仅修改代码顶部参数区即可切换场景、调整变量完成预测分析")
print("场景参数对应对照表：")
print("star      | E0=恒星质量 k=核燃料消耗 ac=星际物质吸积")
print("dynasty   | E0=开国集权 k=土地兼并/腐化 ac=变法开荒中兴")
print("capital   | E0=资本集中度 k=贫富分化 ac=新产业现金流")
print("company   | E0=核心竞争力 k=组织内耗 ac=并购第二曲线")
print("="*72)
