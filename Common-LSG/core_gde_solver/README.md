# GDE-LSG 全域对偶演化求解器
## 模块定位
属于 Common-LSG 公共底层数学内核，基于四条本源公理与层级截面几何（LSG），实现全域对偶演化方程（GDE）的有限差分数值求解。
所有行业模块的截面重建、场演化、约束优化均可复用本模块的标准化微分算子与演化框架，无需重复开发底层几何计算。

## 理论溯源
1. 底层根基：GYY-GOES 四条本源公理
2. 核心方程：全域对偶演化方程（GDE）
3. 几何形式：LSG 多层截面矩阵形式
4. 数值方法：三维七点有限差分 + 稀疏矩阵 + Verlet 二阶时间积分

## 核心能力
- 标准化三维拉普拉斯、梯度稀疏矩阵算子
- LSG 多层级截面耦合算子
- 全域正演演化求解
- 支持自定义边界条件、截面范围、物理常数配置

## 调用示例
```python
from core_gde_solver.operators import build_3d_laplacian, build_lambda_coupling
from core_gde_solver.gde_lsg_solver import gde_rhs, verlet_evolve

# 构造单截面算子
L = build_3d_laplacian(nx=32, ny=32, nz=32, dx=1.0)
# 构造多层耦合
Lambda = build_lambda_coupling(n_layers=8, n_spatial=32**3, dlambda=1.0)
# 执行演化
trajectory = verlet_evolve(Phi0, Phi1, dt=0.02, steps=200)
依赖
numpy >= 1.24
scipy >= 1.10.0
向后兼容说明
本模块为新增可选底层能力，不修改任何现有行业模块代码，原有重建算法可逐步迁移，也可保留独立实现。