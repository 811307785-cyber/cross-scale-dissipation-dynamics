"""
LSG气道重建【全兼容终版】
修复：
1. cmap="orange" 非法色表报错，替换为标准内置 cmap
2. 气管外壁生成逻辑反转，外层表皮裸露，外部旋转可见气管整体
3. 渲染层级置顶外壁，不会被内部气道遮挡
4. 无scipy ball依赖、无高版本专属参数、无语法错误
"""
import os
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
import pyvista as pv
from scipy.ndimage import binary_dilation

# ===================== 全局配置参数 =====================
DICOM_FOLDER = "./dataset/ct_scan/"
SAVE_PREFIX = "airway_final_all_fixed"
USE_GPU = True
TISSUE_GRAY_THRESHOLD = 8.0
MAX_EMERGE_ITER = 30
AIR_HU = -950
SEED_HU_MIN = -100
LUNG_TISSUE_HU = -600
WW_LUNG = 1800
WL_LUNG = -600
HU_DISP_MIN = AIR_HU
HU_DISP_MAX = WL_LUNG + WW_LUNG / 2
BACKGROUND_COLOR = "black"
# =======================================================

# 设备初始化
if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ CUDA 启用：{torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(f"⚠️ 运行于CPU")

# 手动生成球形膨胀核，不依赖scipy ball
def create_sphere_kernel(radius):
    size = radius * 2 + 1
    xx, yy, zz = np.meshgrid(np.arange(size), np.arange(size), np.arange(size))
    center = radius
    dist = np.sqrt((xx - center)**2 + (yy - center)**2 + (zz - center)**2)
    kernel = (dist <= radius).astype(np.uint8)
    return kernel

# 1. 加载DICOM，构建全局固定三维坐标
def load_ct_with_coord(folder):
    slices = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(folder, fname))
            slices.append(ds)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    pix_spacing = slices[0].PixelSpacing
    slice_thick = abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    D = len(slices)
    H, W = slices[0].pixel_array.shape

    z_axis = torch.linspace(0, D-1, D, device=device).reshape(D,1,1).repeat(1,H,W)
    y_axis = torch.linspace(0, H-1, H, device=device).reshape(1,H,1).repeat(D,1,W)
    x_axis = torch.linspace(0, W-1, W, device=device).reshape(1,1,W).repeat(D,H,1)
    coord_3d = torch.stack([z_axis, y_axis, x_axis], dim=-1)

    raw_np = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
    slope = getattr(slices[0], "RescaleSlope", 1.0)
    intercept = getattr(slices[0], "RescaleIntercept", 0.0)
    hu_full = raw_np * slope + intercept
    hu_tensor = torch.from_numpy(hu_full).to(device)

    print(f"三维尺寸 D={D} H={H} W={W} | HU范围：{hu_tensor.min():.1f} ~ {hu_tensor.max():.1f}")
    return coord_3d, hu_tensor, pix_spacing, slice_thick

# 2. 过滤空气空白区域
def get_valid_voxel_mask(hu_vol):
    valid_mask = hu_vol > AIR_HU
    air_ratio = (~valid_mask).float().mean() * 100
    print(f"空气占比 {air_ratio:.2f}%，渲染自动完全透明")
    return valid_mask

# 3. 提取气管、结节高密度种子
def get_seed_points(hu_vol, valid_mask):
    seed_mask = (hu_vol > SEED_HU_MIN) & valid_mask
    seed_cnt = seed_mask.sum().item()
    print(f"提取气道种子体素：{seed_cnt}")
    return seed_mask

# 4. 安全六邻域生长
def safe_neighbor_grow(volume, current_grow_mask, diff_thresh):
    D, H, W = volume.shape
    new_grow = torch.zeros_like(current_grow_mask, dtype=torch.bool, device=device)
    dirs = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
    for dz, dy, dx in dirs:
        if dz == -1:
            czs,cze = 1,D
            nzs,nze = 0,D-1
        elif dz == 1:
            czs,cze = 0,D-1
            nzs,nze = 1,D
        else:
            czs,cze = 0,D
            nzs,nze = 0,D
        if dy == -1:
            cys,cye = 1,H
            nys,nye = 0,H-1
        elif dy == 1:
            cys,cye = 0,H-1
            nys,nye = 1,H
        else:
            cys,cye = 0,H
            nys,nye = 0,H
        if dx == -1:
            cxs,cxe = 1,W
            nxs,nxe = 0,W-1
        elif dx == 1:
            cxs,cxe = 0,W-1
            nxs,nxe = 1,W
        else:
            cxs,cxe = 0,W
            nxs,nxe = 0,W

        curr_hu = volume[czs:cze, cys:cye, cxs:cxe]
        nei_hu = volume[nzs:nze, nys:nye, nxs:nxe]
        curr_conn = current_grow_mask[czs:cze, cys:cye, cxs:cxe]
        gray_diff = torch.abs(curr_hu - nei_hu)
        merge_ok = curr_conn & (gray_diff < diff_thresh)
        new_grow[nzs:nze, nys:nye, nxs:nxe] |= merge_ok
    return new_grow

# 5. 双向全域种子浸润
def seed_infusion_emerge(hu_vol, valid_mask, seed_mask):
    D, H, W = hu_vol.shape
    grow_mask = seed_mask.clone()
    vol = hu_vol.float()
    print(f"启动浸润生长，最大{MAX_EMERGE_ITER}轮")
    for step in range(MAX_EMERGE_ITER):
        old_grow = grow_mask.clone()
        new_connect = safe_neighbor_grow(vol, grow_mask, TISSUE_GRAY_THRESHOLD)
        new_connect = new_connect & valid_mask
        grow_mask |= new_connect
        changed = (grow_mask != old_grow).sum().item()
        print(f"  第{step+1}轮 新增连通体素：{changed}")
        if changed == 0:
            print("气道完全连通，提前终止")
            break
    organ_labels = torch.zeros_like(vol, dtype=torch.int32, device=device)
    organ_labels[grow_mask] = 1
    total_vox = grow_mask.sum().item()
    print(f"浸润完成，气道总连通体素：{total_vox}")
    return organ_labels, grow_mask

# 6. 保存网格 + 反转外壁逻辑（实体向外膨胀，外层裸露）
def save_results(coord_3d, hu_vol, organ_label, airway_mask, pix_spacing, slice_thick, save_prefix):
    hu_np = hu_vol.cpu().numpy()
    label_np = organ_label.cpu().numpy().astype(np.int32)
    airway_np = airway_mask.cpu().numpy().astype(np.float32)
    D, H, W = hu_np.shape

    # 反转逻辑：取气道实体，向外膨胀得到外层表皮
    airway_bin = (airway_np > 0.5).astype(np.uint8)
    kernel3 = create_sphere_kernel(3)
    dilate_full = binary_dilation(airway_bin, kernel3)
    airway_wall_bin = dilate_full & (~airway_bin)
    airway_wall_bin = binary_dilation(airway_wall_bin, create_sphere_kernel(1))
    airway_wall_np = airway_wall_bin.astype(np.float32)

    grid = pv.ImageData()
    grid.dimensions = np.array([W, H, D]) + 1
    grid.spacing = (pix_spacing[0], pix_spacing[1], slice_thick)
    grid.cell_data["HU_original"] = hu_np.transpose(2,1,0).ravel(order="F")
    grid.cell_data["airway_mask"] = airway_np.transpose(2,1,0).ravel(order="F")
    grid.cell_data["airway_wall"] = airway_wall_np.transpose(2,1,0).ravel(order="F")

    os.makedirs("./output", exist_ok=True)
    vtk_path = f"./output/{save_prefix}_volume.vtk"
    grid.save(vtk_path)
    print(f"\n三维网格保存：{vtk_path}")
    return grid

# 7. 渲染修复：替换非法 cmap="orange" 为标准 hot，外壁置顶渲染
def render_volume(grid):
    plotter = pv.Plotter(window_size=(1300,900))
    plotter.background_color = BACKGROUND_COLOR
    plotter.add_title("LSG 气管外壁外侧可见 | 标准色表兼容", font_size=16)

    # 底层肺极淡半透
    plotter.add_volume(
        grid,
        scalars="HU_original",
        cmap="gray",
        clim=[HU_DISP_MIN, HU_DISP_MAX],
        opacity=[0, 0.01, 0.06, 0.12]
    )
    # 气道内部主体
    plotter.add_volume(
        grid,
        scalars="airway_mask",
        cmap="coolwarm",
        opacity=[0, 0, 0.8, 1.0]
    )
    # 气管外壁置顶，替换 orange → hot（标准内置色表，无报错）
    plotter.add_volume(
        grid,
        scalars="airway_wall",
        cmap="hot",
        opacity=[0, 0, 0.7, 0.95]
    )
    plotter.add_axes(xlabel="X mm", ylabel="Y mm", zlabel="Z Slice")
    plotter.camera_position = "iso"
    plotter.show()

# 快速加载渲染
def fast_render(vtk_path):
    g = pv.read(vtk_path)
    render_volume(g)

# 主程序
if __name__ == "__main__":
    print("="*70)
    print("LSG V3.3 气道重建【cmap非法报错+内外轮廓颠倒双修复】")
    print("="*70)

    print("\n【步骤1】构建三维坐标，加载完整HU CT")
    coord_matrix, hu_volume, pix_space, slice_t = load_ct_with_coord(DICOM_FOLDER)

    print("\n【步骤2】过滤空气泥土，渲染自动透明")
    valid_mask_tensor = get_valid_voxel_mask(hu_volume)

    print("\n【步骤3】提取气管、结节生长种子")
    seed_mask_tensor = get_seed_points(hu_volume, valid_mask_tensor)

    print("\n【步骤4】双向全域浸润，自发连通完整气道")
    organ_label_tensor, airway_mask_tensor = seed_infusion_emerge(hu_volume, valid_mask_tensor, seed_mask_tensor)

    print("\n【步骤5】保存网格，生成外侧裸露气管外壁")
    volume_mesh = save_results(coord_matrix, hu_volume, organ_label_tensor, airway_mask_tensor, pix_space, slice_t, SAVE_PREFIX)
    render_volume(volume_mesh)

    print("\n✅ 运行完成：")
    print("1. 替换非法 orange 色表为标准 hot，不再报 cmap 错误")
    print("2. 外壁生成逻辑反转：气道实体向外膨胀，表皮裸露在肺外侧，旋转外部能看见完整气管外形")
    print("3. 外壁渲染图层置顶，不会被内部气道遮挡")
