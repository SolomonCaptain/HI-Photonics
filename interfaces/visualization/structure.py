"""
结构可视化模块

提供光子学器件结构、设计参数、几何布局等可视化功能。
"""

from typing import Optional, Tuple, List, Union, Dict, Any, Literal
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

# matplotlib 后端设置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Polygon, Circle, Patch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# 可选：3D 可视化
try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False


@dataclass
class StructurePlotConfig:
    """结构图配置"""
    # 图像尺寸
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 150
    
    # 颜色映射
    cmap: str = "gray"           # 灰度图
    binary_cmap: List[str] = None  # type: ignore  # 二值图颜色
    
    # 数值范围
    vmin: float = 0.0
    vmax: float = 1.0
    
    # 轴标签
    xlabel: str = "x (μm)"
    ylabel: str = "y (μm)"
    title: Optional[str] = None
    
    # 物理尺寸
    extent: Optional[Tuple[float, float, float, float]] = None
    
    # 轮廓和边界
    show_contour: bool = True
    contour_levels: int = 10
    contour_color: str = "white"
    
    # 材料标签
    show_materials: bool = True
    material_colors: Dict[str, str] = None  # type: ignore
    
    # 边框
    show_border: bool = True
    border_color: str = "black"
    border_width: float = 2.0
    
    def __post_init__(self):
        if self.binary_cmap is None:
            self.binary_cmap = ['white', '#2c3e50']  # 背景、材料
        if self.material_colors is None:
            self.material_colors = {
                'silicon': '#2c3e50',
                'silica': '#ecf0f1',
                'air': 'white',
                'metal': '#7f8c8d',
            }


class StructureVisualizer:
    """
    结构可视化器
    
    支持多种结构可视化：
    - 设计参数分布
    - 二值结构
    - 多层结构
    - 光栅/波导结构
    - 设计演化动画
    """
    
    def __init__(self, config: Optional[StructurePlotConfig] = None):
        self.config = config or StructurePlotConfig()
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """转换为 numpy 数组"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    
    def _get_extent(
        self,
        data: np.ndarray,
        extent: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[float, float, float, float]:
        """获取图像范围"""
        if extent is not None:
            return extent
        return (0, data.shape[-1], 0, data.shape[-2])
    
    def plot_design(
        self,
        design: Union[np.ndarray, torch.Tensor],
        extent: Optional[Tuple[float, float, float, float]] = None,
        ax: Optional[Axes] = None,
        threshold: Optional[float] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制设计参数分布
        
        Args:
            design: 设计参数 [H, W]，值范围 [0, 1]
            extent: 物理范围 (xmin, xmax, ymin, ymax)
            ax: 已有的坐标轴
            threshold: 二值化阈值，None 表示连续灰度图
            **kwargs: 额外参数
            
        Returns:
            (figure, axes)
        """
        design = self._to_numpy(design)
        
        # 创建图形
        if ax is None:
            self._fig, self._ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = self._ax
        else:
            self._ax = ax
            self._fig = ax.figure
        
        # 获取范围
        extent = self._get_extent(design, extent or self.config.extent)
        
        # 确定颜色映射
        if threshold is not None:
            # 二值化
            binary_design = (design > threshold).astype(float)
            cmap = mcolors.ListedColormap(self.config.binary_cmap)
            vmin, vmax = 0, 1
        else:
            binary_design = design
            cmap = kwargs.get('cmap', self.config.cmap)
            vmin = kwargs.get('vmin', self.config.vmin)
            vmax = kwargs.get('vmax', self.config.vmax)
        
        # 绘制设计
        im = ax.imshow(
            binary_design,
            extent=extent,
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        
        # 设置标签
        ax.set_xlabel(kwargs.get('xlabel', self.config.xlabel))
        ax.set_ylabel(kwargs.get('ylabel', self.config.ylabel))
        
        title = kwargs.get('title', self.config.title)
        if title:
            ax.set_title(title)
        
        # 等高线
        if kwargs.get('show_contour', self.config.show_contour) and threshold is None:
            levels = kwargs.get('contour_levels', self.config.contour_levels)
            x = np.linspace(extent[0], extent[1], design.shape[1])
            y = np.linspace(extent[2], extent[3], design.shape[0])
            ax.contour(x, y, design, levels=levels,
                      colors=kwargs.get('contour_color', self.config.contour_color),
                      alpha=0.5, linewidths=0.5)
        
        # 边框
        if kwargs.get('show_border', self.config.show_border):
            rect = Rectangle(
                (extent[0], extent[2]),
                extent[1] - extent[0],
                extent[3] - extent[2],
                fill=False,
                edgecolor=kwargs.get('border_color', self.config.border_color),
                linewidth=kwargs.get('border_width', self.config.border_width)
            )
            ax.add_patch(rect)
        
        # 颜色条
        if kwargs.get('colorbar', False):
            cbar_label = kwargs.get('colorbar_label', 'Density')
            plt.colorbar(im, ax=ax, label=cbar_label)
        
        plt.tight_layout()
        
        return self._fig, ax
    
    def plot_binary_structure(
        self,
        structure: Union[np.ndarray, torch.Tensor],
        extent: Optional[Tuple[float, float, float, float]] = None,
        material_name: str = "silicon",
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制二值结构
        
        Args:
            structure: 二值结构 [H, W]
            extent: 物理范围
            material_name: 材料名称（用于颜色）
            ax: 已有的坐标轴
            
        Returns:
            (figure, axes)
        """
        structure = self._to_numpy(structure)
        
        # 获取材料颜色
        material_color = self.config.material_colors.get(material_name, '#2c3e50')
        bg_color = self.config.material_colors.get('air', 'white')
        
        cmap = mcolors.ListedColormap([bg_color, material_color])
        
        kwargs.setdefault('cmap', cmap)
        kwargs.setdefault('vmin', 0)
        kwargs.setdefault('vmax', 1)
        kwargs.setdefault('show_contour', False)
        
        return self.plot_design(structure, extent, ax, **kwargs)
    
    def plot_multilayer(
        self,
        layers: List[Union[np.ndarray, torch.Tensor]],
        layer_names: Optional[List[str]] = None,
        extent: Optional[Tuple[float, float, float, float]] = None,
        spacing: float = 0.5,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制多层结构（侧视图）
        
        Args:
            layers: 各层结构列表
            layer_names: 层名称列表
            extent: 物理范围
            spacing: 层间距
            
        Returns:
            (figure, axes)
        """
        n_layers = len(layers)
        layers_np = [self._to_numpy(layer) for layer in layers]
        
        layer_names = layer_names or [f"Layer {i+1}" for i in range(n_layers)]
        
        # 计算总高度
        layer_heights = [layer.shape[0] for layer in layers_np]
        total_height = sum(layer_heights) + spacing * (n_layers - 1)
        
        # 创建图形
        self._fig, self._ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        ax = self._ax
        
        # 获取范围
        if extent is None:
            extent = (0, layers_np[0].shape[1], 0, total_height)
        
        # 绘制各层
        y_offset = 0
        for i, (layer, name) in enumerate(zip(layers_np, layer_names)):
            layer_extent = (extent[0], extent[1], y_offset, y_offset + layer_heights[i])
            
            # 使用对应层颜色
            color = self.config.material_colors.get(name, list(self.config.material_colors.values())[i % len(self.config.material_colors)])
            cmap = mcolors.ListedColormap(['white', color])
            
            ax.imshow(
                layer,
                extent=layer_extent,
                origin='lower',
                cmap=cmap,
                vmin=0, vmax=1,
                aspect='auto'
            )
            
            # 层标签
            ax.text(extent[1] + 0.5, y_offset + layer_heights[i] / 2, name,
                   va='center', fontsize=10)
            
            y_offset += layer_heights[i] + spacing
        
        ax.set_xlabel(kwargs.get('xlabel', 'x (μm)'))
        ax.set_ylabel(kwargs.get('ylabel', 'z (μm)'))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(0, total_height)
        
        title = kwargs.get('title')
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        return self._fig, ax
    
    def plot_grating(
        self,
        periods: int,
        period: float,
        fill_factor: float,
        etch_depth: float,
        waveguide_height: float = 0.22,
        width: float = 10.0,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制光栅结构示意图
        
        Args:
            periods: 光栅周期数
            period: 周期长度 (μm)
            fill_factor: 填充因子
            etch_depth: 刻蚀深度 (μm)
            waveguide_height: 波导高度 (μm)
            width: 结构宽度 (μm)
            ax: 已有的坐标轴
            
        Returns:
            (figure, axes)
        """
        if ax is None:
            self._fig, self._ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = self._ax
        else:
            self._ax = ax
            self._fig = ax.figure
        
        silicon_color = self.config.material_colors.get('silicon', '#2c3e50')
        silica_color = self.config.material_colors.get('silica', '#ecf0f1')
        
        # 背景（衬底）
        ax.add_patch(Rectangle(
            (0, -waveguide_height),
            periods * period,
            waveguide_height,
            facecolor=silica_color,
            edgecolor='none'
        ))
        
        # 波导层
        ax.add_patch(Rectangle(
            (0, 0),
            periods * period,
            waveguide_height,
            facecolor=silicon_color,
            edgecolor='none'
        ))
        
        # 光栅齿
        tooth_width = period * fill_factor
        for i in range(periods):
            x_start = i * period
            ax.add_patch(Rectangle(
                (x_start, 0),
                tooth_width,
                waveguide_height - etch_depth,
                facecolor=silicon_color,
                edgecolor='none'
            ))
            # 刻蚀区域
            ax.add_patch(Rectangle(
                (x_start + tooth_width, 0),
                period - tooth_width,
                waveguide_height - etch_depth,
                facecolor=silica_color,
                edgecolor='none'
            ))
        
        ax.set_xlim(-0.5, periods * period + 0.5)
        ax.set_ylim(-waveguide_height, waveguide_height + 0.5)
        ax.set_xlabel(kwargs.get('xlabel', 'x (μm)'))
        ax.set_ylabel(kwargs.get('ylabel', 'z (μm)'))
        
        title = kwargs.get('title', f'Grating: {periods} periods, Λ={period}μm, ff={fill_factor}')
        ax.set_title(title)
        
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        return self._fig, ax
    
    def plot_waveguide(
        self,
        width: float = 0.5,
        height: float = 0.22,
        length: float = 10.0,
        cladding_width: float = 2.0,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制波导结构示意图
        
        Args:
            width: 波导宽度 (μm)
            height: 波导高度 (μm)
            length: 波导长度 (μm)
            cladding_width: 包层宽度 (μm)
            ax: 已有的坐标轴
            
        Returns:
            (figure, axes)
        """
        if ax is None:
            self._fig, self._ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = self._ax
        else:
            self._ax = ax
            self._fig = ax.figure
        
        silicon_color = self.config.material_colors.get('silicon', '#2c3e50')
        silica_color = self.config.material_colors.get('silica', '#ecf0f1')
        
        # 包层
        ax.add_patch(Rectangle(
            (0, -cladding_width / 2),
            length,
            cladding_width,
            facecolor=silica_color,
            edgecolor='none'
        ))
        
        # 波导芯
        ax.add_patch(Rectangle(
            (0, -width / 2),
            length,
            width,
            facecolor=silicon_color,
            edgecolor=silicon_color,
            linewidth=1
        ))
        
        # 尺寸标注
        ax.annotate('', xy=(0, -width / 2 - 0.1), xytext=(0, width / 2 + 0.1),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(-0.3, 0, f'{width}μm', color='red', fontsize=10, ha='right', va='center')
        
        ax.set_xlim(-1, length + 1)
        ax.set_ylim(-cladding_width, cladding_width)
        ax.set_xlabel(kwargs.get('xlabel', 'z (μm)'))
        ax.set_ylabel(kwargs.get('ylabel', 'x (μm)'))
        
        title = kwargs.get('title', f'Waveguide: w={width}μm, h={height}μm')
        ax.set_title(title)
        
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        return self._fig, ax
    
    def plot_design_evolution(
        self,
        designs: List[Union[np.ndarray, torch.Tensor]],
        extent: Optional[Tuple[float, float, float, float]] = None,
        cols: int = 4,
        **kwargs
    ) -> Tuple[Figure, List[Axes]]:
        """
        绘制设计演化过程
        
        Args:
            designs: 设计序列
            extent: 物理范围
            cols: 列数
            
        Returns:
            (figure, axes_list)
        """
        n_designs = len(designs)
        rows = (n_designs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, design in enumerate(designs):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            design_np = self._to_numpy(design)
            extent_i = self._get_extent(design_np, extent)
            
            ax.imshow(design_np, extent=extent_i, origin='lower',
                     cmap=kwargs.get('cmap', self.config.cmap),
                     vmin=kwargs.get('vmin', self.config.vmin),
                     vmax=kwargs.get('vmax', self.config.vmax),
                     aspect='auto')
            
            ax.set_title(f'Iteration {i}', fontsize=9)
            ax.axis('off')
        
        # 隐藏多余子图
        for i in range(n_designs, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        return fig, axes
    
    def plot_optimization_history(
        self,
        objectives: List[float],
        constraints: Optional[List[float]] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制优化历史曲线
        
        Args:
            objectives: 目标函数值序列
            constraints: 约束值序列（可选）
            ax: 已有的坐标轴
            
        Returns:
            (figure, axes)
        """
        if ax is None:
            self._fig, self._ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = self._ax
        else:
            self._ax = ax
            self._fig = ax.figure
        
        iterations = range(len(objectives))
        
        ax.plot(iterations, objectives, 'b-', linewidth=2, label='Objective')
        ax.set_xlabel(kwargs.get('xlabel', 'Iteration'))
        ax.set_ylabel(kwargs.get('ylabel', 'Objective'))
        
        if constraints is not None:
            ax2 = ax.twinx()
            ax2.plot(iterations, constraints, 'r--', linewidth=1.5, label='Constraint')
            ax2.set_ylabel('Constraint', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        title = kwargs.get('title', 'Optimization History')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._fig, ax
    
    def plot_permittivity(
        self,
        eps: Union[np.ndarray, torch.Tensor],
        extent: Optional[Tuple[float, float, float, float]] = None,
        materials: Optional[Dict[str, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制介电常数分布
        
        Args:
            eps: 介电常数分布
            extent: 物理范围
            materials: 材料介电常数字典 {name: eps}
            ax: 已有的坐标轴
            
        Returns:
            (figure, axes)
        """
        eps = self._to_numpy(eps)
        
        # 默认材料
        if materials is None:
            materials = {
                'air': 1.0,
                'silica': 2.1,
                'silicon': 12.0,
            }
        
        # 创建离散颜色映射
        eps_values = sorted(materials.values())
        n_materials = len(eps_values)
        colors = plt.cm.viridis(np.linspace(0, 1, n_materials))
        cmap = mcolors.ListedColormap(colors)
        
        kwargs.setdefault('cmap', cmap)
        kwargs.setdefault('vmin', min(eps_values))
        kwargs.setdefault('vmax', max(eps_values))
        kwargs.setdefault('colorbar', True)
        kwargs.setdefault('colorbar_label', 'ε')
        
        return self.plot_design(eps, extent, ax, **kwargs)
    
    def add_scale_bar(
        self,
        ax: Optional[Axes] = None,
        length: float = 1.0,
        position: Tuple[float, float] = (0.05, 0.05),
        unit: str = "μm",
        **kwargs
    ) -> None:
        """
        添加比例尺
        
        Args:
            ax: 坐标轴
            length: 比例尺长度
            position: 位置（相对坐标）
            unit: 单位
        """
        ax = ax or self._ax
        if ax is None:
            return
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        x = xlim[0] + position[0] * (xlim[1] - xlim[0])
        y = ylim[0] + position[1] * (ylim[1] - ylim[0])
        
        ax.plot([x, x + length], [y, y], 'k-', linewidth=3)
        ax.text(x + length / 2, y + 0.1 * (ylim[1] - ylim[0]),
               f'{length} {unit}', ha='center', fontsize=9)
    
    def save(
        self,
        filepath: Union[str, Path],
        fig: Optional[Figure] = None,
        dpi: Optional[int] = None
    ) -> None:
        """保存图形"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fig = fig or self._fig
        if fig is None:
            raise ValueError("没有可保存的图形")
        
        fig.savefig(filepath, dpi=dpi or self.config.dpi, bbox_inches='tight')
        plt.close(fig)
    
    def show(self, fig: Optional[Figure] = None) -> None:
        """显示图形"""
        fig = fig or self._fig
        if fig is not None:
            plt.show()


def plot_design(
    design: Union[np.ndarray, torch.Tensor],
    extent: Optional[Tuple[float, float, float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    快速绘制设计参数分布
    
    Args:
        design: 设计参数
        extent: 物理范围
        title: 标题
        save_path: 保存路径
        
    Returns:
        (figure, axes)
    """
    config = StructurePlotConfig(title=title)
    visualizer = StructureVisualizer(config)
    fig, ax = visualizer.plot_design(design, extent, **kwargs)
    
    if save_path:
        visualizer.save(save_path, fig)
    
    return fig, ax


def plot_binary_structure(
    structure: Union[np.ndarray, torch.Tensor],
    extent: Optional[Tuple[float, float, float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    快速绘制二值结构
    
    Args:
        structure: 二值结构
        extent: 物理范围
        title: 标题
        save_path: 保存路径
        
    Returns:
        (figure, axes)
    """
    config = StructurePlotConfig(title=title)
    visualizer = StructureVisualizer(config)
    fig, ax = visualizer.plot_binary_structure(structure, extent, **kwargs)
    
    if save_path:
        visualizer.save(save_path, fig)
    
    return fig, ax


def create_structure_visualizer(
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "gray",
    **kwargs
) -> StructureVisualizer:
    """
    创建结构可视化器的便捷函数
    
    Args:
        figsize: 图像尺寸
        cmap: 颜色映射
        **kwargs: 其他配置参数
        
    Returns:
        StructureVisualizer 实例
    """
    config = StructurePlotConfig(figsize=figsize, cmap=cmap, **kwargs)
    return StructureVisualizer(config)
