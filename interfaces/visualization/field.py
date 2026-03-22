"""
场分布可视化模块

提供电磁场、功率流、模场分布等可视化功能。
"""

from typing import Optional, Tuple, List, Union, Dict, Any, Literal
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

# matplotlib 后端设置
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


@dataclass
class FieldPlotConfig:
    """场分布图配置"""
    # 图像尺寸
    figsize: Tuple[float, float] = (10, 8)
    dpi: int = 150
    
    # 颜色映射
    cmap: str = "RdBu_r"          # 场分布默认颜色
    intensity_cmap: str = "hot"   # 强度图颜色
    phase_cmap: str = "hsv"       # 相位图颜色
    
    # 数值范围
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    symmetric: bool = True        # 对称颜色范围
    
    # 轴标签
    xlabel: str = "x (μm)"
    ylabel: str = "y (μm)"
    title: Optional[str] = None
    
    # 颜色条
    colorbar: bool = True
    colorbar_label: str = ""
    
    # 网格和边界
    show_grid: bool = False
    show_contour: bool = False
    contour_levels: int = 20
    
    # 物理尺寸
    extent: Optional[Tuple[float, float, float, float]] = None  # (xmin, xmax, ymin, ymax)


class FieldVisualizer:
    """
    场分布可视化器
    
    支持多种场分布的可视化：
    - 电场/磁场分量 (Ex, Ey, Ez, Hx, Hy, Hz)
    - 功率密度 |E|², |H|²
    - 坡印廷矢量
    - 模场分布
    """
    
    def __init__(self, config: Optional[FieldPlotConfig] = None):
        self.config = config or FieldPlotConfig()
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
    
    def _get_vrange(
        self,
        data: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        symmetric: bool = False
    ) -> Tuple[float, float]:
        """获取数值范围"""
        if vmin is None or vmax is None:
            data_max = np.abs(data).max()
            
            if symmetric:
                vmin = vmin or -data_max
                vmax = vmax or data_max
            else:
                vmin = vmin or data.min()
                vmax = vmax or data.max()
        
        return vmin, vmax
    
    def plot_field(
        self,
        field: Union[np.ndarray, torch.Tensor],
        extent: Optional[Tuple[float, float, float, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制场分布
        
        Args:
            field: 场数据 [H, W] 或 [H, W, C]
            extent: 物理范围 (xmin, xmax, ymin, ymax)
            ax: 已有的坐标轴
            **kwargs: 额外参数覆盖配置
            
        Returns:
            (figure, axes)
        """
        field = self._to_numpy(field)
        
        # 处理复数场
        if np.iscomplexobj(field):
            field = np.abs(field)
        
        # 处理多维场
        if field.ndim > 2:
            field = field[..., 0] if field.shape[-1] <= 4 else field.mean(axis=-1)
        
        # 创建图形
        if ax is None:
            self._fig, self._ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = self._ax
        else:
            self._ax = ax
            self._fig = ax.figure
        
        # 获取范围
        extent = self._get_extent(field, extent or self.config.extent)
        vmin, vmax = self._get_vrange(
            field,
            kwargs.get('vmin', self.config.vmin),
            kwargs.get('vmax', self.config.vmax),
            kwargs.get('symmetric', self.config.symmetric)
        )
        
        # 绘制场分布
        cmap = kwargs.get('cmap', self.config.cmap)
        im = ax.imshow(
            field,
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
        
        # 颜色条
        if kwargs.get('colorbar', self.config.colorbar):
            cbar_label = kwargs.get('colorbar_label', self.config.colorbar_label)
            plt.colorbar(im, ax=ax, label=cbar_label)
        
        # 等高线
        if kwargs.get('show_contour', self.config.show_contour):
            levels = kwargs.get('contour_levels', self.config.contour_levels)
            x = np.linspace(extent[0], extent[1], field.shape[1])
            y = np.linspace(extent[2], extent[3], field.shape[0])
            ax.contour(x, y, field, levels=levels, colors='white', alpha=0.5, linewidths=0.5)
        
        # 网格
        if kwargs.get('show_grid', self.config.show_grid):
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._fig, ax
    
    def plot_intensity(
        self,
        field: Union[np.ndarray, torch.Tensor],
        extent: Optional[Tuple[float, float, float, float]] = None,
        log_scale: bool = False,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制强度分布 |E|²
        
        Args:
            field: 场数据
            extent: 物理范围
            log_scale: 是否使用对数刻度
            ax: 已有的坐标轴
            
        Returns:
            (figure, axes)
        """
        field = self._to_numpy(field)
        
        # 计算强度
        if np.iscomplexobj(field):
            intensity = np.abs(field) ** 2
        else:
            intensity = field ** 2
        
        # 对数刻度
        if log_scale:
            intensity = 10 * np.log10(intensity + 1e-12)
            kwargs.setdefault('colorbar_label', 'Intensity (dB)')
        else:
            kwargs.setdefault('colorbar_label', 'Intensity')
        
        # 强度图使用热图颜色
        kwargs.setdefault('cmap', self.config.intensity_cmap)
        kwargs.setdefault('symmetric', False)
        
        return self.plot_field(intensity, extent, ax, **kwargs)
    
    def plot_phase(
        self,
        field: Union[np.ndarray, torch.Tensor],
        extent: Optional[Tuple[float, float, float, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制相位分布
        
        Args:
            field: 复数场数据
            extent: 物理范围
            ax: 已有的坐标轴
            
        Returns:
            (figure, axes)
        """
        field = self._to_numpy(field)
        
        if not np.iscomplexobj(field):
            raise ValueError("相位图需要复数场数据")
        
        phase = np.angle(field)
        
        kwargs.setdefault('cmap', self.config.phase_cmap)
        kwargs.setdefault('symmetric', True)
        kwargs.setdefault('vmin', -np.pi)
        kwargs.setdefault('vmax', np.pi)
        kwargs.setdefault('colorbar_label', 'Phase (rad)')
        
        return self.plot_field(phase, extent, ax, **kwargs)
    
    def plot_complex_field(
        self,
        field: Union[np.ndarray, torch.Tensor],
        extent: Optional[Tuple[float, float, float, float]] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制复数场的幅度和相位
        
        Args:
            field: 复数场数据
            extent: 物理范围
            
        Returns:
            (figure, axes)
        """
        field = self._to_numpy(field)
        
        if not np.iscomplexobj(field):
            raise ValueError("复数场图需要复数场数据")
        
        fig, axes = plt.subplots(1, 2, figsize=(self.config.figsize[0] * 2, self.config.figsize[1]))
        
        # 幅度
        self.plot_intensity(field, extent, ax=axes[0], title='Amplitude', **kwargs)
        
        # 相位
        self.plot_phase(field, extent, ax=axes[1], title='Phase', **kwargs)
        
        plt.tight_layout()
        
        return fig, axes
    
    def plot_cross_section(
        self,
        field: Union[np.ndarray, torch.Tensor],
        position: Union[int, float] = 0.5,
        axis: int = 0,
        extent: Optional[Tuple[float, float, float, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制场的横截面
        
        Args:
            field: 场数据
            position: 截面位置（相对位置 0-1 或绝对索引）
            axis: 截面方向（0=y轴截面，1=x轴截面）
            extent: 物理范围
            ax: 已有的坐标轴
            
        Returns:
            (figure, axes)
        """
        field = self._to_numpy(field)
        
        if np.iscomplexobj(field):
            field = np.abs(field)
        
        # 获取截面
        if isinstance(position, float) and 0 <= position <= 1:
            idx = int(position * (field.shape[axis] - 1))
        else:
            idx = int(position)
        
        if axis == 0:
            cross_section = field[idx, :]
            x_label = kwargs.get('xlabel', self.config.xlabel)
        else:
            cross_section = field[:, idx]
            x_label = kwargs.get('ylabel', self.config.ylabel)
        
        # 创建图形
        if ax is None:
            self._fig, self._ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            ax = self._ax
        else:
            self._ax = ax
            self._fig = ax.figure
        
        # 坐标轴
        if extent is not None:
            if axis == 0:
                x = np.linspace(extent[0], extent[1], len(cross_section))
            else:
                x = np.linspace(extent[2], extent[3], len(cross_section))
        else:
            x = np.arange(len(cross_section))
        
        # 绘制
        ax.plot(x, cross_section, linewidth=1.5)
        ax.fill_between(x, cross_section, alpha=0.3)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(kwargs.get('ylabel', 'Field amplitude'))
        
        title = kwargs.get('title')
        if title:
            ax.set_title(title)
        
        if kwargs.get('show_grid', self.config.show_grid):
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._fig, ax
    
    def plot_poynting_vector(
        self,
        E: Union[np.ndarray, torch.Tensor],
        H: Union[np.ndarray, torch.Tensor],
        extent: Optional[Tuple[float, float, float, float]] = None,
        show_quiver: bool = True,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        绘制坡印廷矢量分布
        
        Args:
            E: 电场 [H, W, 3] 或 [H, W]
            H: 磁场 [H, W, 3] 或 [H, W]
            extent: 物理范围
            show_quiver: 是否显示矢量箭头
            ax: 已有的坐标轴
            
        Returns:
            (figure, axes)
        """
        E = self._to_numpy(E)
        H = self._to_numpy(H)
        
        # 计算坡印廷矢量 S = E × H
        if E.ndim == 2:
            # 2D 情况，假设 Ez 和 Hz 非零
            Sx = -E * H  # 简化
            Sy = np.zeros_like(Sx)
            S_mag = np.abs(Sx)
        else:
            # 3D 矢量场
            Sx = E[..., 1] * H[..., 2] - E[..., 2] * H[..., 1]
            Sy = E[..., 2] * H[..., 0] - E[..., 0] * H[..., 2]
            S_mag = np.sqrt(Sx**2 + Sy**2)
        
        # 绘制大小
        fig, ax = self.plot_intensity(S_mag, extent, ax=ax, 
                                      colorbar_label='|S|', **kwargs)
        
        # 绘制矢量
        if show_quiver:
            # 下采样
            skip = max(1, max(S_mag.shape) // 20)
            y_idx, x_idx = np.mgrid[0:S_mag.shape[0]:skip, 0:S_mag.shape[1]:skip]
            
            if extent is not None:
                x_pos = np.linspace(extent[0], extent[1], S_mag.shape[1])[x_idx]
                y_pos = np.linspace(extent[2], extent[3], S_mag.shape[0])[y_idx]
            else:
                x_pos = x_idx
                y_pos = y_idx
            
            Sx_down = Sx[::skip, ::skip]
            Sy_down = Sy[::skip, ::skip]
            S_mag_down = S_mag[::skip, ::skip]
            
            # 归一化
            norm = S_mag_down.max() + 1e-12
            ax.quiver(x_pos, y_pos, Sx_down / norm, Sy_down / norm,
                     color='white', alpha=0.7, scale=20)
        
        return fig, ax
    
    def plot_fields_dict(
        self,
        fields: Dict[str, Union[np.ndarray, torch.Tensor]],
        extent: Optional[Tuple[float, float, float, float]] = None,
        cols: int = 3,
        **kwargs
    ) -> Tuple[Figure, List[Axes]]:
        """
        绘制多个场分量
        
        Args:
            fields: 场分量字典 {name: field}
            extent: 物理范围
            cols: 列数
            
        Returns:
            (figure, axes_list)
        """
        n_fields = len(fields)
        rows = (n_fields + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, (name, field) in enumerate(fields.items()):
            row, col = i // cols, i % cols
            ax = axes[row][col]
            
            self.plot_field(field, extent, ax=ax, title=name, colorbar=True, **kwargs)
        
        # 隐藏多余的子图
        for i in range(n_fields, rows * cols):
            row, col = i // cols, i % cols
            axes[row][col].axis('off')
        
        plt.tight_layout()
        
        return fig, axes
    
    def create_animation(
        self,
        fields_sequence: List[Union[np.ndarray, torch.Tensor]],
        extent: Optional[Tuple[float, float, float, float]] = None,
        interval: int = 50,
        **kwargs
    ) -> FuncAnimation:
        """
        创建场分布动画
        
        Args:
            fields_sequence: 时间序列场数据
            extent: 物理范围
            interval: 帧间隔（毫秒）
            
        Returns:
            FuncAnimation 对象
        """
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # 转换数据
        fields_seq = [self._to_numpy(f) for f in fields_sequence]
        if np.iscomplexobj(fields_seq[0]):
            fields_seq = [np.abs(f) for f in fields_seq]
        
        # 确定范围
        extent = self._get_extent(fields_seq[0], extent)
        vmin, vmax = self._get_vrange(
            np.array(fields_seq),
            kwargs.get('vmin'),
            kwargs.get('vmax'),
            kwargs.get('symmetric', self.config.symmetric)
        )
        
        # 初始化图像
        im = ax.imshow(
            fields_seq[0],
            extent=extent,
            origin='lower',
            cmap=kwargs.get('cmap', self.config.cmap),
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        
        ax.set_xlabel(kwargs.get('xlabel', self.config.xlabel))
        ax.set_ylabel(kwargs.get('ylabel', self.config.ylabel))
        
        if kwargs.get('colorbar', self.config.colorbar):
            plt.colorbar(im, ax=ax)
        
        title = ax.set_title('Frame 0')
        
        def update(frame):
            im.set_array(fields_seq[frame])
            title.set_text(f'Frame {frame}')
            return [im, title]
        
        anim = FuncAnimation(
            fig, update,
            frames=len(fields_seq),
            interval=interval,
            blit=True
        )
        
        return anim
    
    def save(
        self,
        filepath: Union[str, Path],
        fig: Optional[Figure] = None,
        dpi: Optional[int] = None
    ) -> None:
        """
        保存图形
        
        Args:
            filepath: 保存路径
            fig: 图形对象，None 使用当前图形
            dpi: 分辨率
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fig = fig or self._fig
        if fig is None:
            raise ValueError("没有可保存的图形")
        
        fig.savefig(filepath, dpi=dpi or self.config.dpi, bbox_inches='tight')
        plt.close(fig)
    
    def save_animation(
        self,
        anim: FuncAnimation,
        filepath: Union[str, Path],
        fps: int = 20,
        **kwargs
    ) -> None:
        """
        保存动画
        
        Args:
            anim: 动画对象
            filepath: 保存路径
            fps: 帧率
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        writer = 'pillow' if filepath.suffix == '.gif' else 'ffmpeg'
        anim.save(str(filepath), writer=writer, fps=fps, **kwargs)
        plt.close(anim._fig)
    
    def show(self, fig: Optional[Figure] = None) -> None:
        """显示图形"""
        fig = fig or self._fig
        if fig is not None:
            plt.show()


def plot_field(
    field: Union[np.ndarray, torch.Tensor],
    extent: Optional[Tuple[float, float, float, float]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    快速绘制场分布
    
    Args:
        field: 场数据
        extent: 物理范围
        title: 标题
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        (figure, axes)
    """
    config = FieldPlotConfig(title=title)
    visualizer = FieldVisualizer(config)
    fig, ax = visualizer.plot_field(field, extent, **kwargs)
    
    if save_path:
        visualizer.save(save_path, fig)
    
    return fig, ax


def plot_intensity(
    field: Union[np.ndarray, torch.Tensor],
    extent: Optional[Tuple[float, float, float, float]] = None,
    log_scale: bool = False,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    快速绘制强度分布
    
    Args:
        field: 场数据
        extent: 物理范围
        log_scale: 是否对数刻度
        title: 标题
        save_path: 保存路径
        
    Returns:
        (figure, axes)
    """
    config = FieldPlotConfig(title=title)
    visualizer = FieldVisualizer(config)
    fig, ax = visualizer.plot_intensity(field, extent, log_scale=log_scale, **kwargs)
    
    if save_path:
        visualizer.save(save_path, fig)
    
    return fig, ax


def create_field_visualizer(
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "RdBu_r",
    **kwargs
) -> FieldVisualizer:
    """
    创建场分布可视化器的便捷函数
    
    Args:
        figsize: 图像尺寸
        cmap: 颜色映射
        **kwargs: 其他配置参数
        
    Returns:
        FieldVisualizer 实例
    """
    config = FieldPlotConfig(figsize=figsize, cmap=cmap, **kwargs)
    return FieldVisualizer(config)
