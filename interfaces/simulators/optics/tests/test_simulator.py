"""
FDTD 仿真器单元测试
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 尝试导入模块
try:
    from optics import OPTICS_AVAILABLE, OpticsSimulator, create_waveguide_simulator
except ImportError:
    OPTICS_AVAILABLE = False
    OpticsSimulator = None


@pytest.mark.skipif(not OPTICS_AVAILABLE, reason="optics C++ extension not built")
class TestOpticsSimulator:
    """Optics 仿真器测试"""
    
    def test_import(self):
        """测试模块导入"""
        from optics import _optics
        assert hasattr(_optics, 'FDTD')
    
    def test_basic_setup(self):
        """测试基本设置"""
        from optics import _optics
        
        sim = _optics.FDTD()
        sim.set_grid(100, 100, 0.02, 0.02)
        sim.set_pml(10, 0.8)
        sim.setup()
        
        assert sim.nx == 100
        assert sim.ny == 100
        assert sim.dt > 0
    
    def test_gaussian_source(self):
        """测试高斯光源"""
        from optics import _optics
        
        sim = _optics.FDTD()
        sim.set_grid(100, 100, 0.02, 0.02)
        
        # 添加高斯光源
        sim.add_gaussian_source(
            wavelength=1.55,
            center=(1.0, 1.0, 0.0),
            size=(0.0, 2.0, 0.0),
            pulse_width=10.0
        )
        
        sim.setup()
        assert sim.total_steps > 0
    
    def test_flux_monitor(self):
        """测试通量监视器"""
        from optics import _optics
        
        sim = _optics.FDTD()
        sim.set_grid(100, 100, 0.02, 0.02)
        
        # 添加监视器
        sim.add_flux_monitor(
            name='output',
            center=(2.0, 1.0, 0.0),
            size=(0.0, 2.0, 0.0),
            frequencies=[1.0/1.55]
        )
        
        sim.setup()
    
    def test_run_short_simulation(self):
        """测试短时仿真"""
        from optics import _optics
        
        sim = _optics.FDTD()
        sim.set_grid(50, 50, 0.02, 0.02)
        sim.set_pml(5)
        
        # 添加光源
        sim.add_gaussian_source(
            wavelength=1.55,
            center=(0.5, 0.5, 0.0),
            size=(0.0, 0.5, 0.0),
            pulse_width=5.0
        )
        
        sim.setup()
        
        # 运行少量时间步
        for _ in range(10):
            sim.step()
        
        assert sim.current_step == 10
        assert sim.current_time > 0
    
    def test_get_field(self):
        """测试获取场数据"""
        from optics import _optics
        
        sim = _optics.FDTD()
        sim.set_grid(50, 50, 0.02, 0.02)
        sim.setup()
        
        # 获取场数据
        Ez = sim.get_field('Ez')
        
        assert Ez.shape[0] == 50 * 50
        assert np.all(Ez == 0)  # 初始场为零
    
    def test_set_epsilon(self):
        """测试设置介电常数"""
        from optics import _optics
        
        sim = _optics.FDTD()
        sim.set_grid(50, 50, 0.02, 0.02)
        sim.setup()
        
        # 创建介电常数分布
        eps = np.ones(50 * 50) * 12.0  # 硅
        
        sim.set_epsilon(eps)
        
        # 获取并验证
        eps_get = sim.get_epsilon()
        assert eps_get.shape[0] == 50 * 50


@pytest.mark.skipif(not OPTICS_AVAILABLE, reason="optics C++ extension not built")
class TestOpticsSimulatorInterface:
    """Python 接口测试"""
    
    def test_simulator_creation(self):
        """测试仿真器创建"""
        from base import SimulationConfig
        
        config = SimulationConfig(
            resolution=50,
            cell_size=(5.0, 5.0, 0.0),
            simulation_time=50.0
        )
        
        sim = OpticsSimulator(config)
        sim.setup()
        
        assert sim._sim is not None
    
    def test_run_simple_simulation(self):
        """测试简单仿真"""
        from base import SimulationConfig, SourceConfig, SourceType
        
        config = SimulationConfig(
            resolution=30,
            cell_size=(3.0, 3.0, 0.0),
            simulation_time=20.0,
            wavelengths=[1.55]
        )
        
        sim = OpticsSimulator(config)
        
        # 添加光源
        source = SourceConfig(
            source_type=SourceType.GAUSSIAN,
            wavelength=1.55,
            center=(0.5, 1.5, 0.0),
            size=(0.0, 0.5, 0.0),
            pulse_width=5.0
        )
        sim.add_source(source)
        
        sim.setup()
        
        # 运行仿真
        design_params = np.ones((90, 90)) * 0.5  # 密度参数
        result = sim.run(design_params)
        
        assert 'Ez' in result or len(result) > 0
    
    def test_waveguide_simulator(self):
        """测试波导仿真器便捷函数"""
        sim = create_waveguide_simulator(
            length=5.0,
            width=0.5,
            resolution=30
        )
        
        sim.setup()
        
        # 创建波导结构
        nx, ny = sim.nx, sim.ny
        design_params = np.zeros((nx, ny))
        
        # 简单的直波导
        center_y = ny // 2
        width_pixels = int(0.5 * 30)  # 宽度对应的像素数
        design_params[:, center_y-width_pixels//2:center_y+width_pixels//2] = 1.0
        
        result = sim.run(design_params)
        
        # 检查结果
        assert sim.progress >= 0.99  # 仿真完成


@pytest.mark.skipif(not OPTICS_AVAILABLE, reason="optics C++ extension not built")
class TestBoundaryConditions:
    """边界条件测试"""
    
    def test_pml_boundary(self):
        """测试 PML 边界"""
        from optics import _optics
        
        sim = _optics.FDTD()
        sim.set_grid(100, 100, 0.02, 0.02)
        sim.set_pml(10, 0.8)
        sim.setup()
        
        # 添加光源在中心
        sim.add_dipole(
            wavelength=1.55,
            center=(1.0, 1.0, 0.0),
            component='Ez'
        )
        
        # 运行仿真
        sim.run(50.0)
        
        # 场应该在 PML 区域衰减
        Ez = sim.get_field('Ez')
        Ez_2d = Ez.reshape(100, 100)
        
        # 检查边界区域
        boundary_val = np.mean(np.abs(Ez_2d[:5, :]))
        center_val = np.mean(np.abs(Ez_2d[45:55, 45:55]))
        
        # PML 应该衰减边界值
        assert boundary_val < center_val * 2  # 允许一些反射


@pytest.mark.skipif(not OPTICS_AVAILABLE, reason="optics C++ extension not built")
class TestSources:
    """光源测试"""
    
    def test_dipole_source(self):
        """测试偶极子源"""
        from optics import _optics
        
        sim = _optics.FDTD()
        sim.set_grid(50, 50, 0.02, 0.02)
        sim.setup()
        
        # 添加偶极子
        sim.add_dipole(
            wavelength=1.55,
            center=(0.5, 0.5, 0.0),
            component='Ez'
        )
        
        # 运行几步
        for _ in range(20):
            sim.step()
        
        # 场应该非零
        Ez = sim.get_field('Ez')
        assert np.max(np.abs(Ez)) > 0
    
    def test_plane_wave_source(self):
        """测试平面波源"""
        from optics import _optics
        
        sim = _optics.FDTD()
        sim.set_grid(50, 50, 0.02, 0.02)
        
        sim.add_plane_wave(
            wavelength=1.55,
            center=(0.25, 0.5, 0.0),
            size=(0.0, 1.0, 0.0),
            angle=0.0
        )
        
        sim.setup()
        
        # 运行几步
        for _ in range(20):
            sim.step()
        
        Ez = sim.get_field('Ez')
        assert np.max(np.abs(Ez)) > 0


class TestSimulatorWithoutExtension:
    """不需要 C++ 扩展的测试"""
    
    def test_config_creation(self):
        """测试配置创建"""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from base import SimulationConfig, BoundaryCondition
        
        config = SimulationConfig(
            resolution=50,
            cell_size=(10.0, 10.0, 0.0),
            boundary_x=BoundaryCondition.PML,
            simulation_time=100.0
        )
        
        assert config.resolution == 50
        assert config.cell_size == (10.0, 10.0, 0.0)
        wavelengths = config.get_wavelengths()
        assert len(wavelengths) == 1
        assert wavelengths[0] == 1.55
    
    def test_source_config(self):
        """测试光源配置"""
        from base import SourceConfig, SourceType
        
        source = SourceConfig(
            source_type=SourceType.GAUSSIAN,
            wavelength=1.55,
            center=(0.0, 0.0, 0.0),
            size=(0.0, 2.0, 0.0),
            pulse_width=10.0
        )
        
        assert source.wavelength == 1.55
        assert source.frequency is not None
        assert abs(source.frequency - 1.0/1.55) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
