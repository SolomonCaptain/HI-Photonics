#pragma once

#include "types.hpp"
#include "grid.hpp"
#include "pml.hpp"
#include "source.hpp"
#include "monitor.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <map>

namespace optics {

/**
 * @brief 仿真配置
 */
struct SimulationConfig {
    // 网格参数
    int resolution = 50;           // 像素/微米
    Vec3f cell_size = {10.0, 10.0, 0.0};  // (x, y, z) 微米
    
    // 边界条件
    BoundaryConfig boundary_x = BoundaryConfig::pml(10);
    BoundaryConfig boundary_y = BoundaryConfig::pml(10);
    BoundaryConfig boundary_z = BoundaryConfig::pml(10);
    
    // 仿真时间
    Float simulation_time = 100.0;  // 时间单位
    
    // 精度
    int accuracy = 2;  // 空间差分精度
    bool force_complex_fields = false;
    
    // 输出
    std::string output_dir;
    bool save_fields = false;
    bool verbose = true;
    
    // 从 Python 创建
    static SimulationConfig from_dict(const std::map<std::string, Float>& params);
};

/**
 * @brief 仿真结果
 */
struct SimulationResult {
    // 通量数据
    std::map<std::string, std::vector<Float>> flux;
    
    // 场数据
    std::map<std::string, std::vector<Float>> fields;
    
    // 性能指标
    std::map<std::string, Float> metrics;
    
    // 元数据
    std::map<std::string, Float> metadata;
};

/**
 * @brief FDTD 仿真器主类
 * 
 * 实现完整的 FDTD 仿真功能，包括：
 * - 2D/3D 电磁场仿真
 * - PML 边界条件
 * - 多种光源类型
 * - 通量和场监视器
 */
class FDTDSimulation {
public:
    FDTDSimulation();
    ~FDTDSimulation();
    
    // ==================== 配置 ====================
    
    // 设置网格
    void set_grid(int nx, int ny, Float dx, Float dy);
    void set_grid(int nx, int ny, int nz, Float dx, Float dy, Float dz);
    void set_grid_from_config(const SimulationConfig& config);
    
    // 设置边界条件
    void set_boundary_x(const BoundaryConfig& cfg);
    void set_boundary_y(const BoundaryConfig& cfg);
    void set_boundary_z(const BoundaryConfig& cfg);
    void set_pml(int layers = 10, Float sigma_max = 0.8);
    
    // 设置极化模式
    void set_polarization(Polarization pol);
    Polarization polarization() const { return polarization_; }
    
    // ==================== 材料 ====================
    
    // 设置介电常数分布
    void set_epsilon(const std::vector<Float>& eps);
    void set_epsilon_from_density(const std::vector<Float>& density,
                                   Float eps_min, Float eps_max);
    
    // 设置材料区域
    void add_material_block(const Vec3f& center, const Vec3f& size, Float epsilon);
    
    // ==================== 光源 ====================
    
    // 添加光源
    void add_gaussian_source(Float wavelength, const Vec3f& center, const Vec3f& size,
                             Float pulse_width = 10.0);
    void add_continuous_source(Float wavelength, const Vec3f& center, const Vec3f& size,
                               Float ramp_time = 50.0);
    void add_plane_wave_source(Float wavelength, const Vec3f& center, const Vec3f& size,
                               Float angle = 0.0, Float pulse_width = 10.0);
    void add_dipole_source(Float wavelength, const Vec3f& center,
                           FieldComponent component = FieldComponent::Ez,
                           Float pulse_width = 10.0);
    
    // 直接添加光源对象
    void add_source(std::shared_ptr<Source> source);
    void clear_sources();
    
    // ==================== 监视器 ====================
    
    // 添加监视器
    void add_flux_monitor(const std::string& name, const Vec3f& center, const Vec3f& size,
                          const std::vector<Float>& frequencies);
    void add_field_monitor(const std::string& name, const Vec3f& center, const Vec3f& size,
                           const std::vector<Float>& frequencies,
                           FieldComponent component = FieldComponent::Ez);
    void add_energy_monitor(const std::string& name, const Vec3f& center, const Vec3f& size);
    
    // 直接添加监视器对象
    void add_monitor(std::shared_ptr<Monitor> monitor);
    void clear_monitors();
    
    // ==================== 运行仿真 ====================
    
    // 初始化仿真
    void setup();
    
    // 运行仿真
    void run();
    void run(Float simulation_time);
    
    // 运行单个时间步
    void step();
    
    // 运行直到完成
    void run_until_done();
    
    // 重置仿真
    void reset();
    
    // ==================== 状态查询 ====================
    
    // 当前时间
    Float current_time() const { return current_time_; }
    int current_step() const { return current_step_; }
    
    // 仿真进度
    Float progress() const;
    bool is_running() const { return is_running_; }
    bool is_complete() const { return is_complete_; }
    
    // 时间参数
    Float dt() const;
    Float simulation_time() const { return simulation_time_; }
    int total_steps() const;
    
    // 网格信息
    const Grid& grid() const { return *grid_; }
    int nx() const;
    int ny() const;
    int nz() const;
    
    // 场数据访问
    const Fields& fields() const { return *fields_; }
    Fields& fields() { return *fields_; }
    
    // ==================== 结果获取 ====================
    
    // 获取仿真结果
    SimulationResult get_results() const;
    
    // 获取场数据
    std::vector<Float> get_field(FieldComponent component) const;
    std::vector<Float> get_epsilon() const;
    
    // 获取通量数据
    std::vector<Float> get_flux(const std::string& monitor_name) const;
    
    // 获取场监视器数据
    std::vector<Float> get_field_data(const std::string& monitor_name) const;
    
    // 获取性能指标
    std::map<std::string, Float> get_metrics() const;
    
    // ==================== 高级功能 ====================
    
    // 设置进度回调
    using ProgressCallback = std::function<void(Float progress, int step)>;
    void set_progress_callback(ProgressCallback callback);
    
    // 设置输出回调
    using OutputCallback = std::function<void(const std::string&)>;
    void set_output_callback(OutputCallback callback);
    
    // 保存场数据到文件
    void save_fields(const std::string& filename) const;
    
    // 从文件加载场数据
    void load_fields(const std::string& filename);
    
private:
    // 核心组件
    std::unique_ptr<Grid> grid_;
    std::unique_ptr<Fields> fields_;
    std::unique_ptr<BoundaryManager> boundary_manager_;
    std::unique_ptr<SourceManager> source_manager_;
    std::unique_ptr<MonitorManager> monitor_manager_;
    
    // 配置
    Polarization polarization_;
    Float simulation_time_;
    int accuracy_;
    
    // 状态
    Float current_time_;
    int current_step_;
    bool is_running_;
    bool is_complete_;
    bool is_setup_;
    
    // 回调
    ProgressCallback progress_callback_;
    OutputCallback output_callback_;
    
    // 内部方法
    void update_E();  // 更新电场
    void update_H();  // 更新磁场
    void apply_sources();  // 应用光源
    void update_monitors();  // 更新监视器
    
    void output(const std::string& msg) const;
    
    // 验证
    void validate_setup() const;
};

} // namespace optics
