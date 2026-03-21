#pragma once

#include "types.hpp"
#include "grid.hpp"
#include <vector>
#include <complex>
#include <memory>
#include <functional>

namespace optics {

/**
 * @brief 监视器基类
 */
class Monitor {
public:
    Monitor(const std::string& name, MonitorType type, 
            const Vec3f& center, const Vec3f& size);
    virtual ~Monitor() = default;
    
    const std::string& name() const { return name_; }
    MonitorType type() const { return type_; }
    const Vec3f& center() const { return center_; }
    const Vec3f& size() const { return size_; }
    
    // 设置频率点
    void set_frequencies(const std::vector<Float>& freqs);
    void set_wavelengths(const std::vector<Float>& wavelengths);
    const std::vector<Float>& frequencies() const { return frequencies_; }
    
    // 是否输出场数据
    bool output_fields() const { return output_fields_; }
    void set_output_fields(bool val) { output_fields_ = val; }
    
    // 检查点是否在监视器区域内
    bool contains(Float x, Float y, Float z) const;
    bool contains(Float x, Float y) const;
    
    // 更新监视器（每个时间步调用）
    virtual void update(const Fields& fields, Float t) = 0;
    
    // 重置数据
    virtual void reset() = 0;
    
    // 获取数据
    virtual std::vector<Float> get_flux_data() const { return {}; }
    virtual std::vector<Float> get_field_data() const { return {}; }
    
protected:
    std::string name_;
    MonitorType type_;
    Vec3f center_;
    Vec3f size_;
    std::vector<Float> frequencies_;
    bool output_fields_;
};

/**
 * @brief 通量监视器
 * 
 * 计算通过某个平面的功率通量。
 */
class FluxMonitor : public Monitor {
public:
    FluxMonitor(const std::string& name, const Vec3f& center, const Vec3f& size,
                const std::vector<Float>& frequencies);
    
    void update(const Fields& fields, Float t) override;
    void reset() override;
    
    std::vector<Float> get_flux_data() const override;
    
    // 获取各频率的通量
    Float get_flux_at_freq(int freq_index) const;
    
private:
    // DFT 累加器
    std::vector<Complex> flux_accumulator_;  // 存储各频率的 DFT 结果
    std::vector<Float> flux_result_;         // 最终通量结果
    
    // 计算通量积分
    Float compute_instantaneous_flux(const Fields& fields) const;
};

/**
 * @brief 场监视器
 * 
 * 记录某个区域内的场分布。
 */
class FieldMonitor : public Monitor {
public:
    FieldMonitor(const std::string& name, const Vec3f& center, const Vec3f& size,
                 const std::vector<Float>& frequencies, FieldComponent component);
    
    void update(const Fields& fields, Float t) override;
    void reset() override;
    
    std::vector<Float> get_field_data() const override;
    
    FieldComponent component() const { return component_; }
    void set_component(FieldComponent comp) { component_ = comp; }
    
    // 获取场数据的形状
    std::array<int, 3> data_shape() const;
    
private:
    FieldComponent component_;
    
    // DFT 场数据
    std::vector<Complex> field_dft_;      // DFT 场数据
    std::vector<Float> field_real_;       // 实部
    std::vector<Float> field_imag_;       // 虚部
    
    // 网格索引
    std::vector<int> indices_i_, indices_j_, indices_k_;
    
    void compute_grid_indices(const Grid& grid);
};

/**
 * @brief 能量监视器
 * 
 * 计算某个区域内的电磁场能量。
 */
class EnergyMonitor : public Monitor {
public:
    EnergyMonitor(const std::string& name, const Vec3f& center, const Vec3f& size);
    
    void update(const Fields& fields, Float t) override;
    void reset() override;
    
    Float total_energy() const { return total_energy_; }
    Float electric_energy() const { return electric_energy_; }
    Float magnetic_energy() const { return magnetic_energy_; }
    
    // 能量历史
    const std::vector<Float>& energy_history() const { return energy_history_; }
    
private:
    Float total_energy_;
    Float electric_energy_;
    Float magnetic_energy_;
    std::vector<Float> energy_history_;
    
    Float compute_electric_energy(const Fields& fields) const;
    Float compute_magnetic_energy(const Fields& fields) const;
};

/**
 * @brief 监视器管理器
 */
class MonitorManager {
public:
    MonitorManager(const Grid& grid) : grid_(grid) {}
    
    // 添加监视器
    void add_monitor(std::shared_ptr<Monitor> monitor);
    
    // 移除监视器
    void remove_monitor(const std::string& name);
    void clear_monitors();
    
    // 获取监视器
    Monitor& get_monitor(const std::string& name);
    Monitor& get_monitor(int index);
    size_t num_monitors() const { return monitors_.size(); }
    
    // 更新所有监视器
    void update_monitors(const Fields& fields, Float t);
    
    // 重置所有监视器
    void reset_monitors();
    
    // 获取通量数据
    std::map<std::string, std::vector<Float>> get_all_flux_data() const;
    
    // 获取场数据
    std::map<std::string, std::vector<Float>> get_all_field_data() const;
    
private:
    const Grid& grid_;
    std::vector<std::shared_ptr<Monitor>> monitors_;
};

} // namespace optics
