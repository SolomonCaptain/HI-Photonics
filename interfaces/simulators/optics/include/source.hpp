#pragma once

#include "types.hpp"
#include "grid.hpp"
#include "grid.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace optics {

/**
 * @brief 光源基类
 */
class Source {
public:
    Source(SourceType type, Float wavelength, const Vec3f& center, const Vec3f& size);
    virtual ~Source() = default;
    
    // 光源类型
    SourceType type() const { return type_; }
    
    // 波长和频率
    Float wavelength() const { return wavelength_; }
    Float frequency() const { return frequency_; }
    void set_wavelength(Float wl);
    
    // 位置和尺寸
    const Vec3f& center() const { return center_; }
    const Vec3f& size() const { return size_; }
    void set_center(const Vec3f& c) { center_ = c; }
    void set_size(const Vec3f& s) { size_ = s; }
    
    // 方向和极化
    int direction() const { return direction_; }
    void set_direction(int dir) { direction_ = dir; }
    
    FieldComponent component() const { return component_; }
    void set_component(FieldComponent comp) { component_ = comp; }
    
    // 计算当前时间步的幅度
    virtual Float amplitude(Float t, Float dt) const = 0;
    
    // 检查点是否在光源区域内
    bool contains(Float x, Float y, Float z) const;
    bool contains(Float x, Float y) const;  // 2D
    
    // 应用光源到场
    virtual void apply(Fields& fields, Float t, Float dt) = 0;
    
protected:
    SourceType type_;
    Float wavelength_;
    Float frequency_;
    Vec3f center_;
    Vec3f size_;
    int direction_;          // +1 或 -1
    FieldComponent component_;
    Float amplitude_scale_;  // 幅度缩放因子
};

/**
 * @brief 高斯脉冲光源
 * 
 * 时间波形: exp(-(t-t0)^2 / (2*tau^2)) * sin(2*pi*f*t)
 */
class GaussianSource : public Source {
public:
    GaussianSource(Float wavelength, const Vec3f& center, const Vec3f& size,
                   Float pulse_width = 10.0, Float cutoff = 5.0);
    
    Float amplitude(Float t, Float dt) const override;
    void apply(Fields& fields, Float t, Float dt) override;
    
    // 参数设置
    void set_pulse_width(Float width) { pulse_width_ = width; update_t0(); }
    void set_cutoff(Float cutoff) { cutoff_ = cutoff; update_t0(); }
    
private:
    Float pulse_width_;  // 脉冲宽度（时间单位）
    Float cutoff_;       // 截断时间（标准差倍数）
    Float t0_;           // 脉冲中心时间
    Float tau_;          // 脉冲宽度参数
    
    void update_t0();
};

/**
 * @brief 连续波光源
 * 
 * 时间波形: sin(2*pi*f*t) 带有平滑开启
 */
class ContinuousSource : public Source {
public:
    ContinuousSource(Float wavelength, const Vec3f& center, const Vec3f& size,
                     Float ramp_time = 50.0);
    
    Float amplitude(Float t, Float dt) const override;
    void apply(Fields& fields, Float t, Float dt) override;
    
    void set_ramp_time(Float time) { ramp_time_ = time; }
    
private:
    Float ramp_time_;  // 平滑开启时间
};

/**
 * @brief 平面波光源
 * 
 * 沿特定方向传播的平面波。
 */
class PlaneWaveSource : public Source {
public:
    PlaneWaveSource(Float wavelength, const Vec3f& center, const Vec3f& size,
                    Float angle = 0.0, Float pulse_width = 10.0);
    
    Float amplitude(Float t, Float dt) const override;
    void apply(Fields& fields, Float t, Float dt) override;
    
    void set_angle(Float angle_deg);
    Float angle() const { return angle_; }
    
private:
    Float angle_;        // 入射角度（度）
    Float angle_rad_;    // 入射角度（弧度）
    Float pulse_width_;
    Float t0_, tau_;
};

/**
 * @brief 偶极子光源
 * 
 * 点偶极子源，用于模拟局部激励。
 */
class DipoleSource : public Source {
public:
    DipoleSource(Float wavelength, const Vec3f& center,
                 FieldComponent component = FieldComponent::Ez,
                 Float pulse_width = 10.0);
    
    Float amplitude(Float t, Float dt) const override;
    void apply(Fields& fields, Float t, Float dt) override;
    
private:
    Float pulse_width_;
    Float t0_, tau_;
};

/**
 * @brief 光源管理器
 */
class SourceManager {
public:
    SourceManager(const Grid& grid) : grid_(grid) {}
    
    // 添加光源
    void add_source(std::shared_ptr<Source> source);
    
    // 移除光源
    void remove_source(int index);
    void clear_sources();
    
    // 获取光源
    Source& get_source(int index);
    size_t num_sources() const { return sources_.size(); }
    
    // 应用所有光源
    void apply_sources(Fields& fields, Float t, Float dt);
    
private:
    const Grid& grid_;
    std::vector<std::shared_ptr<Source>> sources_;
};

} // namespace optics
