#include "source.hpp"
#include <cmath>
#include <algorithm>

namespace optics {

// ==================== Source 基类实现 ====================

Source::Source(SourceType type, Float wavelength, const Vec3f& center, const Vec3f& size)
    : type_(type),
      wavelength_(wavelength),
      center_(center),
      size_(size),
      direction_(1),
      component_(FieldComponent::Ez),
      amplitude_scale_(1.0)
{
    set_wavelength(wavelength);
}

void Source::set_wavelength(Float wl) {
    wavelength_ = wl;
    frequency_ = 1.0 / wl;  // 归一化单位
}

bool Source::contains(Float x, Float y, Float z) const {
    Float half_x = size_.x * 0.5;
    Float half_y = size_.y * 0.5;
    Float half_z = size_.z * 0.5;
    
    return (x >= center_.x - half_x && x <= center_.x + half_x &&
            y >= center_.y - half_y && y <= center_.y + half_y &&
            z >= center_.z - half_z && z <= center_.z + half_z);
}

bool Source::contains(Float x, Float y) const {
    Float half_x = size_.x * 0.5;
    Float half_y = size_.y * 0.5;
    
    return (x >= center_.x - half_x && x <= center_.x + half_x &&
            y >= center_.y - half_y && y <= center_.y + half_y);
}

// ==================== GaussianSource 实现 ====================

GaussianSource::GaussianSource(Float wavelength, const Vec3f& center, const Vec3f& size,
                               Float pulse_width, Float cutoff)
    : Source(SourceType::GAUSSIAN, wavelength, center, size),
      pulse_width_(pulse_width),
      cutoff_(cutoff)
{
    update_t0();
}

void GaussianSource::update_t0() {
    // t0 设置为脉冲峰值时间
    // tau 是脉冲宽度参数
    tau_ = pulse_width_ / (2.0 * std::sqrt(std::log(2.0)));
    t0_ = cutoff_ * tau_;  // 确保起始时幅度足够小
}

Float GaussianSource::amplitude(Float t, Float dt) const {
    // 高斯脉冲：exp(-(t-t0)^2 / (2*tau^2)) * sin(2*pi*f*(t-t0))
    Float t_rel = t - t0_;
    Float envelope = std::exp(-t_rel * t_rel / (2.0 * tau_ * tau_));
    Float carrier = std::sin(2.0 * M_PI * frequency_ * t);
    
    return amplitude_scale_ * envelope * carrier;
}

void GaussianSource::apply(Fields& fields, Float t, Float dt) {
    Float amp = amplitude(t, dt);

    if (!fields.Ez()) return;

    const Grid& grid = fields.grid();
    Float dx = grid.dx();
    Float dy = grid.dy();

    // 遍历光源区域
    int i_start = std::max(0, static_cast<int>((center_.x - size_.x * 0.5) / dx));
    int i_end = std::min(grid.nx() - 1, static_cast<int>((center_.x + size_.x * 0.5) / dx));
    int j_start = std::max(0, static_cast<int>((center_.y - size_.y * 0.5) / dy));
    int j_end = std::min(grid.ny() - 1, static_cast<int>((center_.y + size_.y * 0.5) / dy));

    FieldArray* Ez = fields.Ez();
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            (*Ez)(i, j) += amp * dt;
        }
    }
}

// ==================== ContinuousSource 实现 ====================

ContinuousSource::ContinuousSource(Float wavelength, const Vec3f& center, const Vec3f& size,
                                   Float ramp_time)
    : Source(SourceType::CONTINUOUS, wavelength, center, size),
      ramp_time_(ramp_time)
{
}

Float ContinuousSource::amplitude(Float t, Float dt) const {
    // 连续波：sin(2*pi*f*t) 带有平滑开启
    Float ramp = 1.0;
    if (t < ramp_time_) {
        ramp = 0.5 * (1.0 - std::cos(M_PI * t / ramp_time_));
    }
    
    return amplitude_scale_ * ramp * std::sin(2.0 * M_PI * frequency_ * t);
}

void ContinuousSource::apply(Fields& fields, Float t, Float dt) {
    Float amp = amplitude(t, dt);

    if (!fields.Ez()) return;

    const Grid& grid = fields.grid();
    Float dx = grid.dx();
    Float dy = grid.dy();

    int i_start = std::max(0, static_cast<int>((center_.x - size_.x * 0.5) / dx));
    int i_end = std::min(grid.nx() - 1, static_cast<int>((center_.x + size_.x * 0.5) / dx));
    int j_start = std::max(0, static_cast<int>((center_.y - size_.y * 0.5) / dy));
    int j_end = std::min(grid.ny() - 1, static_cast<int>((center_.y + size_.y * 0.5) / dy));

    FieldArray* Ez = fields.Ez();
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            (*Ez)(i, j) += amp * dt;
        }
    }
}

// ==================== PlaneWaveSource 实现 ====================

PlaneWaveSource::PlaneWaveSource(Float wavelength, const Vec3f& center, const Vec3f& size,
                                 Float angle, Float pulse_width)
    : Source(SourceType::PLANE_WAVE, wavelength, center, size),
      angle_(angle),
      pulse_width_(pulse_width)
{
    set_angle(angle);
    
    tau_ = pulse_width_ / (2.0 * std::sqrt(std::log(2.0)));
    t0_ = 5.0 * tau_;
}

void PlaneWaveSource::set_angle(Float angle_deg) {
    angle_ = angle_deg;
    angle_rad_ = angle_deg * M_PI / 180.0;
}

Float PlaneWaveSource::amplitude(Float t, Float dt) const {
    Float t_rel = t - t0_;
    Float envelope = std::exp(-t_rel * t_rel / (2.0 * tau_ * tau_));
    Float carrier = std::sin(2.0 * M_PI * frequency_ * t);
    
    return amplitude_scale_ * envelope * carrier;
}

void PlaneWaveSource::apply(Fields& fields, Float t, Float dt) {
    Float amp = amplitude(t, dt);

    if (!fields.Ez()) return;

    const Grid& grid = fields.grid();
    Float dx = grid.dx();
    Float dy = grid.dy();

    // 计算波矢方向
    Float kx = 2.0 * M_PI * frequency_ * std::sin(angle_rad_);
    Float ky = 2.0 * M_PI * frequency_ * std::cos(angle_rad_);

    int i_start = std::max(0, static_cast<int>((center_.x - size_.x * 0.5) / dx));
    int i_end = std::min(grid.nx() - 1, static_cast<int>((center_.x + size_.x * 0.5) / dx));
    int j_start = std::max(0, static_cast<int>((center_.y - size_.y * 0.5) / dy));
    int j_end = std::min(grid.ny() - 1, static_cast<int>((center_.y + size_.y * 0.5) / dy));

    FieldArray* Ez = fields.Ez();
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            Float x = i * dx;
            Float y = j * dy;

            // 添加相位因子
            Float phase = kx * (x - center_.x) + ky * (y - center_.y);
            Float value = amp * std::cos(phase);

            (*Ez)(i, j) += value * dt;
        }
    }
}

// ==================== DipoleSource 实现 ====================

DipoleSource::DipoleSource(Float wavelength, const Vec3f& center,
                           FieldComponent component, Float pulse_width)
    : Source(SourceType::DIPOLE, wavelength, center, Vec3f(0, 0, 0)),
      pulse_width_(pulse_width)
{
    set_component(component);
    
    tau_ = pulse_width_ / (2.0 * std::sqrt(std::log(2.0)));
    t0_ = 5.0 * tau_;
}

Float DipoleSource::amplitude(Float t, Float dt) const {
    Float t_rel = t - t0_;
    Float envelope = std::exp(-t_rel * t_rel / (2.0 * tau_ * tau_));
    Float carrier = std::sin(2.0 * M_PI * frequency_ * t);
    
    return amplitude_scale_ * envelope * carrier;
}

void DipoleSource::apply(Fields& fields, Float t, Float dt) {
    Float amp = amplitude(t, dt);

    const Grid& grid = fields.grid();
    Float dx = grid.dx();
    Float dy = grid.dy();

    // 找到最近网格点
    int i = static_cast<int>(center_.x / dx);
    int j = static_cast<int>(center_.y / dy);

    i = std::max(0, std::min(grid.nx() - 1, i));
    j = std::max(0, std::min(grid.ny() - 1, j));

    // 根据分量类型设置场值
    switch (component_) {
        case FieldComponent::Ex:
            if (fields.Ex()) (*fields.Ex())(i, j) += amp * dt;
            break;
        case FieldComponent::Ey:
            if (fields.Ey()) (*fields.Ey())(i, j) += amp * dt;
            break;
        case FieldComponent::Ez:
            if (fields.Ez()) (*fields.Ez())(i, j) += amp * dt;
            break;
        default:
            break;
    }
}

// ==================== SourceManager 实现 ====================

void SourceManager::add_source(std::shared_ptr<Source> source) {
    sources_.push_back(source);
}

void SourceManager::remove_source(int index) {
    if (index >= 0 && static_cast<size_t>(index) < sources_.size()) {
        sources_.erase(sources_.begin() + index);
    }
}

void SourceManager::clear_sources() {
    sources_.clear();
}

Source& SourceManager::get_source(int index) {
    return *sources_.at(index);
}

void SourceManager::apply_sources(Fields& fields, Float t, Float dt) {
    for (auto& source : sources_) {
        source->apply(fields, t, dt);
    }
}

} // namespace optics
