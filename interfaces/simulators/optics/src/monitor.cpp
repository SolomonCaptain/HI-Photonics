#include "monitor.hpp"
#include <cmath>
#include <algorithm>
#include <complex>

namespace optics {

// ==================== Monitor 基类实现 ====================

Monitor::Monitor(const std::string& name, MonitorType type,
                 const Vec3f& center, const Vec3f& size)
    : name_(name),
      type_(type),
      center_(center),
      size_(size),
      output_fields_(false)
{
}

void Monitor::set_frequencies(const std::vector<Float>& freqs) {
    frequencies_ = freqs;
}

void Monitor::set_wavelengths(const std::vector<Float>& wavelengths) {
    frequencies_.resize(wavelengths.size());
    for (size_t i = 0; i < wavelengths.size(); ++i) {
        frequencies_[i] = 1.0 / wavelengths[i];
    }
}

bool Monitor::contains(Float x, Float y, Float z) const {
    Float half_x = size_.x * 0.5;
    Float half_y = size_.y * 0.5;
    Float half_z = size_.z * 0.5;
    
    return (x >= center_.x - half_x && x <= center_.x + half_x &&
            y >= center_.y - half_y && y <= center_.y + half_y &&
            (size_.z == 0 || (z >= center_.z - half_z && z <= center_.z + half_z)));
}

bool Monitor::contains(Float x, Float y) const {
    Float half_x = size_.x * 0.5;
    Float half_y = size_.y * 0.5;
    
    return (x >= center_.x - half_x && x <= center_.x + half_x &&
            y >= center_.y - half_y && y <= center_.y + half_y);
}

// ==================== FluxMonitor 实现 ====================

FluxMonitor::FluxMonitor(const std::string& name, const Vec3f& center, const Vec3f& size,
                         const std::vector<Float>& frequencies)
    : Monitor(name, MonitorType::FLUX, center, size)
{
    set_frequencies(frequencies);
    flux_accumulator_.resize(frequencies_.size(), Complex(0.0, 0.0));
    flux_result_.resize(frequencies_.size(), 0.0);
}

void FluxMonitor::update(const Fields& fields, Float t) {
    // 计算瞬时通量
    Float instant_flux = compute_instantaneous_flux(fields);
    
    // DFT 变换
    for (size_t i = 0; i < frequencies_.size(); ++i) {
        Float omega = 2.0 * M_PI * frequencies_[i];
        Complex phase = Complex(std::cos(omega * t), -std::sin(omega * t));
        flux_accumulator_[i] += instant_flux * phase;
    }
}

void FluxMonitor::reset() {
    std::fill(flux_accumulator_.begin(), flux_accumulator_.end(), Complex(0.0, 0.0));
    std::fill(flux_result_.begin(), flux_result_.end(), 0.0);
}

Float FluxMonitor::compute_instantaneous_flux(const Fields& fields) const {
    // 通量 = Re{E x H*}
    // 对于 2D TM 模式（Ez, Hx, Hy）：
    // Poynting 矢量的 y 分量：Sy = -Ez * Hx
    // Poynting 矢量的 x 分量：Sx = Ez * Hy
    
    if (!fields.Ez()) return 0.0;
    
    const Grid& grid = fields.grid();
    Float dx = grid.dx();
    Float dy = grid.dy();
    
    Float total_flux = 0.0;
    
    // 确定监视器区域
    int i_start = std::max(0, static_cast<int>((center_.x - size_.x * 0.5) / dx));
    int i_end = std::min(grid.nx() - 1, static_cast<int>((center_.x + size_.x * 0.5) / dx));
    int j_start = std::max(0, static_cast<int>((center_.y - size_.y * 0.5) / dy));
    int j_end = std::min(grid.ny() - 1, static_cast<int>((center_.y + size_.y * 0.5) / dy));
    
    // 计算通量积分
    // 假设监视器是 y 方向的平面（测量 x 方向的通量）
    for (int j = j_start; j <= j_end; ++j) {
        int i = (i_start + i_end) / 2;  // 取中间位置
        
        Float Ez_val = (*fields.Ez())(i, j);
        Float Hy_val = fields.Hy() ? (*fields.Hy())(i, j) : 0.0;
        Float Hx_val = fields.Hx() ? (*fields.Hx())(i, j) : 0.0;
        
        // Sx = Ez * Hy (x 方向通量)
        // Sy = -Ez * Hx (y 方向通量)
        total_flux += Ez_val * Hy_val * dy;
    }
    
    return total_flux;
}

std::vector<Float> FluxMonitor::get_flux_data() const {
    return flux_result_;
}

Float FluxMonitor::get_flux_at_freq(int freq_index) const {
    if (freq_index >= 0 && static_cast<size_t>(freq_index) < flux_result_.size()) {
        return flux_result_[freq_index];
    }
    return 0.0;
}

// ==================== FieldMonitor 实现 ====================

FieldMonitor::FieldMonitor(const std::string& name, const Vec3f& center, const Vec3f& size,
                           const std::vector<Float>& frequencies, FieldComponent component)
    : Monitor(name, MonitorType::FIELD, center, size),
      component_(component)
{
    set_frequencies(frequencies);
}

void FieldMonitor::compute_grid_indices(const Grid& grid) {
    Float dx = grid.dx();
    Float dy = grid.dy();
    
    indices_i_.clear();
    indices_j_.clear();
    indices_k_.clear();
    
    int i_start = std::max(0, static_cast<int>((center_.x - size_.x * 0.5) / dx));
    int i_end = std::min(grid.nx() - 1, static_cast<int>((center_.x + size_.x * 0.5) / dx));
    int j_start = std::max(0, static_cast<int>((center_.y - size_.y * 0.5) / dy));
    int j_end = std::min(grid.ny() - 1, static_cast<int>((center_.y + size_.y * 0.5) / dy));
    
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            indices_i_.push_back(i);
            indices_j_.push_back(j);
            indices_k_.push_back(0);
        }
    }
    
    // 初始化 DFT 数据
    size_t num_points = indices_i_.size();
    field_dft_.resize(num_points * frequencies_.size(), Complex(0.0, 0.0));
    field_real_.resize(num_points * frequencies_.size(), 0.0);
    field_imag_.resize(num_points * frequencies_.size(), 0.0);
}

void FieldMonitor::update(const Fields& fields, Float t) {
    if (indices_i_.empty()) {
        compute_grid_indices(fields.grid());
    }
    
    // 获取场数据指针
    const FieldArray* field_ptr = nullptr;
    switch (component_) {
        case FieldComponent::Ex: field_ptr = fields.Ex().get(); break;
        case FieldComponent::Ey: field_ptr = fields.Ey().get(); break;
        case FieldComponent::Ez: field_ptr = fields.Ez().get(); break;
        case FieldComponent::Hx: field_ptr = fields.Hx().get(); break;
        case FieldComponent::Hy: field_ptr = fields.Hy().get(); break;
        case FieldComponent::Hz: field_ptr = fields.Hz().get(); break;
        default: return;
    }
    
    if (!field_ptr) return;
    
    // DFT 变换
    for (size_t k = 0; k < frequencies_.size(); ++k) {
        Float omega = 2.0 * M_PI * frequencies_[k];
        Complex phase = Complex(std::cos(omega * t), -std::sin(omega * t));
        
        for (size_t n = 0; n < indices_i_.size(); ++n) {
            int i = indices_i_[n];
            int j = indices_j_[n];
            
            Float field_val = (*field_ptr)(i, j);
            size_t idx = n + k * indices_i_.size();
            field_dft_[idx] += field_val * phase;
        }
    }
}

void FieldMonitor::reset() {
    std::fill(field_dft_.begin(), field_dft_.end(), Complex(0.0, 0.0));
    std::fill(field_real_.begin(), field_real_.end(), 0.0);
    std::fill(field_imag_.begin(), field_imag_.end(), 0.0);
}

std::vector<Float> FieldMonitor::get_field_data() const {
    // 返回实部和虚部
    std::vector<Float> result;
    result.reserve(field_dft_.size() * 2);
    
    for (const auto& val : field_dft_) {
        result.push_back(val.real());
        result.push_back(val.imag());
    }
    
    return result;
}

std::array<int, 3> FieldMonitor::data_shape() const {
    int ni = static_cast<int>(indices_i_.size());
    int nf = static_cast<int>(frequencies_.size());
    return {ni, nf, 2};  // 点数, 频率数, 实部+虚部
}

// ==================== EnergyMonitor 实现 ====================

EnergyMonitor::EnergyMonitor(const std::string& name, const Vec3f& center, const Vec3f& size)
    : Monitor(name, MonitorType::ENERGY, center, size),
      total_energy_(0.0),
      electric_energy_(0.0),
      magnetic_energy_(0.0)
{
}

void EnergyMonitor::update(const Fields& fields, Float t) {
    electric_energy_ = compute_electric_energy(fields);
    magnetic_energy_ = compute_magnetic_energy(fields);
    total_energy_ = electric_energy_ + magnetic_energy_;
    
    energy_history_.push_back(total_energy_);
}

void EnergyMonitor::reset() {
    total_energy_ = 0.0;
    electric_energy_ = 0.0;
    magnetic_energy_ = 0.0;
    energy_history_.clear();
}

Float EnergyMonitor::compute_electric_energy(const Fields& fields) const {
    // U_e = 0.5 * eps * |E|^2
    Float energy = 0.0;
    
    const Grid& grid = fields.grid();
    Float dx = grid.dx();
    Float dy = grid.dy();
    Float dv = dx * dy;  // 单元面积
    
    int i_start = std::max(0, static_cast<int>((center_.x - size_.x * 0.5) / dx));
    int i_end = std::min(grid.nx() - 1, static_cast<int>((center_.x + size_.x * 0.5) / dx));
    int j_start = std::max(0, static_cast<int>((center_.y - size_.y * 0.5) / dy));
    int j_end = std::min(grid.ny() - 1, static_cast<int>((center_.y + size_.y * 0.5) / dy));
    
    if (fields.Ez()) {
        for (int j = j_start; j <= j_end; ++j) {
            for (int i = i_start; i <= i_end; ++i) {
                Float E = (*fields.Ez())(i, j);
                Float eps = fields.epsilon()(i, j);
                energy += 0.5 * eps * E * E * dv;
            }
        }
    }
    
    return energy;
}

Float EnergyMonitor::compute_magnetic_energy(const Fields& fields) const {
    // U_m = 0.5 * mu * |H|^2
    Float energy = 0.0;
    
    const Grid& grid = fields.grid();
    Float dx = grid.dx();
    Float dy = grid.dy();
    Float dv = dx * dy;
    
    int i_start = std::max(0, static_cast<int>((center_.x - size_.x * 0.5) / dx));
    int i_end = std::min(grid.nx() - 1, static_cast<int>((center_.x + size_.x * 0.5) / dx));
    int j_start = std::max(0, static_cast<int>((center_.y - size_.y * 0.5) / dy));
    int j_end = std::min(grid.ny() - 1, static_cast<int>((center_.y + size_.y * 0.5) / dy));
    
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            Float H_sq = 0.0;
            if (fields.Hx()) H_sq += (*fields.Hx())(i, j) * (*fields.Hx())(i, j);
            if (fields.Hy()) H_sq += (*fields.Hy())(i, j) * (*fields.Hy())(i, j);
            if (fields.Hz()) H_sq += (*fields.Hz())(i, j) * (*fields.Hz())(i, j);
            
            Float mu = fields.mu()(i, j);
            energy += 0.5 * mu * H_sq * dv;
        }
    }
    
    return energy;
}

// ==================== MonitorManager 实现 ====================

void MonitorManager::add_monitor(std::shared_ptr<Monitor> monitor) {
    monitors_.push_back(monitor);
}

void MonitorManager::remove_monitor(const std::string& name) {
    monitors_.erase(
        std::remove_if(monitors_.begin(), monitors_.end(),
            [&name](const std::shared_ptr<Monitor>& m) { return m->name() == name; }),
        monitors_.end()
    );
}

void MonitorManager::clear_monitors() {
    monitors_.clear();
}

Monitor& MonitorManager::get_monitor(const std::string& name) {
    for (auto& m : monitors_) {
        if (m->name() == name) return *m;
    }
    throw OpticsException("Monitor not found: " + name);
}

Monitor& MonitorManager::get_monitor(int index) {
    return *monitors_.at(index);
}

void MonitorManager::update_monitors(const Fields& fields, Float t) {
    for (auto& monitor : monitors_) {
        monitor->update(fields, t);
    }
}

void MonitorManager::reset_monitors() {
    for (auto& monitor : monitors_) {
        monitor->reset();
    }
}

std::map<std::string, std::vector<Float>> MonitorManager::get_all_flux_data() const {
    std::map<std::string, std::vector<Float>> result;
    
    for (const auto& monitor : monitors_) {
        if (monitor->type() == MonitorType::FLUX) {
            result[monitor->name()] = monitor->get_flux_data();
        }
    }
    
    return result;
}

std::map<std::string, std::vector<Float>> MonitorManager::get_all_field_data() const {
    std::map<std::string, std::vector<Float>> result;
    
    for (const auto& monitor : monitors_) {
        if (monitor->type() == MonitorType::FIELD) {
            result[monitor->name()] = monitor->get_field_data();
        }
    }
    
    return result;
}

} // namespace optics
