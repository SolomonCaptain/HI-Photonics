#include "grid.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>

namespace optics {

// ==================== Grid 实现 ====================

Grid::Grid(int nx, int ny, Float dx, Float dy)
    : nx_(nx), ny_(ny), nz_(1), dx_(dx), dy_(dy), dz_(0), dimensions_(2)
{
    compute_dt();
}

Grid::Grid(int nx, int ny, int nz, Float dx, Float dy, Float dz)
    : nx_(nx), ny_(ny), nz_(nz), dx_(dx), dy_(dy), dz_(dz), dimensions_(3)
{
    compute_dt();
}

int Grid::total_cells() const {
    return nx_ * ny_ * nz_;
}

void Grid::compute_dt() {
    // CFL 条件：dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
    // 这里使用归一化单位，c = 1
    Float sum = 0.0;
    if (dx_ > 0) sum += 1.0 / (dx_ * dx_);
    if (dy_ > 0) sum += 1.0 / (dy_ * dy_);
    if (dz_ > 0) sum += 1.0 / (dz_ * dz_);
    
    if (sum > 0) {
        // 稳定性因子，使用 0.99 作为安全裕度
        dt_ = 0.99 / std::sqrt(sum);
    } else {
        dt_ = 0.01;  // 默认值
    }
}

Vec2f Grid::cell_center(int i, int j) const {
    return Vec2f(i * dx_ + 0.5 * dx_, j * dy_ + 0.5 * dy_);
}

Vec3f Grid::cell_center(int i, int j, int k) const {
    return Vec3f(i * dx_ + 0.5 * dx_, j * dy_ + 0.5 * dy_, k * dz_ + 0.5 * dz_);
}

int Grid::inner_x_start() const {
    if (boundary_x_.type == BoundaryType::PML) {
        return boundary_x_.pml_layers;
    }
    return 0;
}

int Grid::inner_x_end() const {
    if (boundary_x_.type == BoundaryType::PML) {
        return nx_ - boundary_x_.pml_layers;
    }
    return nx_;
}

int Grid::inner_y_start() const {
    if (boundary_y_.type == BoundaryType::PML) {
        return boundary_y_.pml_layers;
    }
    return 0;
}

int Grid::inner_y_end() const {
    if (boundary_y_.type == BoundaryType::PML) {
        return ny_ - boundary_y_.pml_layers;
    }
    return ny_;
}

int Grid::inner_z_start() const {
    if (dimensions_ == 2) return 0;
    if (boundary_z_.type == BoundaryType::PML) {
        return boundary_z_.pml_layers;
    }
    return 0;
}

int Grid::inner_z_end() const {
    if (dimensions_ == 2) return 1;
    if (boundary_z_.type == BoundaryType::PML) {
        return nz_ - boundary_z_.pml_layers;
    }
    return nz_;
}

// ==================== FieldArray 实现 ====================

FieldArray::FieldArray(int nx, int ny)
    : nx_(nx), ny_(ny), nz_(1)
{
    data_.resize(nx * ny, 0.0);
}

FieldArray::FieldArray(int nx, int ny, int nz)
    : nx_(nx), ny_(ny), nz_(nz)
{
    data_.resize(nx * ny * nz, 0.0);
}

Float& FieldArray::operator()(int i, int j) {
    return data_[i + j * nx_];
}

Float FieldArray::operator()(int i, int j) const {
    return data_[i + j * nx_];
}

Float& FieldArray::operator()(int i, int j, int k) {
    return data_[i + j * nx_ + k * nx_ * ny_];
}

Float FieldArray::operator()(int i, int j, int k) const {
    return data_[i + j * nx_ + k * nx_ * ny_];
}

void FieldArray::fill(Float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void FieldArray::fill_random(Float min_val, Float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Float> dist(min_val, max_val);
    for (auto& val : data_) {
        val = dist(gen);
    }
}

Float FieldArray::max() const {
    return *std::max_element(data_.begin(), data_.end());
}

Float FieldArray::min() const {
    return *std::min_element(data_.begin(), data_.end());
}

Float FieldArray::sum() const {
    Float s = 0.0;
    for (const auto& val : data_) {
        s += val;
    }
    return s;
}

Float FieldArray::l2_norm() const {
    Float sum_sq = 0.0;
    for (const auto& val : data_) {
        sum_sq += val * val;
    }
    return std::sqrt(sum_sq);
}

void FieldArray::copy_from(const FieldArray& other) {
    std::copy(other.data_.begin(), other.data_.end(), data_.begin());
}

void FieldArray::copy_to(std::vector<Float>& dest) const {
    dest = data_;
}

void FieldArray::copy_from(const std::vector<Float>& src) {
    std::copy(src.begin(), src.end(), data_.begin());
}

// ==================== Fields 实现 ====================

Fields::Fields(const Grid& grid, Polarization pol)
    : grid_(grid),
      polarization_(pol),
      epsilon_(grid.nx(), grid.ny()),
      mu_(grid.nx(), grid.ny()),
      ca_(grid.nx(), grid.ny()),
      cb_(grid.nx(), grid.ny()),
      da_(grid.nx(), grid.ny()),
      db_(grid.nx(), grid.ny())
{
    if (grid_.is_2d()) {
        // 2D 情况
        if (pol == Polarization::TM) {
            // TM: Ez, Hx, Hy
            Ez_ = std::make_unique<FieldArray>(grid.nx(), grid.ny());
            Hx_ = std::make_unique<FieldArray>(grid.nx(), grid.ny());
            Hy_ = std::make_unique<FieldArray>(grid.nx(), grid.ny());
            Ex_ = nullptr;
            Ey_ = nullptr;
            Hz_ = nullptr;
        } else {
            // TE: Hz, Ex, Ey
            Hz_ = std::make_unique<FieldArray>(grid.nx(), grid.ny());
            Ex_ = std::make_unique<FieldArray>(grid.nx(), grid.ny());
            Ey_ = std::make_unique<FieldArray>(grid.nx(), grid.ny());
            Ez_ = nullptr;
            Hx_ = nullptr;
            Hy_ = nullptr;
        }
    } else {
        // 3D 情况：所有场分量
        Ex_ = std::make_unique<FieldArray>(grid.nx(), grid.ny(), grid.nz());
        Ey_ = std::make_unique<FieldArray>(grid.nx(), grid.ny(), grid.nz());
        Ez_ = std::make_unique<FieldArray>(grid.nx(), grid.ny(), grid.nz());
        Hx_ = std::make_unique<FieldArray>(grid.nx(), grid.ny(), grid.nz());
        Hy_ = std::make_unique<FieldArray>(grid.nx(), grid.ny(), grid.nz());
        Hz_ = std::make_unique<FieldArray>(grid.nx(), grid.ny(), grid.nz());
    }
    
    // 初始化材料参数
    epsilon_.fill(1.0);  // 真空
    mu_.fill(1.0);       // 非磁性材料
    
    update_material_coefficients();
}

void Fields::reset() {
    if (Ex_) Ex_->fill(0.0);
    if (Ey_) Ey_->fill(0.0);
    if (Ez_) Ez_->fill(0.0);
    if (Hx_) Hx_->fill(0.0);
    if (Hy_) Hy_->fill(0.0);
    if (Hz_) Hz_->fill(0.0);
}

void Fields::set_epsilon(const std::vector<Float>& eps) {
    epsilon_.copy_from(eps);
    update_material_coefficients();
}

void Fields::set_epsilon_from_density(const std::vector<Float>& density,
                                       Float eps_min, Float eps_max) {
    // 线性插值
    for (size_t i = 0; i < density.size() && i < epsilon_.size(); ++i) {
        epsilon_.data()[i] = eps_min + (eps_max - eps_min) * density[i];
    }
    update_material_coefficients();
}

void Fields::update_material_coefficients() {
    // 更新系数：
    // ca = (1 - sigma*dt/(2*eps)) / (1 + sigma*dt/(2*eps))
    // cb = dt/(eps*dx) / (1 + sigma*dt/(2*eps))
    // 对于无损材料，sigma = 0，所以：
    // ca = 1, cb = dt/(eps*dx)
    
    Float dt = grid_.dt();
    Float dx = grid_.dx();
    Float dy = grid_.dy();
    
    for (size_t i = 0; i < epsilon_.size(); ++i) {
        Float eps = epsilon_.data()[i];
        
        // 电场更新系数
        ca_.data()[i] = 1.0;
        cb_.data()[i] = dt / (eps * dx);
        
        // 磁场更新系数（假设 mu = 1）
        da_.data()[i] = 1.0;
        db_.data()[i] = dt / (1.0 * dx);
    }
}

std::vector<Float> Fields::get_field_data(FieldComponent comp) const {
    std::vector<Float> result;
    
    switch (comp) {
        case FieldComponent::Ex:
            if (Ex_) Ex_->copy_to(result);
            break;
        case FieldComponent::Ey:
            if (Ey_) Ey_->copy_to(result);
            break;
        case FieldComponent::Ez:
            if (Ez_) Ez_->copy_to(result);
            break;
        case FieldComponent::Hx:
            if (Hx_) Hx_->copy_to(result);
            break;
        case FieldComponent::Hy:
            if (Hy_) Hy_->copy_to(result);
            break;
        case FieldComponent::Hz:
            if (Hz_) Hz_->copy_to(result);
            break;
        case FieldComponent::Epsilon:
            epsilon_.copy_to(result);
            break;
        case FieldComponent::Mu:
            mu_.copy_to(result);
            break;
    }
    
    return result;
}

void Fields::set_field_data(FieldComponent comp, const std::vector<Float>& data) {
    switch (comp) {
        case FieldComponent::Ex:
            if (Ex_) Ex_->copy_from(data);
            break;
        case FieldComponent::Ey:
            if (Ey_) Ey_->copy_from(data);
            break;
        case FieldComponent::Ez:
            if (Ez_) Ez_->copy_from(data);
            break;
        case FieldComponent::Hx:
            if (Hx_) Hx_->copy_from(data);
            break;
        case FieldComponent::Hy:
            if (Hy_) Hy_->copy_from(data);
            break;
        case FieldComponent::Hz:
            if (Hz_) Hz_->copy_from(data);
            break;
        case FieldComponent::Epsilon:
            epsilon_.copy_from(data);
            update_material_coefficients();
            break;
        case FieldComponent::Mu:
            mu_.copy_from(data);
            break;
    }
}

} // namespace optics
