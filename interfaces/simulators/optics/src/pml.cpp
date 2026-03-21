#include "pml.hpp"
#include <cmath>
#include <algorithm>

namespace optics {

// ==================== PML 实现 ====================

PML::PML(const Grid& grid, int layers, Float sigma_max, Float kappa_max)
    : grid_(grid), layers_(layers), sigma_max_(sigma_max), kappa_max_(kappa_max)
{
    initialize();
}

Float PML::compute_sigma(int dist, int layers) const {
    // 多项式分布：sigma = sigma_max * (dist / layers)^m
    // 通常 m = 3 或 4
    if (dist >= layers) return 0.0;
    
    Float ratio = static_cast<Float>(layers - dist) / static_cast<Float>(layers);
    return sigma_max_ * std::pow(ratio, 3);
}

Float PML::compute_kappa(int dist, int layers) const {
    if (dist >= layers) return 1.0;
    
    Float ratio = static_cast<Float>(layers - dist) / static_cast<Float>(layers);
    return 1.0 + (kappa_max_ - 1.0) * std::pow(ratio, 2);
}

void PML::initialize() {
    int nx = grid_.nx();
    int ny = grid_.ny();
    
    // 初始化 x 方向 PML 参数
    params_x_.sigma_e.resize(nx, 0.0);
    params_x_.sigma_m.resize(nx, 0.0);
    params_x_.kappa_e.resize(nx, 1.0);
    params_x_.kappa_m.resize(nx, 1.0);
    params_x_.alpha.resize(nx, 0.0);
    
    // 左边界
    for (int i = 0; i < layers_ && i < nx; ++i) {
        int dist = layers_ - i;
        params_x_.sigma_e[i] = compute_sigma(dist, layers_);
        params_x_.sigma_m[i] = params_x_.sigma_e[i];
        params_x_.kappa_e[i] = compute_kappa(dist, layers_);
        params_x_.kappa_m[i] = params_x_.kappa_e[i];
    }
    
    // 右边界
    for (int i = nx - layers_; i < nx; ++i) {
        int dist = i - (nx - layers_);
        params_x_.sigma_e[i] = compute_sigma(dist + 1, layers_);
        params_x_.sigma_m[i] = params_x_.sigma_e[i];
        params_x_.kappa_e[i] = compute_kappa(dist + 1, layers_);
        params_x_.kappa_m[i] = params_x_.kappa_e[i];
    }
    
    // 初始化 y 方向 PML 参数
    params_y_.sigma_e.resize(ny, 0.0);
    params_y_.sigma_m.resize(ny, 0.0);
    params_y_.kappa_e.resize(ny, 1.0);
    params_y_.kappa_m.resize(ny, 1.0);
    params_y_.alpha.resize(ny, 0.0);
    
    // 下边界
    for (int j = 0; j < layers_ && j < ny; ++j) {
        int dist = layers_ - j;
        params_y_.sigma_e[j] = compute_sigma(dist, layers_);
        params_y_.sigma_m[j] = params_y_.sigma_e[j];
        params_y_.kappa_e[j] = compute_kappa(dist, layers_);
        params_y_.kappa_m[j] = params_y_.kappa_e[j];
    }
    
    // 上边界
    for (int j = ny - layers_; j < ny; ++j) {
        int dist = j - (ny - layers_);
        params_y_.sigma_e[j] = compute_sigma(dist + 1, layers_);
        params_y_.sigma_m[j] = params_y_.sigma_e[j];
        params_y_.kappa_e[j] = compute_kappa(dist + 1, layers_);
        params_y_.kappa_m[j] = params_y_.kappa_e[j];
    }
    
    // 初始化 CPML 辅助场
    if (grid_.is_2d()) {
        int size = nx * ny;
        psi_Ex_y_.resize(size, 0.0);
        psi_Ey_x_.resize(size, 0.0);
        psi_Ez_x_.resize(size, 0.0);
        psi_Ez_y_.resize(size, 0.0);
        psi_Hx_y_.resize(size, 0.0);
        psi_Hy_x_.resize(size, 0.0);
        psi_Hz_x_.resize(size, 0.0);
        psi_Hz_y_.resize(size, 0.0);
    }
}

bool PML::in_pml_x(int i) const {
    return i < layers_ || i >= grid_.nx() - layers_;
}

bool PML::in_pml_y(int j) const {
    return j < layers_ || j >= grid_.ny() - layers_;
}

bool PML::in_pml_z(int k) const {
    if (grid_.is_2d()) return false;
    return k < layers_ || k >= grid_.nz() - layers_;
}

void PML::update_E(Fields& fields) {
    // 简化的 PML 更新：在 PML 区域应用衰减
    // 完整的 CPML 实现更复杂，这里使用简化版本
    
    Float dt = grid_.dt();
    
    int nx = grid_.nx();
    int ny = grid_.ny();
    
    // 对 PML 区域内的场应用衰减
    if (fields.Ez()) {
        auto& Ez = fields.Ez();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                Float sigma_x = params_x_.sigma_e[i];
                Float sigma_y = params_y_.sigma_e[j];
                Float kappa_x = params_x_.kappa_e[i];
                Float kappa_y = params_y_.kappa_e[j];
                
                if (sigma_x > 0 || sigma_y > 0) {
                    // 指数衰减
                    Float decay_x = std::exp(-sigma_x * dt / kappa_x);
                    Float decay_y = std::exp(-sigma_y * dt / kappa_y);
                    (*Ez)(i, j) *= decay_x * decay_y;
                }
            }
        }
    }
}

void PML::update_H(Fields& fields) {
    Float dt = grid_.dt();
    
    int nx = grid_.nx();
    int ny = grid_.ny();
    
    // 对 PML 区域内的磁场应用衰减
    if (fields.Hx()) {
        auto& Hx = fields.Hx();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                Float sigma_y = params_y_.sigma_m[j];
                Float kappa_y = params_y_.kappa_m[j];
                
                if (sigma_y > 0) {
                    Float decay = std::exp(-sigma_y * dt / kappa_y);
                    (*Hx)(i, j) *= decay;
                }
            }
        }
    }
    
    if (fields.Hy()) {
        auto& Hy = fields.Hy();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                Float sigma_x = params_x_.sigma_m[i];
                Float kappa_x = params_x_.kappa_m[i];
                
                if (sigma_x > 0) {
                    Float decay = std::exp(-sigma_x * dt / kappa_x);
                    (*Hy)(i, j) *= decay;
                }
            }
        }
    }
}

// ==================== BoundaryManager 实现 ====================

BoundaryManager::BoundaryManager(const Grid& grid)
    : grid_(grid)
{
    // 默认 PML 边界
    boundary_x_ = BoundaryConfig::pml(10);
    boundary_y_ = BoundaryConfig::pml(10);
    boundary_z_ = BoundaryConfig::pml(10);
    
    // 初始化 PML
    pml_ = std::make_unique<PML>(grid_, 10, 0.8, 1.0);
}

void BoundaryManager::set_boundary_x(const BoundaryConfig& cfg) {
    boundary_x_ = cfg;
}

void BoundaryManager::set_boundary_y(const BoundaryConfig& cfg) {
    boundary_y_ = cfg;
}

void BoundaryManager::set_boundary_z(const BoundaryConfig& cfg) {
    boundary_z_ = cfg;
}

void BoundaryManager::apply_E_boundaries(Fields& fields) {
    // 应用 PEC 边界
    if (boundary_x_.type == BoundaryType::PEC ||
        boundary_y_.type == BoundaryType::PEC) {
        apply_pec_E(fields);
    }
    
    // 应用周期边界
    if (boundary_x_.type == BoundaryType::PERIODIC ||
        boundary_y_.type == BoundaryType::PERIODIC) {
        apply_periodic_E(fields);
    }
}

void BoundaryManager::apply_H_boundaries(Fields& fields) {
    // 应用 PMC 边界
    if (boundary_x_.type == BoundaryType::PMC ||
        boundary_y_.type == BoundaryType::PMC) {
        apply_pmc_H(fields);
    }
    
    // 应用周期边界
    if (boundary_x_.type == BoundaryType::PERIODIC ||
        boundary_y_.type == BoundaryType::PERIODIC) {
        apply_periodic_H(fields);
    }
}

void BoundaryManager::update_pml_E(Fields& fields) {
    if (boundary_x_.type == BoundaryType::PML ||
        boundary_y_.type == BoundaryType::PML) {
        pml_->update_E(fields);
    }
}

void BoundaryManager::update_pml_H(Fields& fields) {
    if (boundary_x_.type == BoundaryType::PML ||
        boundary_y_.type == BoundaryType::PML) {
        pml_->update_H(fields);
    }
}

void BoundaryManager::apply_pec_E(Fields& fields) {
    // PEC: 电场切向分量为零
    int nx = grid_.nx();
    int ny = grid_.ny();
    
    if (fields.Ez()) {
        auto& Ez = fields.Ez();
        
        // x 边界
        if (boundary_x_.type == BoundaryType::PEC) {
            for (int j = 0; j < ny; ++j) {
                (*Ez)(0, j) = 0.0;
                (*Ez)(nx - 1, j) = 0.0;
            }
        }
        
        // y 边界
        if (boundary_y_.type == BoundaryType::PEC) {
            for (int i = 0; i < nx; ++i) {
                (*Ez)(i, 0) = 0.0;
                (*Ez)(i, ny - 1) = 0.0;
            }
        }
    }
}

void BoundaryManager::apply_pmc_H(Fields& fields) {
    // PMC: 磁场切向分量为零
    int nx = grid_.nx();
    int ny = grid_.ny();
    
    // x 边界
    if (boundary_x_.type == BoundaryType::PMC) {
        if (fields.Hy()) {
            for (int j = 0; j < ny; ++j) {
                (*fields.Hy())(0, j) = 0.0;
                (*fields.Hy())(nx - 1, j) = 0.0;
            }
        }
    }
    
    // y 边界
    if (boundary_y_.type == BoundaryType::PMC) {
        if (fields.Hx()) {
            for (int i = 0; i < nx; ++i) {
                (*fields.Hx())(i, 0) = 0.0;
                (*fields.Hx())(i, ny - 1) = 0.0;
            }
        }
    }
}

void BoundaryManager::apply_periodic_E(Fields& fields) {
    int nx = grid_.nx();
    int ny = grid_.ny();
    
    if (fields.Ez()) {
        auto& Ez = fields.Ez();
        
        // x 方向周期
        if (boundary_x_.type == BoundaryType::PERIODIC) {
            for (int j = 0; j < ny; ++j) {
                (*Ez)(0, j) = (*Ez)(nx - 2, j);
                (*Ez)(nx - 1, j) = (*Ez)(1, j);
            }
        }
        
        // y 方向周期
        if (boundary_y_.type == BoundaryType::PERIODIC) {
            for (int i = 0; i < nx; ++i) {
                (*Ez)(i, 0) = (*Ez)(i, ny - 2);
                (*Ez)(i, ny - 1) = (*Ez)(i, 1);
            }
        }
    }
}

void BoundaryManager::apply_periodic_H(Fields& fields) {
    int nx = grid_.nx();
    int ny = grid_.ny();
    
    if (fields.Hx()) {
        auto& Hx = fields.Hx();
        
        if (boundary_y_.type == BoundaryType::PERIODIC) {
            for (int i = 0; i < nx; ++i) {
                (*Hx)(i, 0) = (*Hx)(i, ny - 2);
                (*Hx)(i, ny - 1) = (*Hx)(i, 1);
            }
        }
    }
    
    if (fields.Hy()) {
        auto& Hy = fields.Hy();
        
        if (boundary_x_.type == BoundaryType::PERIODIC) {
            for (int j = 0; j < ny; ++j) {
                (*Hy)(0, j) = (*Hy)(nx - 2, j);
                (*Hy)(nx - 1, j) = (*Hy)(1, j);
            }
        }
    }
}

} // namespace optics
