#pragma once

#include "types.hpp"
#include "grid.hpp"
#include <vector>
#include <memory>

namespace optics {

/**
 * @brief PML 参数
 * 
 * 存储每个网格点的 PML 参数。
 */
struct PMLParams {
    std::vector<Float> sigma_e;  // 电导率（电场）
    std::vector<Float> sigma_m;  // 磁导率（磁场）
    std::vector<Float> kappa_e;  // kappa 参数（电场）
    std::vector<Float> kappa_m;  // kappa 参数（磁场）
    std::vector<Float> alpha;    // 低频吸收参数
};

/**
 * @brief 完美匹配层 (PML) 边界条件
 * 
 * 实现 CPML (Convolutional PML) 边界条件，用于吸收边界反射。
 */
class PML {
public:
    PML(const Grid& grid, int layers = 10, Float sigma_max = 0.8, Float kappa_max = 1.0);
    
    // 初始化 PML 参数
    void initialize();
    
    // 更新 PML 辅助场（需要在每个时间步调用）
    void update_E(Fields& fields);
    void update_H(Fields& fields);
    
    // 获取 PML 参数
    const PMLParams& params_x() const { return params_x_; }
    const PMLParams& params_y() const { return params_y_; }
    const PMLParams& params_z() const { return params_z_; }
    
    // 层数
    int layers() const { return layers_; }
    
    // 检查点是否在 PML 区域
    bool in_pml_x(int i) const;
    bool in_pml_y(int j) const;
    bool in_pml_z(int k) const;
    
private:
    const Grid& grid_;
    int layers_;
    Float sigma_max_;
    Float kappa_max_;
    
    // 各方向的 PML 参数
    PMLParams params_x_, params_y_, params_z_;
    
    // 辅助场（用于 CPML）
    std::vector<Float> psi_Ex_y_, psi_Ex_z_;
    std::vector<Float> psi_Ey_x_, psi_Ey_z_;
    std::vector<Float> psi_Ez_x_, psi_Ez_y_;
    std::vector<Float> psi_Hx_y_, psi_Hx_z_;
    std::vector<Float> psi_Hy_x_, psi_Hy_z_;
    std::vector<Float> psi_Hz_x_, psi_Hz_y_;
    
    // 计算单个点的 PML 参数
    Float compute_sigma(int dist, int layers) const;
    Float compute_kappa(int dist, int layers) const;
};

/**
 * @brief 边界条件管理器
 */
class BoundaryManager {
public:
    BoundaryManager(const Grid& grid);
    
    // 设置边界条件
    void set_boundary_x(const BoundaryConfig& cfg);
    void set_boundary_y(const BoundaryConfig& cfg);
    void set_boundary_z(const BoundaryConfig& cfg);
    
    // 应用边界条件
    void apply_E_boundaries(Fields& fields);
    void apply_H_boundaries(Fields& fields);
    
    // PML 更新
    void update_pml_E(Fields& fields);
    void update_pml_H(Fields& fields);
    
    // 获取 PML 对象
    PML& pml() { return *pml_; }
    const PML& pml() const { return *pml_; }
    
private:
    const Grid& grid_;
    BoundaryConfig boundary_x_, boundary_y_, boundary_z_;
    std::unique_ptr<PML> pml_;
    
    // 应用特定类型的边界
    void apply_pec_E(Fields& fields);
    void apply_pmc_H(Fields& fields);
    void apply_periodic_E(Fields& fields);
    void apply_periodic_H(Fields& fields);
};

} // namespace optics
