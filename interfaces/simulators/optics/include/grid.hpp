#pragma once

#include "types.hpp"
#include <vector>
#include <array>
#include <memory>

namespace optics {

/**
 * @brief Yee 网格类
 * 
 * 实现 FDTD 的 Yee 网格，管理空间离散化和场分量的位置。
 * 
 * Yee 网格结构（2D TM 模式）：
 * 
 *      Hy(i,j+1/2) ---- Hx(i+1/2,j+1/2) ---- Hy(i+1,j+1/2)
 *           |                                   |
 *           |              Ez(i+1/2,j+1/2)      |
 *           |                                   |
 *      Hy(i,j) -------- Hx(i+1/2,j) -------- Hy(i+1,j)
 *           |                                   |
 *           |              Ez(i+1/2,j)         |
 *           |                                   |
 *      Hy(i,j-1/2) ---- Hx(i+1/2,j-1/2) ---- Hy(i+1,j-1/2)
 */
class Grid {
public:
    // 构造函数
    Grid(int nx, int ny, Float dx, Float dy);
    Grid(int nx, int ny, int nz, Float dx, Float dy, Float dz);
    
    // 网格尺寸
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    int total_cells() const;
    
    // 空间步长
    Float dx() const { return dx_; }
    Float dy() const { return dy_; }
    Float dz() const { return dz_; }
    
    // 网格分辨率（像素/微米）
    Float resolution() const { return 1.0 / dx_; }  // 假设 dx 单位为微米
    
    // 物理尺寸
    Float size_x() const { return nx_ * dx_; }
    Float size_y() const { return ny_ * dy_; }
    Float size_z() const { return nz_ * dz_; }
    
    // 时间步长（CFL 条件）
    Float dt() const { return dt_; }
    void set_dt(Float dt) { dt_ = dt; }
    void compute_dt();  // 根据 CFL 条件计算
    
    // 维度
    int dimensions() const { return dimensions_; }
    bool is_2d() const { return dimensions_ == 2; }
    bool is_3d() const { return dimensions_ == 3; }
    
    // 坐标转换
    Vec2f cell_center(int i, int j) const;
    Vec3f cell_center(int i, int j, int k) const;
    
    // 边界设置
    void set_boundary_x(const BoundaryConfig& cfg) { boundary_x_ = cfg; }
    void set_boundary_y(const BoundaryConfig& cfg) { boundary_y_ = cfg; }
    void set_boundary_z(const BoundaryConfig& cfg) { boundary_z_ = cfg; }
    
    const BoundaryConfig& boundary_x() const { return boundary_x_; }
    const BoundaryConfig& boundary_y() const { return boundary_y_; }
    const BoundaryConfig& boundary_z() const { return boundary_z_; }
    
    // 内部区域（排除 PML）
    int inner_x_start() const;
    int inner_x_end() const;
    int inner_y_start() const;
    int inner_y_end() const;
    int inner_z_start() const;
    int inner_z_end() const;
    
    // 有效区域尺寸（排除边界）
    int inner_nx() const { return inner_x_end() - inner_x_start(); }
    int inner_ny() const { return inner_y_end() - inner_y_start(); }
    int inner_nz() const { return inner_z_end() - inner_z_start(); }
    
private:
    int nx_, ny_, nz_;       // 网格点数
    Float dx_, dy_, dz_;     // 空间步长（微米）
    Float dt_;               // 时间步长
    int dimensions_;         // 维度（2 或 3）
    
    BoundaryConfig boundary_x_, boundary_y_, boundary_z_;
};

/**
 * @brief 场分量数组
 * 
 * 管理电磁场各分量的数据存储和访问。
 */
class FieldArray {
public:
    FieldArray(int nx, int ny);
    FieldArray(int nx, int ny, int nz);
    
    // 数据访问（2D）
    Float& operator()(int i, int j);
    Float operator()(int i, int j) const;
    
    // 数据访问（3D）
    Float& operator()(int i, int j, int k);
    Float operator()(int i, int j, int k) const;
    
    // 原始数据访问
    Float* data() { return data_.data(); }
    const Float* data() const { return data_.data(); }
    
    // 尺寸
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }
    size_t size() const { return data_.size(); }
    
    // 填充
    void fill(Float value);
    void fill_random(Float min_val = -1.0, Float max_val = 1.0);
    
    // 数学运算
    Float max() const;
    Float min() const;
    Float sum() const;
    Float l2_norm() const;
    
    // 复制
    void copy_from(const FieldArray& other);
    void copy_to(std::vector<Float>& dest) const;
    void copy_from(const std::vector<Float>& src);
    
private:
    std::vector<Float> data_;
    int nx_, ny_, nz_;
};

/**
 * @brief 电磁场集合
 * 
 * 包含所有电磁场分量的集合。
 */
class Fields {
public:
    Fields(const Grid& grid, Polarization pol = Polarization::TM);
    
    // 场分量访问
    FieldArray& Ex() { return *Ex_; }
    FieldArray& Ey() { return *Ey_; }
    FieldArray& Ez() { return *Ez_; }
    FieldArray& Hx() { return *Hx_; }
    FieldArray& Hy() { return *Hy_; }
    FieldArray& Hz() { return *Hz_; }
    
    const FieldArray& Ex() const { return *Ex_; }
    const FieldArray& Ey() const { return *Ey_; }
    const FieldArray& Ez() const { return *Ez_; }
    const FieldArray& Hx() const { return *Hx_; }
    const FieldArray& Hy() const { return *Hy_; }
    const FieldArray& Hz() const { return *Hz_; }
    
    // 材料参数
    FieldArray& epsilon() { return epsilon_; }
    FieldArray& mu() { return mu_; }
    const FieldArray& epsilon() const { return epsilon_; }
    const FieldArray& mu() const { return mu_; }
    
    // 导数材料参数（用于更新方程）
    FieldArray& ca() { return ca_; }  // 电场更新系数 a
    FieldArray& cb() { return cb_; }  // 电场更新系数 b
    FieldArray& da() { return da_; }  // 磁场更新系数 a
    FieldArray& db() { return db_; }  // 磁场更新系数 b
    
    // 极化模式
    Polarization polarization() const { return polarization_; }
    
    // 网格引用
    const Grid& grid() const { return grid_; }
    
    // 重置所有场
    void reset();
    
    // 设置材料分布
    void set_epsilon(const std::vector<Float>& eps);
    void set_epsilon_from_density(const std::vector<Float>& density, 
                                   Float eps_min, Float eps_max);
    
    // 更新材料系数
    void update_material_coefficients();
    
    // 场数据导出（用于 Python）
    std::vector<Float> get_field_data(FieldComponent comp) const;
    void set_field_data(FieldComponent comp, const std::vector<Float>& data);
    
private:
    const Grid& grid_;
    Polarization polarization_;
    
    // 电场分量
    std::unique_ptr<FieldArray> Ex_, Ey_, Ez_;
    // 磁场分量
    std::unique_ptr<FieldArray> Hx_, Hy_, Hz_;
    
    // 材料参数
    FieldArray epsilon_;   // 相对介电常数
    FieldArray mu_;        // 相对磁导率
    
    // 更新系数（预先计算以提高效率）
    FieldArray ca_, cb_;   // 电场更新系数
    FieldArray da_, db_;   // 磁场更新系数
};

} // namespace optics
