#pragma once

#include <vector>
#include <array>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <functional>

namespace optics {

// 物理常数
constexpr double C0 = 299792458.0;  // 光速 (m/s)
constexpr double EPS0 = 8.854187817e-12;  // 真空介电常数 (F/m)
constexpr double MU0 = 4.0 * 3.14159265358979323846e-7;  // 真空磁导率 (H/m)

// 常用双精度浮点类型
using Float = double;
using Complex = std::complex<Float>;

// 2D/3D 向量
template<typename T>
struct Vec2 {
    T x, y;
    Vec2() : x(0), y(0) {}
    Vec2(T x_, T y_) : x(x_), y(y_) {}
    
    Vec2 operator+(const Vec2& other) const { return Vec2(x + other.x, y + other.y); }
    Vec2 operator-(const Vec2& other) const { return Vec2(x - other.x, y - other.y); }
    Vec2 operator*(T scalar) const { return Vec2(x * scalar, y * scalar); }
    T dot(const Vec2& other) const { return x * other.x + y * other.y; }
};

template<typename T>
struct Vec3 {
    T x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
    Vec3 operator-(const Vec3& other) const { return Vec3(x - other.x, y - other.y, z - other.z); }
    Vec3 operator*(T scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    T dot(const Vec3& other) const { return x * other.x + y * other.y + z * other.z; }
};

using Vec2f = Vec2<Float>;
using Vec2i = Vec2<int>;
using Vec3f = Vec3<Float>;
using Vec3i = Vec3<int>;

// 边界条件类型
enum class BoundaryType {
    PML,        // 完美匹配层
    PERIODIC,   // 周期边界
    PMC,        // 完美磁导体
    PEC,        // 完美电导体
    BLOCH       // Bloch 边界
};

// 光源类型
enum class SourceType {
    GAUSSIAN,
    CONTINUOUS,
    PLANE_WAVE,
    DIPOLE,
    MODE_SOURCE
};

// 极化模式
enum class Polarization {
    TM,  // Ez, Hx, Hy
    TE   // Hz, Ex, Ey
};

// 监视器类型
enum class MonitorType {
    FIELD,      // 场分布
    FLUX,       // 通量
    MODE,       // 模式
    ENERGY      // 能量
};

// 场分量
enum class FieldComponent {
    Ex, Ey, Ez,
    Hx, Hy, Hz,
    Epsilon,  // 介电常数
    Mu        // 磁导率
};

// 边界配置
struct BoundaryConfig {
    BoundaryType type = BoundaryType::PML;
    int pml_layers = 10;           // PML 层数
    Float pml_sigma_max = 0.8;     // PML 最大电导率
    Float pml_kappa_max = 1.0;     // PML kappa 参数
    Float bloch_k = 0.0;           // Bloch 波矢
    
    static BoundaryConfig pml(int layers = 10, Float sigma_max = 0.8) {
        BoundaryConfig cfg;
        cfg.type = BoundaryType::PML;
        cfg.pml_layers = layers;
        cfg.pml_sigma_max = sigma_max;
        return cfg;
    }
    
    static BoundaryConfig periodic() {
        BoundaryConfig cfg;
        cfg.type = BoundaryType::PERIODIC;
        return cfg;
    }
    
    static BoundaryConfig pec() {
        BoundaryConfig cfg;
        cfg.type = BoundaryType::PEC;
        return cfg;
    }
    
    static BoundaryConfig pmc() {
        BoundaryConfig cfg;
        cfg.type = BoundaryType::PMC;
        return cfg;
    }
};

// 异常类
class OpticsException : public std::runtime_error {
public:
    explicit OpticsException(const std::string& msg) : std::runtime_error(msg) {}
};

} // namespace optics
