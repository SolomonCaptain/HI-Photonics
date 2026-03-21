#include "simulation.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace optics {

// ==================== FDTDSimulation 实现 ====================

FDTDSimulation::FDTDSimulation()
    : polarization_(Polarization::TM),
      simulation_time_(100.0),
      accuracy_(2),
      current_time_(0.0),
      current_step_(0),
      is_running_(false),
      is_complete_(false),
      is_setup_(false)
{
}

FDTDSimulation::~FDTDSimulation() = default;

void FDTDSimulation::set_grid(int nx, int ny, Float dx, Float dy) {
    grid_ = std::make_unique<Grid>(nx, ny, dx, dy);
    is_setup_ = false;
}

void FDTDSimulation::set_grid(int nx, int ny, int nz, Float dx, Float dy, Float dz) {
    grid_ = std::make_unique<Grid>(nx, ny, nz, dx, dy, dz);
    is_setup_ = false;
}

void FDTDSimulation::set_grid_from_config(const SimulationConfig& config) {
    Float dx = 1.0 / config.resolution;  // 微米
    
    int nx = static_cast<int>(config.cell_size.x * config.resolution);
    int ny = static_cast<int>(config.cell_size.y * config.resolution);
    int nz = config.cell_size.z > 0 ? static_cast<int>(config.cell_size.z * config.resolution) : 1;
    
    if (nz > 1) {
        set_grid(nx, ny, nz, dx, dx, dx);
    } else {
        set_grid(nx, ny, dx, dx);
    }
    
    set_boundary_x(config.boundary_x);
    set_boundary_y(config.boundary_y);
    set_boundary_z(config.boundary_z);
    
    simulation_time_ = config.simulation_time;
    accuracy_ = config.accuracy;
}

void FDTDSimulation::set_boundary_x(const BoundaryConfig& cfg) {
    if (grid_) grid_->set_boundary_x(cfg);
    if (boundary_manager_) boundary_manager_->set_boundary_x(cfg);
}

void FDTDSimulation::set_boundary_y(const BoundaryConfig& cfg) {
    if (grid_) grid_->set_boundary_y(cfg);
    if (boundary_manager_) boundary_manager_->set_boundary_y(cfg);
}

void FDTDSimulation::set_boundary_z(const BoundaryConfig& cfg) {
    if (grid_) grid_->set_boundary_z(cfg);
    if (boundary_manager_) boundary_manager_->set_boundary_z(cfg);
}

void FDTDSimulation::set_pml(int layers, Float sigma_max) {
    BoundaryConfig cfg = BoundaryConfig::pml(layers, sigma_max);
    set_boundary_x(cfg);
    set_boundary_y(cfg);
    set_boundary_z(cfg);
}

void FDTDSimulation::set_polarization(Polarization pol) {
    polarization_ = pol;
    is_setup_ = false;
}

void FDTDSimulation::set_epsilon(const std::vector<Float>& eps) {
    if (!fields_) {
        throw OpticsException("Fields not initialized. Call setup() first.");
    }
    fields_->set_epsilon(eps);
}

void FDTDSimulation::set_epsilon_from_density(const std::vector<Float>& density,
                                               Float eps_min, Float eps_max) {
    if (!fields_) {
        throw OpticsException("Fields not initialized. Call setup() first.");
    }
    fields_->set_epsilon_from_density(density, eps_min, eps_max);
}

void FDTDSimulation::add_material_block(const Vec3f& center, const Vec3f& size, Float epsilon) {
    if (!fields_ || !grid_) {
        throw OpticsException("Fields not initialized. Call setup() first.");
    }
    
    Float dx = grid_->dx();
    Float dy = grid_->dy();
    
    int i_start = std::max(0, static_cast<int>((center.x - size.x * 0.5) / dx));
    int i_end = std::min(grid_->nx() - 1, static_cast<int>((center.x + size.x * 0.5) / dx));
    int j_start = std::max(0, static_cast<int>((center.y - size.y * 0.5) / dy));
    int j_end = std::min(grid_->ny() - 1, static_cast<int>((center.y + size.y * 0.5) / dy));
    
    for (int j = j_start; j <= j_end; ++j) {
        for (int i = i_start; i <= i_end; ++i) {
            fields_->epsilon()(i, j) = epsilon;
        }
    }
    
    fields_->update_material_coefficients();
}

void FDTDSimulation::add_gaussian_source(Float wavelength, const Vec3f& center, const Vec3f& size,
                                          Float pulse_width) {
    auto source = std::make_shared<GaussianSource>(wavelength, center, size, pulse_width);
    add_source(source);
}

void FDTDSimulation::add_continuous_source(Float wavelength, const Vec3f& center, const Vec3f& size,
                                            Float ramp_time) {
    auto source = std::make_shared<ContinuousSource>(wavelength, center, size, ramp_time);
    add_source(source);
}

void FDTDSimulation::add_plane_wave_source(Float wavelength, const Vec3f& center, const Vec3f& size,
                                            Float angle, Float pulse_width) {
    auto source = std::make_shared<PlaneWaveSource>(wavelength, center, size, angle, pulse_width);
    add_source(source);
}

void FDTDSimulation::add_dipole_source(Float wavelength, const Vec3f& center,
                                        FieldComponent component, Float pulse_width) {
    auto source = std::make_shared<DipoleSource>(wavelength, center, component, pulse_width);
    add_source(source);
}

void FDTDSimulation::add_source(std::shared_ptr<Source> source) {
    if (!source_manager_) {
        source_manager_ = std::make_unique<SourceManager>(*grid_);
    }
    source_manager_->add_source(source);
}

void FDTDSimulation::clear_sources() {
    if (source_manager_) {
        source_manager_->clear_sources();
    }
}

void FDTDSimulation::add_flux_monitor(const std::string& name, const Vec3f& center, const Vec3f& size,
                                       const std::vector<Float>& frequencies) {
    auto monitor = std::make_shared<FluxMonitor>(name, center, size, frequencies);
    add_monitor(monitor);
}

void FDTDSimulation::add_field_monitor(const std::string& name, const Vec3f& center, const Vec3f& size,
                                        const std::vector<Float>& frequencies, FieldComponent component) {
    auto monitor = std::make_shared<FieldMonitor>(name, center, size, frequencies, component);
    add_monitor(monitor);
}

void FDTDSimulation::add_energy_monitor(const std::string& name, const Vec3f& center, const Vec3f& size) {
    auto monitor = std::make_shared<EnergyMonitor>(name, center, size);
    add_monitor(monitor);
}

void FDTDSimulation::add_monitor(std::shared_ptr<Monitor> monitor) {
    if (!monitor_manager_) {
        monitor_manager_ = std::make_unique<MonitorManager>(*grid_);
    }
    monitor_manager_->add_monitor(monitor);
}

void FDTDSimulation::clear_monitors() {
    if (monitor_manager_) {
        monitor_manager_->clear_monitors();
    }
}

void FDTDSimulation::setup() {
    if (!grid_) {
        throw OpticsException("Grid not set. Call set_grid() first.");
    }
    
    // 创建场对象
    fields_ = std::make_unique<Fields>(*grid_, polarization_);
    
    // 创建边界管理器
    boundary_manager_ = std::make_unique<BoundaryManager>(*grid_);
    
    // 创建光源管理器
    source_manager_ = std::make_unique<SourceManager>(*grid_);
    
    // 创建监视器管理器
    monitor_manager_ = std::make_unique<MonitorManager>(*grid_);
    
    // 重置状态
    current_time_ = 0.0;
    current_step_ = 0;
    is_running_ = false;
    is_complete_ = false;
    is_setup_ = true;
    
    output("FDTD simulation setup complete.");
    output("  Grid size: " + std::to_string(grid_->nx()) + " x " + std::to_string(grid_->ny()));
    output("  Time step: " + std::to_string(grid_->dt()));
    output("  Total steps: " + std::to_string(total_steps()));
}

void FDTDSimulation::validate_setup() const {
    if (!is_setup_) {
        throw OpticsException("Simulation not setup. Call setup() first.");
    }
    if (!grid_) {
        throw OpticsException("Grid not initialized.");
    }
    if (!fields_) {
        throw OpticsException("Fields not initialized.");
    }
}

void FDTDSimulation::run() {
    run(simulation_time_);
}

void FDTDSimulation::run(Float sim_time) {
    validate_setup();
    
    simulation_time_ = sim_time;
    int total = total_steps();
    
    output("Running simulation for " + std::to_string(sim_time) + " time units (" + 
           std::to_string(total) + " steps)...");
    
    is_running_ = true;
    
    while (current_step_ < total && is_running_) {
        step();
        
        // 进度回调
        if (progress_callback_) {
            progress_callback_(progress(), current_step_);
        }
    }
    
    is_running_ = false;
    is_complete_ = true;
    
    output("Simulation complete.");
}

void FDTDSimulation::step() {
    validate_setup();
    
    Float dt = grid_->dt();
    
    // 1. 更新 H 场
    update_H();
    
    // 2. 应用 H 边界条件
    boundary_manager_->apply_H_boundaries(*fields_);
    boundary_manager_->update_pml_H(*fields_);
    
    // 3. 更新 E 场
    update_E();
    
    // 4. 应用 E 边界条件
    boundary_manager_->apply_E_boundaries(*fields_);
    boundary_manager_->update_pml_E(*fields_);
    
    // 5. 应用光源
    apply_sources();
    
    // 6. 更新监视器
    update_monitors();
    
    // 更新时间
    current_time_ += dt;
    current_step_++;
}

void FDTDSimulation::update_E() {
    // TM 模式：Ez, Hx, Hy
    // Ez(i,j) = ca * Ez(i,j) + cb * (Hy(i,j) - Hy(i-1,j) - Hx(i,j) + Hx(i,j-1))

    if (polarization_ == Polarization::TM) {
        FieldArray* Ez = fields_->Ez();
        FieldArray* Hx = fields_->Hx();
        FieldArray* Hy = fields_->Hy();

        if (!Ez || !Hx || !Hy) return;

        int nx = grid_->nx();
        int ny = grid_->ny();

        // 使用内部区域
        int i_start = grid_->inner_x_start() + 1;
        int i_end = grid_->inner_x_end() - 1;
        int j_start = grid_->inner_y_start() + 1;
        int j_end = grid_->inner_y_end() - 1;

        for (int j = j_start; j < j_end; ++j) {
            for (int i = i_start; i < i_end; ++i) {
                Float curl = (*Hy)(i, j) - (*Hy)(i - 1, j) - (*Hx)(i, j) + (*Hx)(i, j - 1);
                (*Ez)(i, j) = fields_->ca()(i, j) * (*Ez)(i, j) +
                              fields_->cb()(i, j) * curl;
            }
        }
    }
    // TE 模式实现类似
}

void FDTDSimulation::update_H() {
    // TM 模式：
    // Hx(i,j) = da * Hx(i,j) + db * (-Ez(i,j+1) + Ez(i,j))
    // Hy(i,j) = da * Hy(i,j) + db * (Ez(i+1,j) - Ez(i,j))

    if (polarization_ == Polarization::TM) {
        FieldArray* Ez = fields_->Ez();
        FieldArray* Hx = fields_->Hx();
        FieldArray* Hy = fields_->Hy();

        if (!Ez || !Hx || !Hy) return;

        int nx = grid_->nx();
        int ny = grid_->ny();

        int i_start = grid_->inner_x_start();
        int i_end = grid_->inner_x_end();
        int j_start = grid_->inner_y_start();
        int j_end = grid_->inner_y_end();

        for (int j = j_start; j < j_end; ++j) {
            for (int i = i_start; i < i_end; ++i) {
                // Hx
                if (j < ny - 1) {
                    Float curl = -(*Ez)(i, j + 1) + (*Ez)(i, j);
                    (*Hx)(i, j) = fields_->da()(i, j) * (*Hx)(i, j) +
                                  fields_->db()(i, j) * curl;
                }

                // Hy
                if (i < nx - 1) {
                    Float curl = (*Ez)(i + 1, j) - (*Ez)(i, j);
                    (*Hy)(i, j) = fields_->da()(i, j) * (*Hy)(i, j) +
                                  fields_->db()(i, j) * curl;
                }
            }
        }
    }
}

void FDTDSimulation::apply_sources() {
    if (source_manager_) {
        source_manager_->apply_sources(*fields_, current_time_, grid_->dt());
    }
}

void FDTDSimulation::update_monitors() {
    if (monitor_manager_) {
        monitor_manager_->update_monitors(*fields_, current_time_);
    }
}

void FDTDSimulation::reset() {
    if (fields_) fields_->reset();
    if (monitor_manager_) monitor_manager_->reset_monitors();
    
    current_time_ = 0.0;
    current_step_ = 0;
    is_running_ = false;
    is_complete_ = false;
}

void FDTDSimulation::run_until_done() {
    run();
}

Float FDTDSimulation::progress() const {
    if (simulation_time_ <= 0) return 0.0;
    return current_time_ / simulation_time_;
}

Float FDTDSimulation::dt() const {
    return grid_ ? grid_->dt() : 0.0;
}

int FDTDSimulation::total_steps() const {
    if (!grid_ || simulation_time_ <= 0) return 0;
    return static_cast<int>(simulation_time_ / grid_->dt());
}

int FDTDSimulation::nx() const { return grid_ ? grid_->nx() : 0; }
int FDTDSimulation::ny() const { return grid_ ? grid_->ny() : 0; }
int FDTDSimulation::nz() const { return grid_ ? grid_->nz() : 0; }

SimulationResult FDTDSimulation::get_results() const {
    SimulationResult result;
    
    if (monitor_manager_) {
        result.flux = monitor_manager_->get_all_flux_data();
        result.fields = monitor_manager_->get_all_field_data();
    }
    
    // 添加性能指标
    result.metrics = get_metrics();
    
    return result;
}

std::vector<Float> FDTDSimulation::get_field(FieldComponent component) const {
    if (!fields_) return {};
    return fields_->get_field_data(component);
}

std::vector<Float> FDTDSimulation::get_epsilon() const {
    return get_field(FieldComponent::Epsilon);
}

std::vector<Float> FDTDSimulation::get_flux(const std::string& monitor_name) const {
    if (!monitor_manager_) return {};
    
    try {
        Monitor& m = monitor_manager_->get_monitor(monitor_name);
        if (m.type() == MonitorType::FLUX) {
            return m.get_flux_data();
        }
    } catch (...) {}
    
    return {};
}

std::vector<Float> FDTDSimulation::get_field_data(const std::string& monitor_name) const {
    if (!monitor_manager_) return {};
    
    try {
        Monitor& m = monitor_manager_->get_monitor(monitor_name);
        if (m.type() == MonitorType::FIELD) {
            return m.get_field_data();
        }
    } catch (...) {}
    
    return {};
}

std::map<std::string, Float> FDTDSimulation::get_metrics() const {
    std::map<std::string, Float> metrics;
    
    metrics["current_time"] = current_time_;
    metrics["current_step"] = static_cast<Float>(current_step_);
    metrics["progress"] = progress();
    
    if (fields_) {
        if (fields_->Ez()) {
            metrics["Ez_max"] = fields_->Ez()->max();
            metrics["Ez_min"] = fields_->Ez()->min();
            metrics["Ez_norm"] = fields_->Ez()->l2_norm();
        }
    }
    
    return metrics;
}

void FDTDSimulation::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = callback;
}

void FDTDSimulation::set_output_callback(OutputCallback callback) {
    output_callback_ = callback;
}

void FDTDSimulation::output(const std::string& msg) const {
    if (output_callback_) {
        output_callback_(msg);
    } else {
        std::cout << "[FDTD] " << msg << std::endl;
    }
}

void FDTDSimulation::save_fields(const std::string& filename) const {
    if (!fields_) return;
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw OpticsException("Cannot open file: " + filename);
    }
    
    // 写入网格信息
    int dims[3] = {grid_->nx(), grid_->ny(), grid_->nz()};
    file.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    
    // 写入场数据
    if (fields_->Ez()) {
        auto data = fields_->get_field_data(FieldComponent::Ez);
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(Float));
    }
}

void FDTDSimulation::load_fields(const std::string& filename) {
    // 实现加载逻辑
}

} // namespace optics
