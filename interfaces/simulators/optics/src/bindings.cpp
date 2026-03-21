#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "simulation.hpp"

namespace py = pybind11;
using namespace optics;

// 辅助函数：numpy 数组转换
std::vector<Float> numpy_to_vector(const py::array_t<Float>& arr) {
    py::buffer_info buf = arr.request();
    Float* ptr = static_cast<Float*>(buf.ptr);
    return std::vector<Float>(ptr, ptr + buf.size);
}

py::array_t<Float> vector_to_numpy(const std::vector<Float>& vec) {
    return py::array_t<Float>(vec.size(), vec.data());
}

// Python 包装类
class PyFDTDSimulation {
public:
    PyFDTDSimulation() : sim_(std::make_unique<FDTDSimulation>()) {}
    
    // 网格设置
    void set_grid_2d(int nx, int ny, Float dx, Float dy) {
        sim_->set_grid(nx, ny, dx, dy);
    }
    
    void set_grid_3d(int nx, int ny, int nz, Float dx, Float dy, Float dz) {
        sim_->set_grid(nx, ny, nz, dx, dy, dz);
    }
    
    // 边界条件
    void set_pml(int layers = 10, Float sigma_max = 0.8) {
        sim_->set_pml(layers, sigma_max);
    }
    
    void set_periodic_boundary(const std::string& direction) {
        BoundaryConfig cfg = BoundaryConfig::periodic();
        if (direction == "x") sim_->set_boundary_x(cfg);
        else if (direction == "y") sim_->set_boundary_y(cfg);
        else if (direction == "z") sim_->set_boundary_z(cfg);
    }
    
    // 材料
    void set_epsilon(const py::array_t<Float>& eps) {
        sim_->set_epsilon(numpy_to_vector(eps));
    }
    
    void set_epsilon_from_density(const py::array_t<Float>& density, 
                                    Float eps_min, Float eps_max) {
        sim_->set_epsilon_from_density(numpy_to_vector(density), eps_min, eps_max);
    }
    
    // 光源
    void add_gaussian_source(Float wavelength, 
                             const std::tuple<Float, Float, Float>& center,
                             const std::tuple<Float, Float, Float>& size,
                             Float pulse_width = 10.0) {
        Vec3f c{std::get<0>(center), std::get<1>(center), std::get<2>(center)};
        Vec3f s{std::get<0>(size), std::get<1>(size), std::get<2>(size)};
        sim_->add_gaussian_source(wavelength, c, s, pulse_width);
    }
    
    void add_continuous_source(Float wavelength,
                               const std::tuple<Float, Float, Float>& center,
                               const std::tuple<Float, Float, Float>& size,
                               Float ramp_time = 50.0) {
        Vec3f c{std::get<0>(center), std::get<1>(center), std::get<2>(center)};
        Vec3f s{std::get<0>(size), std::get<1>(size), std::get<2>(size)};
        sim_->add_continuous_source(wavelength, c, s, ramp_time);
    }
    
    void add_plane_wave(Float wavelength,
                        const std::tuple<Float, Float, Float>& center,
                        const std::tuple<Float, Float, Float>& size,
                        Float angle = 0.0, Float pulse_width = 10.0) {
        Vec3f c{std::get<0>(center), std::get<1>(center), std::get<2>(center)};
        Vec3f s{std::get<0>(size), std::get<1>(size), std::get<2>(size)};
        sim_->add_plane_wave_source(wavelength, c, s, angle, pulse_width);
    }
    
    void add_dipole(Float wavelength,
                    const std::tuple<Float, Float, Float>& center,
                    const std::string& component = "Ez",
                    Float pulse_width = 10.0) {
        Vec3f c{std::get<0>(center), std::get<1>(center), std::get<2>(center)};
        FieldComponent comp = (component == "Ex") ? FieldComponent::Ex :
                              (component == "Ey") ? FieldComponent::Ey :
                              FieldComponent::Ez;
        sim_->add_dipole_source(wavelength, c, comp, pulse_width);
    }
    
    // 监视器
    void add_flux_monitor(const std::string& name,
                          const std::tuple<Float, Float, Float>& center,
                          const std::tuple<Float, Float, Float>& size,
                          const std::vector<Float>& frequencies) {
        Vec3f c{std::get<0>(center), std::get<1>(center), std::get<2>(center)};
        Vec3f s{std::get<0>(size), std::get<1>(size), std::get<2>(size)};
        sim_->add_flux_monitor(name, c, s, frequencies);
    }
    
    void add_field_monitor(const std::string& name,
                           const std::tuple<Float, Float, Float>& center,
                           const std::tuple<Float, Float, Float>& size,
                           const std::vector<Float>& frequencies,
                           const std::string& component = "Ez") {
        Vec3f c{std::get<0>(center), std::get<1>(center), std::get<2>(center)};
        Vec3f s{std::get<0>(size), std::get<1>(size), std::get<2>(size)};
        FieldComponent comp = (component == "Ex") ? FieldComponent::Ex :
                              (component == "Ey") ? FieldComponent::Ey :
                              (component == "Hx") ? FieldComponent::Hx :
                              (component == "Hy") ? FieldComponent::Hy :
                              (component == "Hz") ? FieldComponent::Hz :
                              FieldComponent::Ez;
        sim_->add_field_monitor(name, c, s, frequencies, comp);
    }
    
    // 仿真控制
    void setup() { sim_->setup(); }
    void run(Float time = -1.0) {
        if (time < 0) sim_->run();
        else sim_->run(time);
    }
    void step() { sim_->step(); }
    void reset() { sim_->reset(); }
    
    // 状态
    Float current_time() const { return sim_->current_time(); }
    int current_step() const { return sim_->current_step(); }
    Float progress() const { return sim_->progress(); }
    bool is_complete() const { return sim_->is_complete(); }
    Float dt() const { return sim_->dt(); }
    int total_steps() const { return sim_->total_steps(); }
    
    // 网格信息
    int nx() const { return sim_->nx(); }
    int ny() const { return sim_->ny(); }
    int nz() const { return sim_->nz(); }
    
    // 获取场数据
    py::array_t<Float> get_field(const std::string& component = "Ez") {
        FieldComponent comp = (component == "Ex") ? FieldComponent::Ex :
                              (component == "Ey") ? FieldComponent::Ey :
                              (component == "Hx") ? FieldComponent::Hx :
                              (component == "Hy") ? FieldComponent::Hy :
                              (component == "Hz") ? FieldComponent::Hz :
                              (component == "epsilon") ? FieldComponent::Epsilon :
                              FieldComponent::Ez;
        return vector_to_numpy(sim_->get_field(comp));
    }
    
    py::array_t<Float> get_epsilon() {
        return vector_to_numpy(sim_->get_epsilon());
    }
    
    py::array_t<Float> get_flux(const std::string& name) {
        return vector_to_numpy(sim_->get_flux(name));
    }
    
    // 设置仿真时间
    void set_simulation_time(Float time) {
        // 通过配置设置
    }
    
    // 获取结果
    py::dict get_results() {
        py::dict result;
        auto res = sim_->get_results();
        
        // 通量数据
        py::dict flux;
        for (const auto& [name, data] : res.flux) {
            flux[name.c_str()] = vector_to_numpy(data);
        }
        result["flux"] = flux;
        
        // 场数据
        py::dict fields;
        for (const auto& [name, data] : res.fields) {
            fields[name.c_str()] = vector_to_numpy(data);
        }
        result["fields"] = fields;
        
        // 性能指标
        py::dict metrics;
        for (const auto& [name, value] : res.metrics) {
            metrics[name.c_str()] = value;
        }
        result["metrics"] = metrics;
        
        return result;
    }
    
    // 设置极化
    void set_polarization(const std::string& pol) {
        if (pol == "TE") sim_->set_polarization(Polarization::TE);
        else sim_->set_polarization(Polarization::TM);
    }
    
private:
    std::unique_ptr<FDTDSimulation> sim_;
};

PYBIND11_MODULE(optics, m) {
    m.doc() = "FDTD optics simulator for photonics design";
    
    // 主仿真类
    py::class_<PyFDTDSimulation>(m, "FDTD")
        .def(py::init<>())
        
        // 网格
        .def("set_grid", &PyFDTDSimulation::set_grid_2d,
             "Set 2D grid", py::arg("nx"), py::arg("ny"), py::arg("dx"), py::arg("dy"))
        .def("set_grid_3d", &PyFDTDSimulation::set_grid_3d,
             "Set 3D grid", py::arg("nx"), py::arg("ny"), py::arg("nz"), 
             py::arg("dx"), py::arg("dy"), py::arg("dz"))
        
        // 边界
        .def("set_pml", &PyFDTDSimulation::set_pml,
             "Set PML boundary", py::arg("layers") = 10, py::arg("sigma_max") = 0.8)
        .def("set_periodic_boundary", &PyFDTDSimulation::set_periodic_boundary,
             "Set periodic boundary", py::arg("direction"))
        
        // 材料
        .def("set_epsilon", &PyFDTDSimulation::set_epsilon,
             "Set permittivity distribution", py::arg("epsilon"))
        .def("set_epsilon_from_density", &PyFDTDSimulation::set_epsilon_from_density,
             "Set permittivity from density [0,1]", 
             py::arg("density"), py::arg("eps_min"), py::arg("eps_max"))
        
        // 光源
        .def("add_gaussian_source", &PyFDTDSimulation::add_gaussian_source,
             "Add Gaussian pulse source",
             py::arg("wavelength"), py::arg("center"), py::arg("size"),
             py::arg("pulse_width") = 10.0)
        .def("add_continuous_source", &PyFDTDSimulation::add_continuous_source,
             "Add continuous wave source",
             py::arg("wavelength"), py::arg("center"), py::arg("size"),
             py::arg("ramp_time") = 50.0)
        .def("add_plane_wave", &PyFDTDSimulation::add_plane_wave,
             "Add plane wave source",
             py::arg("wavelength"), py::arg("center"), py::arg("size"),
             py::arg("angle") = 0.0, py::arg("pulse_width") = 10.0)
        .def("add_dipole", &PyFDTDSimulation::add_dipole,
             "Add dipole source",
             py::arg("wavelength"), py::arg("center"),
             py::arg("component") = "Ez", py::arg("pulse_width") = 10.0)
        
        // 监视器
        .def("add_flux_monitor", &PyFDTDSimulation::add_flux_monitor,
             "Add flux monitor",
             py::arg("name"), py::arg("center"), py::arg("size"), py::arg("frequencies"))
        .def("add_field_monitor", &PyFDTDSimulation::add_field_monitor,
             "Add field monitor",
             py::arg("name"), py::arg("center"), py::arg("size"),
             py::arg("frequencies"), py::arg("component") = "Ez")
        
        // 仿真控制
        .def("setup", &PyFDTDSimulation::setup, "Setup simulation")
        .def("run", &PyFDTDSimulation::run, "Run simulation", py::arg("time") = -1.0)
        .def("step", &PyFDTDSimulation::step, "Run one time step")
        .def("reset", &PyFDTDSimulation::reset, "Reset simulation")
        
        // 状态
        .def_property_readonly("current_time", &PyFDTDSimulation::current_time)
        .def_property_readonly("current_step", &PyFDTDSimulation::current_step)
        .def_property_readonly("progress", &PyFDTDSimulation::progress)
        .def_property_readonly("is_complete", &PyFDTDSimulation::is_complete)
        .def_property_readonly("dt", &PyFDTDSimulation::dt)
        .def_property_readonly("total_steps", &PyFDTDSimulation::total_steps)
        
        // 网格信息
        .def_property_readonly("nx", &PyFDTDSimulation::nx)
        .def_property_readonly("ny", &PyFDTDSimulation::ny)
        .def_property_readonly("nz", &PyFDTDSimulation::nz)
        
        // 数据获取
        .def("get_field", &PyFDTDSimulation::get_field,
             "Get field data", py::arg("component") = "Ez")
        .def("get_epsilon", &PyFDTDSimulation::get_epsilon, "Get permittivity")
        .def("get_flux", &PyFDTDSimulation::get_flux,
             "Get flux data", py::arg("name"))
        .def("get_results", &PyFDTDSimulation::get_results, "Get all results")
        
        // 设置
        .def("set_polarization", &PyFDTDSimulation::set_polarization,
             "Set polarization mode (TM/TE)", py::arg("mode"))
        ;
    
    // 版本信息
    m.attr("__version__") = "0.1.0";
}
