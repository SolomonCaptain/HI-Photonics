"""
优化约束模块

提供光子学逆向设计中的各种约束条件。

约束类别:
1. 几何约束 - 最小特征尺寸、曲率、连通性等
2. 物理约束 - 材料色散、热效应等
3. 制造约束 - 制造公差、鲁棒性等
"""

from .dispersion import (
    DispersionModel,
    MaterialProperties,
    MATERIAL_DATABASE,
    DispersionCalculator,
    DispersionConstraint,
    MultiWavelengthSimulator,
    WavelengthDemuxConstraint,
    get_material_index,
    list_available_materials,
)

from .manufacturing_tolerance import (
    ToleranceType,
    ManufacturingSpecs,
    PROCESS_SPECS,
    EdgeRoughnessModel,
    EtchBiasModel,
    RobustnessConstraint,
    CDVariationConstraint,
    DesignRuleCheck,
    get_process_specs,
    list_available_processes,
)

from .thermal import (
    ThermalConfig,
    HeatEquationSolver,
    TemperatureConstraint,
    ThermalStressConstraint,
    ThermoOpticConstraint,
    HeatDissipationConstraint,
    ThermalConstraint,
    create_thermal_constraint,
    compute_temperature_field,
    estimate_thermal_index_change,
)


__all__ = [
    # 色散相关
    'DispersionModel',
    'MaterialProperties',
    'MATERIAL_DATABASE',
    'DispersionCalculator',
    'DispersionConstraint',
    'MultiWavelengthSimulator',
    'WavelengthDemuxConstraint',
    'get_material_index',
    'list_available_materials',
    
    # 制造公差相关
    'ToleranceType',
    'ManufacturingSpecs',
    'PROCESS_SPECS',
    'EdgeRoughnessModel',
    'EtchBiasModel',
    'RobustnessConstraint',
    'CDVariationConstraint',
    'DesignRuleCheck',
    'get_process_specs',
    'list_available_processes',
    
    # 热效应相关
    'ThermalConfig',
    'HeatEquationSolver',
    'TemperatureConstraint',
    'ThermalStressConstraint',
    'ThermoOpticConstraint',
    'HeatDissipationConstraint',
    'ThermalConstraint',
    'create_thermal_constraint',
    'compute_temperature_field',
    'estimate_thermal_index_change',
]
