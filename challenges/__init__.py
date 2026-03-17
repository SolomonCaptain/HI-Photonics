"""
光子学设计挑战模块

提供各种光子学逆向设计问题的定义。
"""

from .base import (
    DesignChallenge,
    DesignSpec,
    PerformanceTarget,
    ChallengeFactory,
    register_challenge
)

from .grating_coupler import (
    GratingCouplerChallenge,
    MockGratingSimulator
)

from .metagrating import (
    MetagratingChallenge,
    MockMetagratingSimulator
)

from .wavelength_demux import (
    WavelengthDemuxChallenge,
    MockDemuxSimulator
)

__all__ = [
    # 基类
    'DesignChallenge',
    'DesignSpec',
    'PerformanceTarget',
    'ChallengeFactory',
    'register_challenge',
    
    # 具体挑战
    'GratingCouplerChallenge',
    'MetagratingChallenge',
    'WavelengthDemuxChallenge',
    
    # 模拟仿真器（测试用）
    'MockGratingSimulator',
    'MockMetagratingSimulator',
    'MockDemuxSimulator',
]
