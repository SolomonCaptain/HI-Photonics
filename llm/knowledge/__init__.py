"""
光子学知识库模块

包含器件知识、模型说明、设计规则等。
"""

from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent

# 器件类型定义
DEVICE_TYPES = {
    "grating_coupler": {
        "name": "光栅耦合器",
        "description": "用于光纤与光子芯片之间的光耦合",
        "key_parameters": ["wavelength", "efficiency", "bandwidth", "insertion_loss"],
        "typical_specs": {
            "wavelength": "1550 nm (C-band)",
            "efficiency": "60-80%",
            "bandwidth": "50-100 nm"
        }
    },
    "metagrating": {
        "name": "超构光栅",
        "description": "利用超表面结构实现高效光耦合",
        "key_parameters": ["wavelength", "efficiency", "angle"],
        "typical_specs": {
            "wavelength": "1550 nm",
            "efficiency": "80-95%"
        }
    },
    "wavelength_demux": {
        "name": "波分复用器",
        "description": "将不同波长的光信号分离到不同通道",
        "key_parameters": ["wavelengths", "crosstalk", "insertion_loss"],
        "typical_specs": {
            "wavelengths": "1310 nm, 1550 nm",
            "crosstalk": "< -20 dB",
            "insertion_loss": "< 2 dB"
        }
    }
}

# 材料属性
MATERIALS = {
    "silicon": {
        "name": "硅",
        "refractive_index": 3.48,
        "wavelength_range": "1.2 - 8 μm"
    },
    "sio2": {
        "name": "二氧化硅",
        "refractive_index": 1.44,
        "wavelength_range": "0.2 - 4 μm"
    },
    "sin": {
        "name": "氮化硅",
        "refractive_index": 2.0,
        "wavelength_range": "0.4 - 5 μm"
    }
}
