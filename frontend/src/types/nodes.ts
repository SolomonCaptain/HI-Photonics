/**
 * 节点类型定义
 * 对应后端 core/nodes 模块
 */

export const NodeType = {
    // 参数化节点
    PARAMETERIZATION: 'parameterization',

    // 仿真节点
    SIMULATION: 'simulation',

    // 目标函数节点
    OBJECTIVE: 'objective',

    // 滤波器节点
    FILTER: 'filter',

    // 投影节点
    PROJECTION: 'projection',

    // 约束节点
    CONSTRAINT: 'constraint',

    // 模型节点
    MODEL_TRAIN: 'model_train',
    MODEL_INFER: 'model_infer',

    // 数据节点
    DATA_LOAD: 'data_load',
    DATA_SAVE: 'data_save',

    // 优化节点
    OPTIMIZER: 'optimizer',

    // 输出节点
    OUTPUT: 'output',
} as const;

export type NodeType = typeof NodeType[keyof typeof NodeType];

export interface NodePort {
    id: string;
    name: string;
    type: 'input' | 'output';
    dataType: 'design' | 'performance' | 'params' | 'model' | 'data' | 'any';
    required: boolean;
    multiple?: boolean;
}

export interface NodeDefinition {
    type: NodeType;
    name: string;
    category: string;
    description: string;
    icon: string;
    inputs: NodePort[];
    outputs: NodePort[];
    params: NodeParameter[];
}

export interface NodeParameter {
    key: string;
    label: string;
    type: 'number' | 'string' | 'select' | 'boolean' | 'array' | 'object';
    default: any;
    options?: { label: string; value: any }[];
    min?: number;
    max?: number;
    step?: number;
    description?: string;
}

export interface NodeInstance {
    id: string;
    type: NodeType;
    position: { x: number; y: number };
    data: {
        label: string;
        params: Record<string, any>;
        status: 'idle' | 'running' | 'success' | 'error';
        progress?: number;
        result?: any;
        error?: string;
    };
}

export interface NodeConnection {
    id: string;
    source: string;
    sourceHandle: string;
    target: string;
    targetHandle: string;
}

export interface Workflow {
    id: string;
    name: string;
    description: string;
    nodes: NodeInstance[];
    edges: NodeConnection[];
    createdAt: string;
    updatedAt: string;
}

export interface ExecutionResult {
    nodeId: string;
    status: 'success' | 'error';
    output?: any;
    error?: string;
    duration: number;
}

// 侧边栏面板类型
export const SidebarPanelType = {
    ASSETS: 'assets',
    NODES: 'nodes',
    MODELS: 'models',
    WORKFLOWS: 'workflows',
    TEMPLATES: 'templates',
    AI_ASSISTANT: 'ai_assistant',
} as const;

export type SidebarPanelType = typeof SidebarPanelType[keyof typeof SidebarPanelType];

// 侧边栏面板信息
export interface SidebarPanelInfo {
    type: SidebarPanelType;
    name: string;
    icon: string;
    description: string;
}

// 资产类型
export const AssetType = {
    SPECTRUM: 'spectrum',       // 光谱图
    GDS: 'gds',                 // GDS版图
    STRUCTURE: 'structure',     // 结构设计
    FIELD: 'field',             // 场分布
    DATASET: 'dataset',         // 数据集
    MODEL_WEIGHTS: 'model_weights', // 模型权重
} as const;

export type AssetType = typeof AssetType[keyof typeof AssetType];

// 资产实例
export interface Asset {
    id: string;
    name: string;
    type: AssetType;
    description?: string;
    createdAt: string;
    updatedAt: string;
    size?: number;
    metadata?: Record<string, any>;
    thumbnail?: string;
}

// 模板类型
export interface WorkflowTemplate {
    id: string;
    name: string;
    description: string;
    category: string;
    icon: string;
    nodes: NodeInstance[];
    edges: NodeConnection[];
    tags?: string[];
}