/**
 * 节点类型定义
 * 对应后端 core/nodes 模块
 */

export enum NodeType {
    // 参数化节点
    PARAMETERIZATION = 'parameterization',

    // 仿真节点
    SIMULATION = 'simulation',

    // 目标函数节点
    OBJECTIVE = 'objective',

    // 滤波器节点
    FILTER = 'filter',

    // 投影节点
    PROJECTION = 'projection',

    // 约束节点
    CONSTRAINT = 'constraint',

    // 模型节点
    MODEL_TRAIN = 'model_train',
    MODEL_INFER = 'model_infer',

    // 数据节点
    DATA_LOAD = 'data_load',
    DATA_SAVE = 'data_save',

    // 优化节点
    OPTIMIZER = 'optimizer',

    // 输出节点
    OUTPUT = 'output',
}

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