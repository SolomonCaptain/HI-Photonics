/**
 * 模型类型定义
 */

export type ModelType =
    | 'tnn'     // 串联网络
    | 'mdn'     // 混合密度网络
    | 'cgan'    // 条件生成对抗网络
    | 'pinn'    // 物理信息神经网络
    | 'gnn'     // 图神经网络
    | 'hilab';  // HiLab （VAE + 贝叶斯优化）

export interface ModelConfig {
    name: string;
    type: ModelType;
    params: Record<string, any>;
}

export interface TrainingConfig {
    epochs: number;
    batchSize: number;
    learningRate: number;
    weightDecay: number;
    patience: number;
    device: 'cpu' | 'cuda';
}

export interface TrainingProgress {
    epoch: number;
    totalEpochs: number;
    loss: number;
    metrics: Record<string, number>;
    status: 'running' | 'completed' | 'failed';
}

export interface ModelInfo {
    type: ModelType;
    displayName: string;
    description: string;
    icon: string;
    params: ModelParameter[];
}

export interface ModelParameter {
    key: string;
    label: string;
    type: 'nummber' | 'select' | 'boolean';
    default: any;
    options?: { label: string; value: any }[];
    min?: number;
    max?: number;
}