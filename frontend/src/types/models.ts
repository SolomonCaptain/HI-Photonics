/**
 * 模型类型定义
 */

export type ModelType = 
  | 'tnn'      // Tandem Network
  | 'mdn'      // Mixture Density Network
  | 'cgan'     // Conditional GAN
  | 'pinn'     // Physics-Informed Neural Network
  | 'gnn'      // Graph Neural Network
  | 'hilab';   // HiLab Hybrid Inverse Design

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
  type: 'number' | 'select' | 'boolean';
  default: any;
  options?: { label: string; value: any }[];
  min?: number;
  max?: number;
}
