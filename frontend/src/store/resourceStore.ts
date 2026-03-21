/**
 * 资源管理状态
 * 使用 Zustand 进行状态管理
 */

import { create } from 'zustand';
import { resourceApi } from '../utils/api';

// 类型定义
export type ResourceCategory = 'inputs' | 'outputs' | 'models' | 'workflows';

export type AssetType =
    | 'dataset' | 'spectrum' | 'gds' | 'structure'   // 输入类
    | 'design' | 'simulation' | 'export'              // 输出类
    | 'field' | 'model_weights';                       // 其他

export type ModelType = 'tnn' | 'mdn' | 'cgan' | 'pinn' | 'gnn' | 'hilab';

export interface Asset {
    id: string;
    name: string;
    type: AssetType;
    category: ResourceCategory;
    description?: string;
    filePath: string;
    fileSize: number;
    createdAt: string;
    updatedAt: string;
    metadata?: Record<string, any>;
    thumbnail?: string;
}

export interface ModelInfo {
    id: string;
    name: string;
    type: ModelType;
    challenge: string;
    description?: string;
    filePath: string;
    fileSize: number;
    metrics?: Record<string, number>;
    createdAt: string;
    updatedAt: string;
    isPretrained: boolean;
}

export interface SavedWorkflow {
    id: string;
    name: string;
    description?: string;
    nodes: any[];
    edges: any[];
    filePath: string;
    createdAt: string;
    updatedAt: string;
    tags?: string[];
}

export interface WorkflowTemplate {
    id: string;
    name: string;
    description?: string;
    category: string;
    icon: string;
    nodes: any[];
    edges: any[];
    tags?: string[];
    filePath: string;
}

export interface DirectoryInfo {
    path: string;
    name: string;
    category: ResourceCategory;
    totalSize: number;
    fileCount: number;
    subdirectories: string[];
}

interface ResourceState {
    // 资产
    assets: Asset[];
    assetsLoading: boolean;
    assetsError: string | null;
    selectedAsset: Asset | null;

    // 模型
    models: ModelInfo[];
    modelsLoading: boolean;
    modelsError: string | null;
    selectedModel: ModelInfo | null;

    // 工作流
    workflows: SavedWorkflow[];
    workflowsLoading: boolean;
    workflowsError: string | null;

    // 模板
    templates: WorkflowTemplate[];
    templatesLoading: boolean;
    templatesError: string | null;

    // 目录信息
    directoryInfo: Record<ResourceCategory, DirectoryInfo | null>;

    // 搜索和过滤
    searchQuery: string;
    filterCategory: ResourceCategory | null;
    filterAssetType: AssetType | null;
    filterModelType: ModelType | null;

    // Actions - 资产
    fetchAssets: (category?: ResourceCategory, assetType?: AssetType, search?: string) => Promise<void>;
    uploadAsset: (formData: FormData) => Promise<Asset>;
    updateAsset: (assetId: string, category: ResourceCategory, data: Partial<Asset>) => Promise<void>;
    deleteAsset: (assetId: string, category: ResourceCategory) => Promise<void>;
    downloadAsset: (assetId: string, category: ResourceCategory) => Promise<void>;
    setSelectedAsset: (asset: Asset | null) => void;

    // Actions - 模型
    fetchModels: (modelType?: ModelType, challenge?: string, pretrainedOnly?: boolean) => Promise<void>;
    deleteModel: (modelId: string) => Promise<void>;
    downloadModel: (modelId: string) => Promise<void>;
    setSelectedModel: (model: ModelInfo | null) => void;

    // Actions - 工作流
    fetchWorkflows: (search?: string) => Promise<void>;
    saveWorkflow: (data: { name: string; nodes: any[]; edges: any[]; description?: string; tags?: string[] }) => Promise<SavedWorkflow>;
    deleteWorkflow: (workflowId: string) => Promise<void>;

    // Actions - 模板
    fetchTemplates: (category?: string) => Promise<void>;

    // Actions - 目录
    fetchDirectoryInfo: (category: ResourceCategory) => Promise<void>;

    // Actions - 搜索和过滤
    setSearchQuery: (query: string) => void;
    setFilterCategory: (category: ResourceCategory | null) => void;
    setFilterAssetType: (type: AssetType | null) => void;
    setFilterModelType: (type: ModelType | null) => void;

    // 清除错误
    clearErrors: () => void;
}

export const useResourceStore = create<ResourceState>((set) => ({
    // 初始状态
    assets: [],
    assetsLoading: false,
    assetsError: null,
    selectedAsset: null,

    models: [],
    modelsLoading: false,
    modelsError: null,
    selectedModel: null,

    workflows: [],
    workflowsLoading: false,
    workflowsError: null,

    templates: [],
    templatesLoading: false,
    templatesError: null,

    directoryInfo: {
        inputs: null,
        outputs: null,
        models: null,
        workflows: null,
    },

    searchQuery: '',
    filterCategory: null,
    filterAssetType: null,
    filterModelType: null,

    // 资产操作
    fetchAssets: async (category, assetType, search) => {
        set({ assetsLoading: true, assetsError: null });
        try {
            const response = await resourceApi.listAssets({ category, assetType, search });
            set({ assets: response.data, assetsLoading: false });
        } catch (error: any) {
            set({ assetsError: error.message || 'Failed to fetch assets', assetsLoading: false });
        }
    },

    uploadAsset: async (formData) => {
        set({ assetsLoading: true, assetsError: null });
        try {
            const response = await resourceApi.uploadAsset(formData);
            const newAsset = response.data;
            set(state => ({
                assets: [...state.assets, newAsset],
                assetsLoading: false
            }));
            return newAsset;
        } catch (error: any) {
            set({ assetsError: error.message || 'Failed to upload asset', assetsLoading: false });
            throw error;
        }
    },

    updateAsset: async (assetId, category, data) => {
        set({ assetsLoading: true, assetsError: null });
        try {
            const response = await resourceApi.updateAsset(assetId, category, data);
            const updatedAsset = response.data;
            set(state => ({
                assets: state.assets.map(a => a.id === assetId ? updatedAsset : a),
                assetsLoading: false
            }));
        } catch (error: any) {
            set({ assetsError: error.message || 'Failed to update asset', assetsLoading: false });
            throw error;
        }
    },

    deleteAsset: async (assetId, category) => {
        set({ assetsLoading: true, assetsError: null });
        try {
            await resourceApi.deleteAsset(assetId, category);
            set(state => ({
                assets: state.assets.filter(a => a.id !== assetId),
                selectedAsset: state.selectedAsset?.id === assetId ? null : state.selectedAsset,
                assetsLoading: false
            }));
        } catch (error: any) {
            set({ assetsError: error.message || 'Failed to delete asset', assetsLoading: false });
            throw error;
        }
    },

    downloadAsset: async (assetId, category) => {
        try {
            const response = await resourceApi.downloadAsset(assetId, category);
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `asset_${assetId}`);
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
        } catch (error: any) {
            set({ assetsError: error.message || 'Failed to download asset' });
        }
    },

    setSelectedAsset: (asset) => set({ selectedAsset: asset }),

    // 模型操作
    fetchModels: async (modelType, challenge, pretrainedOnly) => {
        set({ modelsLoading: true, modelsError: null });
        try {
            const response = await resourceApi.listModels({ modelType, challenge, pretrainedOnly });
            set({ models: response.data, modelsLoading: false });
        } catch (error: any) {
            set({ modelsError: error.message || 'Failed to fetch models', modelsLoading: false });
        }
    },

    deleteModel: async (modelId) => {
        set({ modelsLoading: true, modelsError: null });
        try {
            await resourceApi.deleteModel(modelId);
            set(state => ({
                models: state.models.filter(m => m.id !== modelId),
                selectedModel: state.selectedModel?.id === modelId ? null : state.selectedModel,
                modelsLoading: false
            }));
        } catch (error: any) {
            set({ modelsError: error.message || 'Failed to delete model', modelsLoading: false });
            throw error;
        }
    },

    downloadModel: async (modelId) => {
        try {
            const response = await resourceApi.downloadModel(modelId);
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `model_${modelId}.pt`);
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
        } catch (error: any) {
            set({ modelsError: error.message || 'Failed to download model' });
        }
    },

    setSelectedModel: (model) => set({ selectedModel: model }),

    // 工作流操作
    fetchWorkflows: async (search) => {
        set({ workflowsLoading: true, workflowsError: null });
        try {
            const response = await resourceApi.listWorkflows(search);
            set({ workflows: response.data, workflowsLoading: false });
        } catch (error: any) {
            set({ workflowsError: error.message || 'Failed to fetch workflows', workflowsLoading: false });
        }
    },

    saveWorkflow: async (data) => {
        set({ workflowsLoading: true, workflowsError: null });
        try {
            const response = await resourceApi.saveWorkflow(data);
            const newWorkflow = response.data;
            set(state => ({
                workflows: [...state.workflows, newWorkflow],
                workflowsLoading: false
            }));
            return newWorkflow;
        } catch (error: any) {
            set({ workflowsError: error.message || 'Failed to save workflow', workflowsLoading: false });
            throw error;
        }
    },

    deleteWorkflow: async (workflowId) => {
        set({ workflowsLoading: true, workflowsError: null });
        try {
            await resourceApi.deleteWorkflow(workflowId);
            set(state => ({
                workflows: state.workflows.filter(w => w.id !== workflowId),
                workflowsLoading: false
            }));
        } catch (error: any) {
            set({ workflowsError: error.message || 'Failed to delete workflow', workflowsLoading: false });
            throw error;
        }
    },

    // 模板操作
    fetchTemplates: async (category) => {
        set({ templatesLoading: true, templatesError: null });
        try {
            const response = await resourceApi.listTemplates(category);
            set({ templates: response.data, templatesLoading: false });
        } catch (error: any) {
            set({ templatesError: error.message || 'Failed to fetch templates', templatesLoading: false });
        }
    },

    // 目录操作
    fetchDirectoryInfo: async (category) => {
        try {
            const response = await resourceApi.getDirectoryInfo(category);
            set(state => ({
                directoryInfo: { ...state.directoryInfo, [category]: response.data }
            }));
        } catch (error: any) {
            console.error(`Failed to fetch directory info for ${category}:`, error);
        }
    },

    // 搜索和过滤
    setSearchQuery: (query) => set({ searchQuery: query }),
    setFilterCategory: (category) => set({ filterCategory: category }),
    setFilterAssetType: (type) => set({ filterAssetType: type }),
    setFilterModelType: (type) => set({ filterModelType: type }),

    // 清除错误
    clearErrors: () => set({
        assetsError: null,
        modelsError: null,
        workflowsError: null,
        templatesError: null,
    }),
}));
