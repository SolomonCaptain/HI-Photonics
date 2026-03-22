/**
 * API 客户端
 */

import axios from 'axios';

// 支持环境变量配置 API 地址
// Docker 部署时通过 Nginx 反向代理，使用相对路径 /api
// 开发环境使用完整 URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// 请求拦截器
apiClient.interceptors.request.use(
    (config) => {
        // 在此处添加认证 Token
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// 响应拦截器
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
    }
);

// 工作流 API
export const workflowApi = {
    // 获取节点定义
    getNodeDefinition: () =>
        apiClient.get('/api/workflow/nodes'),

    // 执行工作流 - 发送节点和边作为 JSON body
    executeWorkflow: (nodes: any[], edges: any[]) =>
        apiClient.post('/api/workflow/execute', { nodes, edges }),

    // 执行单个节点 - 发送节点和输入作为 JSON body
    executeNode: (node: any, inputs: Record<string, any>) =>
        apiClient.post('/api/workflow/execute-node', { node, inputs }),

    // 保存工作流
    saveWorkflow: (workflow: any) =>
        apiClient.post('/api/workflow/', workflow),

    // 获取工作流
    getWorkflow: (id: string) =>
        apiClient.get(`/api/workflow/${id}`),
};

// 模型 API
export const modelApi = {
    // 获取模型列表
    listModels: () =>
        apiClient.get('/api/models/'),

    // 获取模型信息
    getModelInfo: (modelType: string) =>
        apiClient.get(`/api/models/${modelType}`),

    // 开始训练
    trainModel: (config: any) =>
        apiClient.post('/api/models/train', config),

    // 逆向设计
    inverseDesign: (request: any) =>
        apiClient.post('/api/models/inverse-design', request),

    // 获取预训练模型列表
    listPretrained: () =>
        apiClient.get('/api/models/pretrained/list'),
};

// 系统 API
export const systemApi = {
    // 健康检查
    healthCheck: () =>
        apiClient.get('/health'),

    // 系统信息
    getSystemInfo: () =>
        apiClient.get('/api/system/info'),
};

// 资源管理 API
export const resourceApi = {
    // ===== 资产管理 =====
    // 列出资产
    listAssets: (params?: { category?: string; assetType?: string; search?: string }) =>
        apiClient.get('/api/resources/assets', { params }),

    // 获取资产详情
    getAsset: (assetId: string, category: string) =>
        apiClient.get(`/api/resources/assets/${assetId}`, { params: { category } }),

    // 上传资产
    uploadAsset: (formData: FormData) =>
        apiClient.post('/api/resources/assets', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        }),

    // 更新资产元数据
    updateAsset: (assetId: string, category: string, data: any) =>
        apiClient.patch(`/api/resources/assets/${assetId}`, data, { params: { category } }),

    // 删除资产
    deleteAsset: (assetId: string, category: string) =>
        apiClient.delete(`/api/resources/assets/${assetId}`, { params: { category } }),

    // 下载资产
    downloadAsset: (assetId: string, category: string) =>
        apiClient.get(`/api/resources/assets/${assetId}/download`, {
            params: { category },
            responseType: 'blob',
        }),

    // 批量删除资产
    batchDeleteAssets: (assetIds: string[], category: string) =>
        apiClient.post('/api/resources/assets/batch-delete', assetIds, { params: { category } }),

    // ===== 模型管理 =====
    // 列出模型
    listModels: (params?: { modelType?: string; challenge?: string; pretrainedOnly?: boolean }) =>
        apiClient.get('/api/resources/models', { params }),

    // 获取模型详情
    getModel: (modelId: string) =>
        apiClient.get(`/api/resources/models/${modelId}`),

    // 删除模型
    deleteModel: (modelId: string) =>
        apiClient.delete(`/api/resources/models/${modelId}`),

    // 下载模型
    downloadModel: (modelId: string) =>
        apiClient.get(`/api/resources/models/${modelId}/download`, {
            responseType: 'blob',
        }),

    // ===== 工作流管理 =====
    // 列出已保存的工作流
    listWorkflows: (search?: string) =>
        apiClient.get('/api/resources/workflows', { params: { search } }),

    // 获取工作流详情
    getWorkflow: (workflowId: string) =>
        apiClient.get(`/api/resources/workflows/${workflowId}`),

    // 保存工作流
    saveWorkflow: (data: { name: string; nodes: any[]; edges: any[]; description?: string; tags?: string[] }) =>
        apiClient.post('/api/resources/workflows', data),

    // 删除工作流
    deleteWorkflow: (workflowId: string) =>
        apiClient.delete(`/api/resources/workflows/${workflowId}`),

    // ===== 模板管理 =====
    // 列出模板
    listTemplates: (category?: string) =>
        apiClient.get('/api/resources/templates', { params: { category } }),

    // 获取模板详情
    getTemplate: (templateId: string) =>
        apiClient.get(`/api/resources/templates/${templateId}`),

    // ===== 目录信息 =====
    // 获取目录信息
    getDirectoryInfo: (category: string) =>
        apiClient.get(`/api/resources/directories/${category}`),
};

export default apiClient;