/**
 * API 客户端
 */

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

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

    // 执行工作流
    executeWorkflow: (nodes: any[], edges: any[]) =>
        apiClient.post('/api/workflow/execute', { nodes, edges }),

    // 执行单个节点
    executeNode: (node: any, inputs: any) =>
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

export default apiClient;