/**
 * LLM API 客户端
 */

import apiClient from './api';

// ===== 类型定义 =====

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
}

export interface ChatRequest {
    message: string;
    history?: ChatMessage[];
    use_rag?: boolean;
}

export interface ChatResponse {
    response: string;
    success: boolean;
    error?: string;
}

export interface IntentResponse {
    device_type: string;
    target_specs: Record<string, any>;
    constraints: Record<string, any>;
    preferences: Record<string, any>;
    confidence: number;
    clarification_needed: string[];
    success: boolean;
    error?: string;
}

export interface WorkflowResponse {
    pipeline_name: string;
    challenge: string;
    model_config: Record<string, any>;
    training_config: Record<string, any>;
    optimization_config: Record<string, any>;
    rationale: string;
    alternatives: string[];
    success: boolean;
    error?: string;
}

export interface ReportResponse {
    summary: string;
    metrics_analysis: Record<string, any>;
    design_features: string[];
    potential_issues: string[];
    suggestions: string[];
    raw_response: string;
    success: boolean;
    error?: string;
}

// ===== 向量数据库类型定义 =====

export type VectorDBType = 'qdrant' | 'chroma';

export interface VectorDBTypeInfo {
    id: VectorDBType;
    name: string;
    description: string;
    features: string[];
}

export interface VectorDBInfo {
    current_type: VectorDBType;
    available_types: VectorDBType[];
    collection_info?: {
        name: string;
        count?: number;
        vectors_count?: number;
        points_count?: number;
        status?: string;
        db_type?: string;
        persistent?: boolean;
        persist_directory?: string;
    } | null;
    success: boolean;
    error?: string;
}

export interface SwitchVectorDBRequest {
    vector_db_type: VectorDBType;
}

export interface SwitchVectorDBResponse {
    previous_type: VectorDBType;
    current_type: VectorDBType;
    success: boolean;
    error?: string;
}

// ===== LLM API =====

export const llmApi = {
    /**
     * 与 LLM 助手对话
     */
    chat: (request: ChatRequest) =>
        apiClient.post<ChatResponse>('/api/llm/chat', {
            message: request.message,
            history: request.history,
            use_rag: request.use_rag ?? true,
        }),

    /**
     * 解析用户设计意图
     */
    parseIntent: (userInput: string, useRag = true) =>
        apiClient.post<IntentResponse>('/api/llm/parse-intent', {
            user_input: userInput,
            use_rag: useRag,
        }),

    /**
     * 生成工作流配置
     */
    generateWorkflow: (intent: IntentResponse, useRag = true) =>
        apiClient.post<WorkflowResponse>('/api/llm/generate-workflow', {
            intent,
            use_rag: useRag,
        }),

    /**
     * 解释设计结果
     */
    explainResults: (designResult: Record<string, any>, simulationResult: Record<string, any>, detailLevel = 'normal') =>
        apiClient.post<ReportResponse>('/api/llm/explain-results', {
            design_result: designResult,
            simulation_result: simulationResult,
            detail_level: detailLevel,
        }),

    /**
     * 健康检查
     */
    healthCheck: () =>
        apiClient.get('/api/llm/health'),

    // ===== 向量数据库 API =====

    /**
     * 获取当前向量数据库信息
     */
    getVectorDBInfo: () =>
        apiClient.get<VectorDBInfo>('/api/llm/vector-db/info'),

    /**
     * 切换向量数据库
     */
    switchVectorDB: (vectorDBType: VectorDBType) =>
        apiClient.post<SwitchVectorDBResponse>('/api/llm/vector-db/switch', {
            vector_db_type: vectorDBType,
        }),

    /**
     * 初始化向量数据库
     */
    initializeVectorDB: (recreate = false) =>
        apiClient.post('/api/llm/vector-db/initialize', null, {
            params: { recreate },
        }),

    /**
     * 获取可用的向量数据库类型
     */
    getAvailableVectorDBTypes: () =>
        apiClient.get<{ types: VectorDBTypeInfo[] }>('/api/llm/vector-db/types'),
};

export default llmApi;
