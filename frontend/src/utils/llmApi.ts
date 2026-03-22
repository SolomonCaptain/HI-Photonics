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
};

export default llmApi;
