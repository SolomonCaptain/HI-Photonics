/**
 * LLM 助手状态管理
 * 使用 Zustand 进行状态管理
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatMessage, IntentResponse, WorkflowResponse, VectorDBType, VectorDBInfo } from '../utils/llmApi';
import { llmApi } from '../utils/llmApi';
import toast from 'react-hot-toast';

interface LLMState {
    // 对话状态
    messages: ChatMessage[];
    isLoading: boolean;
    error: string | null;

    // 意图解析结果
    currentIntent: IntentResponse | null;
    
    // 工作流建议
    workflowSuggestion: WorkflowResponse | null;

    // 向量数据库状态
    vectorDBType: VectorDBType;
    vectorDBInfo: VectorDBInfo | null;
    isVectorDBLoading: boolean;

    // UI 状态
    isExpanded: boolean;

    // Actions
    sendMessage: (content: string) => Promise<void>;
    clearMessages: () => void;
    
    parseIntent: (userInput: string) => Promise<IntentResponse | null>;
    generateWorkflow: (intent: IntentResponse) => Promise<WorkflowResponse | null>;
    
    // 向量数据库操作
    fetchVectorDBInfo: () => Promise<void>;
    switchVectorDB: (type: VectorDBType) => Promise<boolean>;
    
    setExpanded: (expanded: boolean) => void;
    setError: (error: string | null) => void;
}

// 预设的欢迎消息
const WELCOME_MESSAGE: ChatMessage = {
    role: 'assistant',
    content: '你好！我是光子学逆向设计助手。我可以帮助你：\n\n• **解析设计意图** - 描述你想要的器件，我会帮你提取设计参数\n• **生成工作流配置** - 根据你的需求推荐合适的模型和参数\n• **解答技术问题** - 关于光子学设计、模型选择、参数优化等\n\n试试说："帮我设计一个1550nm波长的光栅耦合器，效率要大于70%"'
};

export const useLLMStore = create<LLMState>()(
    persist(
        (set, get) => ({
            messages: [WELCOME_MESSAGE],
            isLoading: false,
            error: null,
            currentIntent: null,
            workflowSuggestion: null,
            vectorDBType: 'qdrant',
            vectorDBInfo: null,
            isVectorDBLoading: false,
            isExpanded: false,

            fetchVectorDBInfo: async () => {
                set({ isVectorDBLoading: true });
                try {
                    const response = await llmApi.getVectorDBInfo();
                    if (response.data.success) {
                        set({
                            vectorDBInfo: response.data,
                            vectorDBType: response.data.current_type,
                            isVectorDBLoading: false,
                        });
                    }
                } catch (error: any) {
                    console.error('获取向量数据库信息失败:', error);
                    set({ isVectorDBLoading: false });
                }
            },

            switchVectorDB: async (type: VectorDBType) => {
                const previousType = get().vectorDBType;
                set({ isVectorDBLoading: true });
                
                try {
                    const response = await llmApi.switchVectorDB(type);
                    if (response.data.success) {
                        set({
                            vectorDBType: type,
                            isVectorDBLoading: false,
                        });
                        toast.success(`已切换到 ${type === 'qdrant' ? 'Qdrant' : 'Chroma'} 向量数据库`);
                        // 刷新信息
                        get().fetchVectorDBInfo();
                        return true;
                    } else {
                        throw new Error(response.data.error || '切换失败');
                    }
                } catch (error: any) {
                    set({
                        vectorDBType: previousType,
                        isVectorDBLoading: false,
                    });
                    toast.error(`切换向量数据库失败: ${error.message}`);
                    return false;
                }
            },

            sendMessage: async (content: string) => {
                const userMessage: ChatMessage = { role: 'user', content };
                set(state => ({
                    messages: [...state.messages, userMessage],
                    isLoading: true,
                    error: null,
                }));

                try {
                    const history = get().messages.slice(-10); // 保留最近10条消息
                    
                    const response = await llmApi.chat({
                        message: content,
                        history: history.slice(0, -1), // 不包含刚添加的用户消息
                        use_rag: true,
                    });

                    if (response.data.success) {
                        const assistantMessage: ChatMessage = {
                            role: 'assistant',
                            content: response.data.response,
                        };
                        set(state => ({
                            messages: [...state.messages, assistantMessage],
                            isLoading: false,
                        }));
                    } else {
                        throw new Error(response.data.error || '对话失败');
                    }
                } catch (error: any) {
                    const errorMessage = error.response?.data?.error || error.message || '对话失败';
                    set(_state => ({
                        isLoading: false,
                        error: errorMessage,
                    }));
                    toast.error(`对话失败: ${errorMessage}`);
                }
            },

            clearMessages: () => {
                set({
                    messages: [WELCOME_MESSAGE],
                    currentIntent: null,
                    workflowSuggestion: null,
                    error: null,
                });
            },

            parseIntent: async (userInput: string) => {
                set({ isLoading: true, error: null });

                try {
                    const response = await llmApi.parseIntent(userInput);

                    if (response.data.success) {
                        set({
                            currentIntent: response.data,
                            isLoading: false,
                        });
                        return response.data;
                    } else {
                        throw new Error(response.data.error || '意图解析失败');
                    }
                } catch (error: any) {
                    const errorMessage = error.response?.data?.error || error.message || '意图解析失败';
                    set({ isLoading: false, error: errorMessage });
                    toast.error(`意图解析失败: ${errorMessage}`);
                    return null;
                }
            },

            generateWorkflow: async (intent: IntentResponse) => {
                set({ isLoading: true, error: null });

                try {
                    const response = await llmApi.generateWorkflow(intent);

                    if (response.data.success) {
                        set({
                            workflowSuggestion: response.data,
                            isLoading: false,
                        });
                        return response.data;
                    } else {
                        throw new Error(response.data.error || '工作流生成失败');
                    }
                } catch (error: any) {
                    const errorMessage = error.response?.data?.error || error.message || '工作流生成失败';
                    set({ isLoading: false, error: errorMessage });
                    toast.error(`工作流生成失败: ${errorMessage}`);
                    return null;
                }
            },

            setExpanded: (expanded) => set({ isExpanded: expanded }),
            setError: (error) => set({ error }),
        }),
        {
            name: 'llm-assistant-storage',
            partialize: (state) => ({
                messages: state.messages,
                vectorDBType: state.vectorDBType,
            }),
        }
    )
);

export default useLLMStore;
