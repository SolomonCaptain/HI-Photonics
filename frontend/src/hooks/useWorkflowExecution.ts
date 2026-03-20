/**
 * 工作流执行 Hook
 */

import { useState, useCallback } from 'react';
import { workflowApi } from '../utils/api';
import { NodeInstance, NodeConnection } from '../types';

interface ExecutionState {
    isExecuting: boolean;
    currentNodeId: string | null;
    results: Record<string, any>;
    error: string | null;
}

export function useWorkflowExecution() {
    const [state, setState] = useState<ExecutionState>({
        isExecuting: false,
        currentNodeId: null,
        results: {},
        error: null
    });

    const executeWorkflow = useCallback(async (
        nodes: NodeInstance[],
        edges: NodeConnection[],
        onProgress?: (nodeId: string, status: string) => void
    ) => {
        setState(prev => ({ ...prev, isExecuting: true, error: null }));

        try {
            const response = await workflowApi.executeWorkflow(nodes, edges);
            const results = response.data;

            // 转换为字典格式
            const resultsMap: Record<string, any> = {};
            results.forEach((result: any) => {
                resultsMap[result.node_id] = result;
                if (onProgress) {
                    onProgress(result.node_id, result.status);
                }
            });

            setState(prev => ({
                ...prev,
                isExecuting: false,
                results: resultsMap
            }));

            return results;
        } catch (error: any) {
            setState(prev => ({
                ...prev,
                isExecuting: false,
                error: error.message
            }));
            throw error;
        }
    }, []);

    const executeNode = useCallback(async (
        node: NodeInstance,
        inputs: Record<string, any>
    ) => {
        setState(prev => ({ ...prev, currentNodeId: node.id }));

        try {
            const response = await workflowApi.executeNode(node, inputs);
            const result = response.data;

            setState(prev => ({
                ...prev,
                currentNodeId: null,
                results: { ...prev.results, [node.id]: result }
            }));

            return result;
        } catch (error: any) {
            setState(prev => ({
                ...prev,
                currentNodeId: null,
                error: error.message
            }));
            throw error;
        }
    }, []);

    const clearReasults = useCallback(() => {
        setState(prev => ({ ...prev, results: {}, error: null }));
    }, []);

    return {
        ...state,
        executeWorkflow,
        executeNode,
        clearReasults
    };
}