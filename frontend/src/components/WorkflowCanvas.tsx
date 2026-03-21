/**
 * 工作流画布组件
 * 使用 React Flow 实现类 ComfyUI 的节点编辑器
 */

import React, { useCallback, useRef } from "react";
import {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    addEdge,
    applyNodeChanges,
    applyEdgeChanges,
    BackgroundVariant,
    ReactFlowProvider,
    ConnectionLineType
} from '@xyflow/react';
import type {
    Connection,
    NodeTypes,
    ReactFlowInstance,
    OnNodesChange,
    OnEdgesChange,
    Node,
    Edge
} from '@xyflow/react';
import { Box } from '@mui/material';
import { WorkflowNode } from './nodes';
import { useWorkflowStore } from '../store/workflowStore';
import type { NodeType, NodeInstance, NodeConnection } from '../types';
import '@xyflow/react/dist/style.css';

const nodeTypes: NodeTypes = {
    parameterization: WorkflowNode,
    simulation: WorkflowNode,
    objective: WorkflowNode,
    filter: WorkflowNode,
    projection: WorkflowNode,
    constraint: WorkflowNode,
    model_train: WorkflowNode,
    model_infer: WorkflowNode,
    data_load: WorkflowNode,
    data_save: WorkflowNode,
    optimizer: WorkflowNode,
    output: WorkflowNode
};

const WorkflowCanvas: React.FC = () => {
    const reactFlowWrapper = useRef<HTMLDivElement>(null);
    const [reactFlowInstance, setReactFlowInstance] = React.useState<ReactFlowInstance<any, any> | null>(null);

    const {
        nodes,
        edges,
        addNode,
        addEdge: storeAddEdge,
        setNodes,
        setEdges,
        setSelectedNode
    } = useWorkflowStore();

    // 处理节点变化（拖拽、选择等）
    const onNodesChange: OnNodesChange = useCallback((changes) => {
        setNodes(applyNodeChanges(changes, nodes as Node[]) as NodeInstance[]);
    }, [nodes, setNodes]);

    // 处理边变化
    const onEdgesChange: OnEdgesChange = useCallback((changes) => {
        setEdges(applyEdgeChanges(changes, edges as Edge[]) as NodeConnection[]);
    }, [edges, setEdges]);

    // 连接处理
    const onConnect = useCallback((connection: Connection) => {
        if (connection.source && connection.target && connection.sourceHandle && connection.targetHandle) {
            storeAddEdge({
                id: '',
                source: connection.source,
                target: connection.target,
                sourceHandle: connection.sourceHandle,
                targetHandle: connection.targetHandle
            });
        }
        setEdges(addEdge(connection, edges as Edge[]) as NodeConnection[]);
    }, [storeAddEdge, setEdges, edges]);

    // 拖放处理
    const onDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onDrop = useCallback((event: React.DragEvent) => {
        event.preventDefault();

        const type = event.dataTransfer.getData('application/reactflow') as NodeType;
        if (!type || !reactFlowInstance) return;

        const position = reactFlowInstance.screenToFlowPosition({
            x: event.clientX,
            y: event.clientY,
        });

        addNode(type, position);
    }, [reactFlowInstance, addNode]);

    // 点击画布空白处取消选择
    const onPanelClick = useCallback(() => {
        setSelectedNode(null);
    }, [setSelectedNode]);

    // 点击节点选中
    const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
        setSelectedNode(node.id);
    }, [setSelectedNode]);

    return (
        <Box
            ref={reactFlowWrapper}
            sx={{
                flex: 1,
                height: '100%',
                background: '#0f0f1a'
            }}
        >
            <ReactFlow
                nodes={nodes as Node[]}
                edges={edges as Edge[]}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onInit={setReactFlowInstance}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onPaneClick={onPanelClick}
                onNodeClick={onNodeClick}
                nodeTypes={nodeTypes}
                fitView
                snapToGrid
                snapGrid={[15, 15]}
                defaultEdgeOptions={{
                    type: 'smoothstep',
                    animated: true,
                    style: { stroke: '#3b82f6', strokeWidth: 2 }
                }}
                connectionLineStyle={{ stroke: '#3b82f6', strokeWidth: 2 }}
                connectionLineType={ConnectionLineType.SmoothStep}
            >
                <Background
                    variant={BackgroundVariant.Dots}
                    gap={20}
                    size={1}
                    color="#2d3748"
                />
                <Controls
                    style={{
                        background: 'rgba(26, 26, 46, 0.9)',
                        borderRadius: 8,
                        border: '1px solid #2d3748'
                    }}
                />
                <MiniMap
                    nodeColor={(node) => {
                        const status = node.data?.status;
                        switch (status) {
                            case 'success': return '#22c55e';
                            case 'error': return '#ef4444';
                            case 'running': return '#3b82f6';
                            default: return '#4a5568';
                        }
                    }}
                    style={{
                        background: 'rgba(26, 26, 46, 0.9)',
                        borderRadius: 8,
                        border: '1px solid #2d3748'
                    }}
                />
            </ReactFlow>
        </Box>
    );
};

export default function WorkflowCanvasWrapper() {
    return (
        <ReactFlowProvider>
            <WorkflowCanvas />
        </ReactFlowProvider>
    );
}