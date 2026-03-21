/**
 * 工作流状态管理
 * 使用 Zustand 进行状态管理
 */

import { create } from 'zustand';
import type {
    NodeInstance,
    NodeConnection,
    NodeDefinition
} from '../types';
import { NodeType, SidebarPanelType } from '../types';
import type { SidebarPanelType as SidebarPanelTypeType } from '../types';
import { workflowApi } from '../utils/api';
import toast from 'react-hot-toast';

// 节点定义注册表
export const NODE_DEFINITIONS: Record<NodeType, NodeDefinition> = {
    [NodeType.PARAMETERIZATION]: {
        type: NodeType.PARAMETERIZATION,
        name: '参数化',
        category: '设计',
        description: '定义设计参数空间',
        icon: 'Tune',
        inputs: [],
        outputs: [
            { id: 'design', name: '设计参数', type: 'output', dataType: 'design', required: true }
        ],
        params: [
            { key: 'shape', label: '设计形状', type: 'array', default: [100, 22] },
            { key: 'method', label: '参数化方法', type: 'select', default: 'density',
                options: [{ label: '密度', value: 'density' }, { label: '拓扑', value: 'topology' }] },
            { key: 'bounds', label: '值范围', type: 'array', default: [0, 1] }
        ]
    },

    [NodeType.SIMULATION]: {
        type: NodeType.SIMULATION,
        name: '仿真器',
        category: '仿真',
        description: '运行 FDTD/RCWA 仿真',
        icon: 'PlayArrow',
        inputs: [
            { id: 'design', name: '设计参数', type: 'input', dataType: 'design', required: true }
        ],
        outputs: [
            { id: 'performance', name: '性能指标', type: 'output', dataType: 'performance', required: true },
            { id: 'fields', name: '场分布', type: 'output', dataType: 'data', required: false }
        ],
        params: [
            { key: 'simulator', label: '仿真器类型', type: 'select', default: 'meep',
                options: [{ label: 'Meep FDTD', value: 'meep' }, { label: 'RCWA', value: 'rcwa' }] },
            { key: 'wavelength', label: '波长（μm）', type: 'number', default: 1.55, min: 0.4, max: 10 },
            { key: 'resolution', label: '分辨率', type: 'number', default: 50, min: 10, max: 200 }
        ]
    },

    [NodeType.OBJECTIVE]: {
        type: NodeType.OBJECTIVE,
        name: '目标函数',
        category: '优化',
        description: '定义优化目标',
        icon: 'Flag',
        inputs: [
            { id: 'performance', name: '性能指标', type: 'input', dataType: 'performance', required: true }
        ],
        outputs: [
            { id: 'loss', name: '损失值', type: 'output', dataType: 'params', required: true }
        ],
        params: [
            { key: 'type', label: '目标类型', type: 'select', default: 'efficiency',
                options: [{ label: '效率最大化', value: 'efficiency' }, { label: '带宽最大化', value: 'bandwidth' }] },
            { key: 'target', label: '目标值', type: 'number', default: 0.9, min: 0, max: 1 },
            { key: 'weight', label: '权重', type: 'number', default: 1.0, min: 0, max: 10 }
        ]
    },

    [NodeType.FILTER]: {
        type: NodeType.FILTER,
        name: '滤波器',
        category: '处理',
        description: '对设计参数进行滤波',
        icon: 'FilterAlt',
        inputs: [
            { id: 'design', name: '设计参数', type: 'input', dataType: 'design', required: true }
        ],
        outputs: [
            { id: 'filtered', name: '滤波效果', type: 'output', dataType: 'design', required: true }
        ],
        params: [
            { key: 'type', label: '滤波器类型', type: 'select', default: 'gaussian',
                options: [{ label: '高斯滤波', value: 'gaussian' }, { label: '形态学', value: 'morphological' }] },
            { key: 'sigma', label: '平滑度（σ）', type: 'number', default: 1.0, min: 0.1, max: 10 }
        ]
    },

    [NodeType.PROJECTION]: {
        type: NodeType.PROJECTION,
        name: '投影',
        category: '处理',
        description: '连续值投影到离散值',
        icon: 'Layers',
        inputs: [
            { id: 'design', name: '设计参数', type: 'input', dataType: 'design', required: true }
        ],
        outputs: [
            { id: 'projected', name: '投影结果', type: 'output', dataType: 'design', required: true }
        ],
        params: [
            { key: 'method', label: '投影方法', type: 'select', default: 'sigmoid',
                options: [{ label: 'Sigmoid', value: 'sigmoid' }, { label: 'Heaviside', value: 'heaviside' }] },
            { key: 'threshold', label: '阈值', type: 'number', default: 0.5, min: 0, max: 1 },
            { key: 'sharpness', label: '锐度（β）', type: 'number', default: 1.0, min: 0.1, max: 100 }
        ]
    },

    [NodeType.CONSTRAINT]: {
        type: NodeType.CONSTRAINT,
        name: '约束',
        category: '优化',
        description: '添加设计约束',
        icon: 'Block',
        inputs: [
            { id: 'design', name: '设计参数', type: 'input', dataType: 'design', required: true }
        ],
        outputs: [
            { id: 'penalty', name: '惩罚值', type: 'output', dataType: 'params', required: true }
        ],
        params: [
            { key: 'type', label: '约束类型', type: 'select', default: 'volume',
                options: [{ label: '体积约束', value: 'volime' }, { label: '对称性', value: 'symmetry' }, { label: '曲率', value: 'curvature' }] },
            { key: 'target', label: '目标值', type: 'number', default: 0.5, min: 0, max: 1 }
        ]
    },

    [NodeType.MODEL_TRAIN]: {
        type: NodeType.MODEL_TRAIN,
        name: '模型训练',
        category: '模型',
        description: '训练神经网络模型',
        icon: 'ModelTraining',
        inputs: [
            { id: 'data', name: '训练数据', type: 'input', dataType: 'data', required: true }
        ],
        outputs: [
            { id: 'model', name: '训练模型', type: 'output', dataType: 'model', required: true }
        ],
        params: [
            { key: 'modelType', label: '模型类型', type: 'select', default: 'hilab',
                options: [
                    { label: 'TNN', value: 'tnn' },
                    { label: 'MDN', value: 'mdn' },
                    { label: 'CGAN', value: 'cgan' },
                    { label: 'PINN', value: 'pinn' },
                    { label: 'GNN', value: 'gnn' },
                    { label: 'HiLab', value: 'hilab' },
                ]
            },
            { key: 'epochs', label: '训练轮数', type: 'number', default: 100, min: 1, max: 10000 },
            { key: 'batchSize', label: '批次大小', type: 'number', default: 32, min: 1, max: 512 },
            { key: 'learningRate', label: '学习率', type: 'number', default: 0.001, min: 0.00001, max: 1 }
        ]
    },

    [NodeType.MODEL_INFER]: {
        type: NodeType.MODEL_INFER,
        name: '模型推理',
        category: '模型',
        description: '使用模型进行逆向设计',
        icon: 'Psychology',
        inputs: [
            { id: 'model', name: '模型', type: 'input', dataType: 'model', required: true },
            { id: 'target', name: '目标性能', type: 'input', dataType: 'performance', required: true }
        ],
        outputs: [
            { id: 'design', name: '设计结果', type: 'output', dataType: 'design', required: true }
        ],
        params: [
            { key: 'numSamples', label: '采样数量', type: 'number', default: 1, min: 1, max: 100 },
            { key: 'diversity', label: '多样性权重', type: 'number', default: 0.0, min: 0, max: 1 }
        ]
    },

    [NodeType.DATA_LOAD]: {
        type: NodeType.DATA_LOAD,
        name: '数据加载',
        category: '数据',
        description: '加载训练/测试数据',
        icon: 'Storage',
        inputs: [],
        outputs: [
            { id: 'data', name: '数据集', type: 'output', dataType: 'data', required: true }
        ],
        params: [
            { key: 'source', label: '数据源', type: 'select', default: 'synthetic',
                options: [{ label: '合成数据', value: 'synthetic' }, { label: 'HDF5 文件', value: 'hdf5' }] },
            { key: 'numSamples', label: '样本数量', type: 'number', default: 1000, min: 10, max: 100000 },
            { key: 'designShape', label: '设计形状', type: 'array', default: [100, 22] }
        ]
    },

    [NodeType.DATA_SAVE]: {
        type: NodeType.DATA_SAVE,
        name: '数据保存',
        category: '数据',
        description: '保存设计/模型数据',
        icon: 'Save',
        inputs: [
            { id: 'data', name: '数据', type: 'input', dataType: 'any', required: true }
        ],
        outputs: [],
        params: [
            { key: 'format', label: '保存格式', type: 'select', default: 'hdf5',
                options: [{ label: 'HDF5', value: 'hdf5' }, { label: 'NPZ', value: 'npz' }, { label: 'JSON', value: 'json' }] },
            { key: 'filename', label: '文件名', type: 'string', default: 'output' }
        ]
    },

    [NodeType.OPTIMIZER]: {
        type: NodeType.OPTIMIZER,
        name: '优化器',
        category: '优化',
        description: '拓扑优化求解器',
        icon: 'AutoFixHigh',
        inputs: [
            { id: 'objective', name: '目标函数', type: 'input', dataType: 'params', required: true },
            { id: 'design', name: '初始设计', type: 'input', dataType: 'design', required: true }
        ],
        outputs: [
            { id: 'optimized', name: '优化结果', type: 'output', dataType: 'design', required: true }
        ],
        params: [
            { key: 'method', label: '优化方法', type: 'select', default: 'adam',
                options: [
                    { label: 'Adam', value: 'adam' },
                    { label: 'L-BFGS', value: 'lbfgs' },
                    { label: '贝叶斯优化', value: 'bayesian' },
                    { label: '进化算法', value: 'evolutionary' }
                ]
            },
            { key: 'iterations', label: '迭代次数', type: 'number', default: 100, min: 1, max: 10000 },
            { key: 'learningRate', label: '学习率', type: 'number', default: 0.01, min: 0.0001, max: 1 }
        ]
    },

    [NodeType.OUTPUT]: {
        type: NodeType.OUTPUT,
        name: '输出',
        category: '输出',
        description: '查看和可视化结果',
        icon: 'Visiblity',
        inputs: [
            { id: 'design', name: '设计', type: 'input', dataType: 'design', required: false },
            { id: 'performance', name: '性能', type: 'input', dataType: 'performance', required: true }
        ],
        outputs: [],
        params: [
            { key: 'showHeatmap', label: '显示热力图', type: 'boolean', default: true },
            { key: 'showMetrics', label: '显示指标', type: 'boolean', default: true }
        ]
    }
};

interface WorkflowState {
    // 工作流
    nodes: NodeInstance[];
    edges: NodeConnection[];
    selectedNodeId: string | null;

    // 执行状态
    isExecuting: boolean;
    executionResults: Record<string, any>;
    executionError: string | null;
    currentExecutingNodeId: string | null;

    // UI 状态
    sidebarOpen: boolean;
    propertiesOpen: boolean;
    currentPanel: SidebarPanelTypeType;

    // Actions
    addNode: (type: NodeType, position: { x: number; y: number }) => void;
    removeNode: (id: string) => void;
    updateNodePosition: (id: string, position: { x: number; y: number }) => void;
    updateNodeParams: (id: string, params: Record<string, any>) => void;
    setSelectedNode: (id: string | null) => void;

    addEdge: (edge: NodeConnection) => void;
    removeEdge: (id: string) => void;
    setNodes: (nodes: NodeInstance[]) => void;
    setEdges: (edges: NodeConnection[]) => void;

    executeNode: (id: string) => Promise<void>;
    executeAll: () => Promise<void>;
    updateNodeStatus: (id: string, status: 'idle' | 'running' | 'success' | 'error', result?: any, error?: string) => void;

    setSidebarOpen: (open: boolean) => void;
    setPropertiesOpen: (open: boolean) => void;
    setCurrentPanel: (panel: SidebarPanelTypeType) => void;
    togglePanel: (panel: SidebarPanelTypeType) => void;

    clearWorkflow: () => void;
    loadWorkflow: (nodes: NodeInstance[], edges: NodeConnection[]) => void;
}

let nodeIdCounter = 0;
let edgeIdCounter = 0;

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
    nodes: [],
    edges: [],
    selectedNodeId: null,
    isExecuting: false,
    executionResults: {},
    executionError: null,
    currentExecutingNodeId: null,
    sidebarOpen: true,
    propertiesOpen: true,
    currentPanel: SidebarPanelType.NODES,

    addNode: (type, position) => {
        const definition = NODE_DEFINITIONS[type];
        const id = `node_${++nodeIdCounter}`;

        const defaultParams: Record<string, any> = {};
        definition.params.forEach(p => {
            defaultParams[p.key] = p.default;
        });

        const newNode: NodeInstance = {
            id,
            type,
            position,
            data: {
                label: definition.name,
                params: defaultParams,
                status: 'idle'
            }
        };

        set(state => ({ nodes: [...state.nodes, newNode] }));
    },

    removeNode: (id) => {
        set(state => ({
            nodes: state.nodes.filter(n => n.id !== id),
            edges: state.edges.filter(e => e.source !== id && e.target !== id),
            selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId
        }));
    },

    updateNodePosition: (id, position) => {
        set(state => ({
            nodes: state.nodes.map(n => n.id === id ? { ...n, position } : n)
        }));
    },

    updateNodeParams: (id, params) => {
        set(state => ({
            nodes: state.nodes.map(n =>
                n.id === id
                    ? { ...n, data: { ...n.data, params: { ...n.data.params, ...params } } }
                    : n
            )
        }));
    },

    setSelectedNode: (id) => {
        set({ selectedNodeId: id });
    },

    addEdge: (edge) => {
        const id = `edge_${++edgeIdCounter}`;
        set(state => ({ edges: [ ...state.edges, { ...edge, id } ] }));
    },

    removeEdge: (id) => {
        set(state => ({ edges: state.edges.filter(e => e.id !== id) }));
    },

    setNodes: (nodes) => set({ nodes }),
    setEdges: (edges) => set({ edges }),

    updateNodeStatus: (id, status, result?, error?) => {
        set(state => ({
            nodes: state.nodes.map(n =>
                n.id === id
                    ? {
                        ...n,
                        data: {
                            ...n.data,
                            status,
                            result: result || n.data.result,
                            error: error || n.data.error
                        }
                    }
                    : n
            ),
            executionResults: result
                ? { ...state.executionResults, [id]: result }
                : state.executionResults
        }));
    },

    executeNode: async (id) => {
        const { nodes, edges, updateNodeStatus } = get();
        const node = nodes.find(n => n.id === id);

        if (!node) return;

        // 设置为运行中状态
        updateNodeStatus(id, 'running');
        set({ currentExecutingNodeId: id, executionError: null });

        try {
            // 收集输入数据（从上游节点的输出获取）
            const inputs: Record<string, any> = {};
            for (const edge of edges) {
                if (edge.target === id) {
                    const sourceResult = get().executionResults[edge.source];
                    if (sourceResult) {
                        inputs[edge.targetHandle] = sourceResult[edge.sourceHandle];
                    }
                }
            }

            // 调用后端 API 执行节点
            const response = await workflowApi.executeNode({
                id: node.id,
                type: node.type,
                position: node.position,
                data: node.data
            }, inputs);

            const result = response.data;

            if (result.status === 'success') {
                updateNodeStatus(id, 'success', result.output);
                set({ currentExecutingNodeId: null });
            } else {
                updateNodeStatus(id, 'error', null, result.error || '执行失败');
                set({ currentExecutingNodeId: null, executionError: result.error });
            }
        } catch (error: any) {
            const errorMessage = error.response?.data?.detail || error.message || '执行失败';
            updateNodeStatus(id, 'error', null, errorMessage);
            set({ currentExecutingNodeId: null, executionError: errorMessage });
            toast.error(`节点执行失败: ${errorMessage}`);
        }
    },

    executeAll: async () => {
        const { nodes, edges, updateNodeStatus } = get();

        if (nodes.length === 0) {
            toast.error('工作流为空，无法执行');
            return;
        }

        set({ isExecuting: true, executionError: null });

        // 重置所有节点状态
        nodes.forEach(n => updateNodeStatus(n.id, 'idle'));

        try {
            // 调用后端 API 执行整个工作流
            const response = await workflowApi.executeWorkflow(
                nodes.map(n => ({
                    id: n.id,
                    type: n.type,
                    position: n.position,
                    data: n.data
                })),
                edges.map(e => ({
                    id: e.id,
                    source: e.source,
                    source_handle: e.sourceHandle || '',
                    target: e.target,
                    target_handle: e.targetHandle || ''
                }))
            );

            const results = response.data;

            // 更新每个节点的执行结果
            for (const result of results) {
                if (result.status === 'success') {
                    updateNodeStatus(result.node_id, 'success', result.output);
                } else {
                    updateNodeStatus(result.node_id, 'error', null, result.error);
                }
            }

            // 统计执行结果
            const successCount = results.filter((r: { status: string }) => r.status === 'success').length;
            const errorCount = results.filter((r: { status: string }) => r.status === 'error').length;

            if (errorCount === 0) {
                toast.success(`工作流执行完成，${successCount} 个节点成功`);
            } else {
                toast.error(`工作流执行完成，${successCount} 成功，${errorCount} 失败`);
            }
        } catch (error: any) {
            const errorMessage = error.response?.data?.detail || error.message || '执行失败';
            set({ executionError: errorMessage });
            toast.error(`工作流执行失败: ${errorMessage}`);
        } finally {
            set({ isExecuting: false, currentExecutingNodeId: null });
        }
    },

    setSidebarOpen: (open) => set({ sidebarOpen: open }),
    setPropertiesOpen: (open) => set({ propertiesOpen: open }),
    setCurrentPanel: (panel) => set({ currentPanel: panel }),
    togglePanel: (panel) => {
        const { currentPanel, sidebarOpen } = get();
        if (currentPanel === panel && sidebarOpen) {
            // 如果点击的是当前已打开的面板，则关闭侧边栏
            set({ sidebarOpen: false });
        } else {
            // 否则打开侧边栏并切换到对应面板
            set({ currentPanel: panel, sidebarOpen: true });
        }
    },

    clearWorkflow: () => {
        set({
            nodes: [],
            edges: [],
            selectedNodeId: null,
            executionResults: {},
            executionError: null,
            currentExecutingNodeId: null,
            isExecuting: false
        });
    },

    loadWorkflow: (nodes, edges) => {
        set({
            nodes,
            edges,
            selectedNodeId: null,
            executionResults: {},
            executionError: null
        });
    }
}));