/**
 * 自定义工作流节点组件
 */

import React, { memo } from "react";
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import {
    Box,
    Typography,
    Chip,
    CircularProgress,
    Tooltip,
    alpha
} from '@mui/material';
import {
    PlayArrow,
    CheckCircle,
    Error,
    HourglassEmpty,
    Tune,
    Storage,
    ModelTraining,
    Visibility,
    AutoFixHigh,
    FilterAlt,
    Layers,
    Block,
    Flag,
    Psychology,
    Save
} from '@mui/icons-material';
import type { NodeInstance } from '../../types';
import { NodeType } from '../../types';
import { NODE_DEFINITIONS } from "../../store";

const iconMap: Record<NodeType, React.ReactElement> = {
    [NodeType.PARAMETERIZATION]: <Tune />,
    [NodeType.SIMULATION]: <PlayArrow />,
    [NodeType.OBJECTIVE]: <Flag />,
    [NodeType.FILTER]: <FilterAlt />,
    [NodeType.PROJECTION]: <Layers />,
    [NodeType.CONSTRAINT]: <Block />,
    [NodeType.MODEL_TRAIN]: <ModelTraining />,
    [NodeType.MODEL_INFER]: <Psychology />,
    [NodeType.DATA_LOAD]: <Storage />,
    [NodeType.DATA_SAVE]: <Save />,
    [NodeType.OPTIMIZER]: <AutoFixHigh />,
    [NodeType.OUTPUT]: <Visibility />
};

const categoryColors: Record<string, { bg: string; border: string; text: string }> = {
    '设计': { bg: '#1e3a5f', border: '#3b82f6', text: '#93c5fd' },
    '仿真': { bg: '#3d1f5c', border: '#8b5cf6', text: '#c4b5fd' },
    '优化': { bg: '#4a1d32', border: '#ef4444', text: '#fca5a5' },
    '处理': { bg: '#1e3a3a', border: '#14b8a6', text: '#5eead4' },
    '模型': { bg: '#1f2e1f', border: '#22c55e', text: '#86efac' },
    '数据': { bg: '#3d3520', border: '#eab308', text: '#fde047' },
    '输出': { bg: '#2d3748', border: '#64748b', text: '#cbd5e1' }
};

const statusIcons: Record<string, React.ReactElement> = {
    idle: <HourglassEmpty sx={{ fontSize: 14, color: '#64748b' }} />,
    running: <CircularProgress size={14} sx={{ color: '#3b82f6' }} />,
    success: <CheckCircle sx={{ fontSize: 14, color: '#22c55e' }} />,
    error: <Error sx={{ fontSize: 14, color: '#ef4444' }} />
};

interface WorkflowNodeProps extends NodeProps {
    data: NodeInstance['data'];
}

const WorkflowNode: React.FC<WorkflowNodeProps> = ({ type, data, selected }) => {
    const nodeType = type as NodeType;
    const definition = NODE_DEFINITIONS[nodeType];
    const colors = categoryColors[definition?.category || '输出'];

    return (
        <Box
            sx={{
                minWidth: 180,
                maxWidth: 250,
                background: `linear-gradient(135deg, ${alpha(colors.bg, 0.95)}, ${alpha(colors.bg, 0.85)})`,
                border: `2px solid ${selected ? '#fff' : colors.border}`,
                borderRadius: 2,
                boxShadow: selected
                    ? `0 0 20px ${alpha(colors.border, 0.5)}`
                    : `0 4px 12px rgba(0,0,0,0.3)`,
                transition: 'all 0.2s ease',
                backdropFilter: 'blur(10px)',
                '&:hover': {
                    boxShadow: `0 0 15px ${alpha(colors.border, 0.4)}`
                }
            }}
        >
            {/* 输入端口 */}
            {definition?.inputs.map((input, index) => (
                <Handle
                    key={input.id}
                    type="target"
                    position={Position.Left}
                    id={input.id}
                    style={{
                        top: 50 + index * 24,
                        background: colors.border,
                        width: 12,
                        height: 12,
                        border: '2px solid #fff'
                    }}
                />
            ))}

            {/* 节点头部 */}
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    p: 1,
                    borderBottom: `1px solid ${alpha(colors.border, 0.3)}`
                }}
            >
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        width: 28,
                        height: 28,
                        borderRadius: 1,
                        background: alpha(colors.border, 0.2),
                        colors: colors.text
                    }}
                >
                    {iconMap[nodeType]}
                </Box>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography
                        variant="body2"
                        sx={{
                            fontWeight: 600,
                            color: '#fff',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap'
                        }}
                    >
                        {data.label}
                    </Typography>
                    <Chip
                        label={definition?.category}
                        size="small"
                        sx={{
                            height: 16,
                            fontSize: '0.6rem',
                            background: alpha(colors.border, 0.3),
                            color: colors.text
                        }}
                    />
                </Box>
                <Tooltip title={data.status}>
                    {statusIcons[data.status]}
                </Tooltip>
            </Box>

            {/* 节点参数预览 */}
            <Box sx={{ p: 1 }}>
                {definition?.params.slice(0, 2).map(param => (
                    <Typography
                        key={param.key}
                        variant="caption"
                        sx={{
                            display: 'block',
                            color: alpha('#fff', 0.6),
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowarp'
                        }}
                    >
                        {param.label}: {String(data.params[param.key] ?? param.default)}
                    </Typography>
                ))}
                {definition && definition.params.length > 2 && (
                    <Typography variant="caption" sx={{ color: alpha('#fff', 0.4) }}>
                        +{definition.params.length - 2} 更多参数
                    </Typography>
                )}
            </Box>

            {/* 输出端口 */}
            {definition?.outputs.map((output, index) => (
                <Handle
                    key={output.id}
                    type="source"
                    position={Position.Right}
                    id={output.id}
                    style={{
                        top: 50 + index * 24,
                        background: colors.border,
                        width: 12,
                        height:12,
                        border: '2px solid #fff'
                    }}
                />
            ))}
        </Box>
    );
};

export default memo(WorkflowNode);