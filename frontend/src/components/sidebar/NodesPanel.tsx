/**
 * 节点库面板组件
 * 显示可用的节点类型，支持拖放到画布
 */

import React, { useState } from 'react';
import {
    Box,
    Typography,
    TextField,
    InputAdornment,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    List,
    ListItem,
    ListItemIcon,
    ListItemText
} from '@mui/material';
import {
    Search,
    ExpandMore,
    Tune,
    PlayArrow,
    Flag,
    FilterAlt,
    Layers,
    Block,
    ModelTraining,
    Psychology,
    Storage,
    Save,
    AutoFixHigh,
    Visibility
} from '@mui/icons-material';
import { NodeType } from '../../types';
import { NODE_DEFINITIONS } from '../../store/workflowStore';
import { useWorkflowStore } from '../../store/workflowStore';

// 节点类型图标映射
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

// 节点分类颜色映射
const categoryColors: Record<string, string> = {
    '设计': '#3b82f6',
    '数据': '#8b5cf6',
    '模型': '#06b6d4',
    '仿真': '#22c55e',
    '处理': '#f59e0b',
    '优化': '#ef4444',
    '输出': '#ec4899'
};

const categoryOrder = ['设计', '数据', '模型', '仿真', '处理', '优化', '输出'];

const NodesPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const { addNode } = useWorkflowStore();

    // 分组节点
    const groupedNodes = categoryOrder.reduce((acc, category) => {
        const nodes = Object.entries(NODE_DEFINITIONS)
            .filter(([_, def]) => def.category === category)
            .filter(([_, def]) =>
                searchQuery === '' ||
                def.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                def.description.toLowerCase().includes(searchQuery.toLowerCase())
            );
        if (nodes.length > 0) {
            acc[category] = nodes;
        }
        return acc;
    }, {} as Record<string, [string, typeof NODE_DEFINITIONS[NodeType]][]>);

    const handleDragStart = (e: React.DragEvent, nodeType: NodeType) => {
        e.dataTransfer.setData('application/reactflow', nodeType);
        e.dataTransfer.effectAllowed = 'move';
    };

    const handleAddNode = (nodeType: NodeType) => {
        addNode(nodeType, { x: 300, y: 200 });
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 搜索框 */}
            <Box sx={{ p: 2 }}>
                <TextField
                    size="small"
                    placeholder="搜索节点..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <Search sx={{ color: '#64748b', fontSize: 20 }} />
                            </InputAdornment>
                        )
                    }}
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            background: 'rgba(255,255,255,0.05)',
                            borderRadius: 2,
                            '& fieldset': { borderColor: '#2d3748' },
                            '&:hover fieldset': { borderColor: '#4a5568' },
                            '& input': { color: '#e2e8f0' }
                        }
                    }}
                />
            </Box>

            {/* 节点列表 */}
            <Box sx={{ flex: 1, overflow: 'auto', px: 1, pb: 2 }}>
                {Object.entries(groupedNodes).map(([category, nodes]) => (
                    <Accordion
                        key={category}
                        defaultExpanded
                        sx={{
                            background: 'transparent',
                            boxShadow: 'none',
                            '&:before': { display: 'none' },
                            '& .MuiAccordionSummary-root': {
                                minHeight: 36,
                                px: 1
                            }
                        }}
                    >
                        <AccordionSummary
                            expandIcon={<ExpandMore sx={{ color: '#64748b' }} />}
                            sx={{ '& .MuiAccordionSummary-content': { my: 0.5 } }}
                        >
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Box
                                    sx={{
                                        width: 8,
                                        height: 8,
                                        borderRadius: '50%',
                                        background: categoryColors[category] || '#64748b'
                                    }}
                                />
                                <Typography
                                    variant="subtitle2"
                                    sx={{ color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase' }}
                                >
                                    {category}
                                </Typography>
                                <Typography
                                    variant="caption"
                                    sx={{ color: '#64748b', ml: 0.5 }}
                                >
                                    ({nodes.length})
                                </Typography>
                            </Box>
                        </AccordionSummary>
                        <AccordionDetails sx={{ p: 0 }}>
                            <List dense disablePadding>
                                {nodes.map(([key, definition]) => (
                                    <ListItem
                                        key={key}
                                        draggable
                                        onDragStart={(e) => handleDragStart(e, definition.type)}
                                        onClick={() => handleAddNode(definition.type)}
                                        sx={{
                                            borderRadius: 1,
                                            mx: 1,
                                            mb: 0.5,
                                            cursor: 'grab',
                                            background: 'rgba(255,255,255,0.03)',
                                            border: '1px solid transparent',
                                            '&:hover': {
                                                background: 'rgba(255,255,255,0.08)',
                                                border: '1px solid rgba(255,255,255,0.1)'
                                            },
                                            '&:active': {
                                                cursor: 'grabbing'
                                            }
                                        }}
                                    >
                                        <ListItemIcon sx={{ minWidth: 36, color: categoryColors[definition.category] || '#64748b' }}>
                                            {iconMap[definition.type]}
                                        </ListItemIcon>
                                        <ListItemText
                                            primary={definition.name}
                                            secondary={definition.description}
                                            primaryTypographyProps={{
                                                sx: { color: '#e2e8f0', fontSize: '0.875rem' }
                                            }}
                                            secondaryTypographyProps={{
                                                sx: { color: '#64748b', fontSize: '0.7rem' },
                                                noWrap: true
                                            }}
                                        />
                                    </ListItem>
                                ))}
                            </List>
                        </AccordionDetails>
                    </Accordion>
                ))}
            </Box>
        </Box>
    );
};

export default NodesPanel;
