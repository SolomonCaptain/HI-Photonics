/**
 * 节点库侧边栏
 * 类似 ComfyUI 的节点选择面板
 */

import React, { useState } from "react";
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
    ListItemText,
    IconButton,
    Divider,
    Chip    
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
    Visibility,
    ChevronLeft,
    ChevronRight
} from '@mui/icons-material';
import { NodeType } from '../../types';
import { NODE_DEFINITIONS } from '../../store/workflowStore';
import { useWorkflowStore } from '../../store/workflowStore';

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

const categoryOrder = ['设计', '数据', '模型', '仿真', '处理', '优化', '输出'];

const NodeSidebar: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const { addNode, sidebarOpen, setSidebarOpen } = useWorkflowStore();

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

    if (!sidebarOpen) {
        return (
            <Box
                sx={{
                    width: 40,
                    background: 'linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)',
                    borderRight: '1px solid #2d3748',
                    display: 'flex',
                    alignItems: 'flex-start',
                    justifyContent: 'center',
                    pt: 2
                }}
            >
                <IconButton onClick={() => setSidebarOpen(true)} sx={{ color: '#64748b' }}>
                    <ChevronRight />
                </IconButton>
            </Box>
        );
    }

    return (
        <Box
            sx={{
                width: 280,
                background: 'linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)',
                borderRight: '1px solid #2d3748',
                display: 'flex',
                flexDirection: 'column',
                height: '100%',
                overflow: 'hidden'
            }}
        >
            {/* 标题栏 */}
            <Box
                sx={{
                    p: 2,
                    borderBottom: '1px solid #2d3748',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                }}
            >
                <Typography variant="h6" sx={{ color: '#e2e8f0', fontWeight: 600 }}>
                    节点库
                </Typography>
                <IconButton onClick={() => setSidebarOpen(false)} sx={{ color: '#64748b' }}>
                    <ChevronLeft />
                </IconButton>
            </Box>

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
                        '&.MuiOutlinedInput-root': {
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
                            <Typography
                                variant="subtitle2"
                                sx={{ color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase' }}
                            >
                                {category}
                            </Typography>
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
                                        <ListItemIcon sx={{ minWidth: 36, color: '#64748b' }}>
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

            {/* 快捷操作 */}
            <Divider sx={{ borderColor: '#2d3748' }} />
            <Box sx={{ p: 2 }}>
                <Typography variant="caption" sx={{ color: '#64748b', display: 'block', mb: 1 }}>
                    快捷模板
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                        label="HiLab 工作流"
                        size="small"
                        onClick={() => {}}
                        sx={{
                            background: 'rgba(34, 197, 94, 0.2)',
                            color: '#86efac',
                            border: '1px solid rgba(34, 197, 94, 0.3)',
                            '&:hover': { background: 'rgba(34, 197, 94, 0.3)' }
                        }}
                    />
                    <Chip
                        label="拓扑优化"
                        size="small"
                        onClick={() => {}}
                        sx={{
                            background: 'rgba(59, 130, 246, 0.2)',
                            color: '#93c5fd',
                            border: '1px solid rgba(59, 130, 246, 0.3)',
                            '&:hover': { background: 'rgba(59, 130, 246, 0.3)' }
                        }}
                    />
                </Box>
            </Box>
        </Box>
    );
};

export default NodeSidebar;