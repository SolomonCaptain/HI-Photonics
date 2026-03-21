/**
 * 模型面板组件
 * 显示逆向设计相关模型（TNN、MDN、CGAN、PINN、GNN、HiLab）
 */

import React, { useState } from 'react';
import {
    Box,
    Typography,
    TextField,
    InputAdornment,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    ListItemSecondaryAction,
    IconButton,
    Chip,
    LinearProgress,
    Divider
} from '@mui/material';
import {
    Search,
    AccountTree,
    Functions,
    AutoAwesome,
    Science,
    Share,
    Tune,
    PlayArrow,
    Info,
    MoreVert
} from '@mui/icons-material';
import type { ModelType } from '../../types/models';

// 模型信息
interface ModelInfoItem {
    type: ModelType;
    name: string;
    description: string;
    icon: React.ReactElement;
    color: string;
    tags: string[];
    status: 'available' | 'training' | 'error';
    progress?: number;
}

const modelInfos: ModelInfoItem[] = [
    {
        type: 'tnn',
        name: 'TNN',
        description: '串联神经网络，快速逆向设计',
        icon: <AccountTree />,
        color: '#3b82f6',
        tags: ['快速', '轻量'],
        status: 'available'
    },
    {
        type: 'mdn',
        name: 'MDN',
        description: '混合密度网络，概率分布输出',
        icon: <Functions />,
        color: '#8b5cf6',
        tags: ['概率', '多解'],
        status: 'available'
    },
    {
        type: 'cgan',
        name: 'CGAN',
        description: '条件生成对抗网络，高质量设计',
        icon: <AutoAwesome />,
        color: '#06b6d4',
        tags: ['生成', '多样'],
        status: 'training',
        progress: 67
    },
    {
        type: 'pinn',
        name: 'PINN',
        description: '物理信息神经网络，约束满足',
        icon: <Science />,
        color: '#22c55e',
        tags: ['物理约束', '可解释'],
        status: 'available'
    },
    {
        type: 'gnn',
        name: 'GNN',
        description: '图神经网络，结构化设计',
        icon: <Share />,
        color: '#f59e0b',
        tags: ['图结构', '灵活'],
        status: 'error'
    },
    {
        type: 'hilab',
        name: 'HiLab',
        description: 'VAE + 贝叶斯优化，探索式设计',
        icon: <Tune />,
        color: '#ef4444',
        tags: ['探索', '优化'],
        status: 'available'
    }
];

const ModelsPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedModel, setSelectedModel] = useState<string | null>(null);

    // 过滤模型
    const filteredModels = modelInfos.filter(model =>
        searchQuery === '' ||
        model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    );

    // 状态颜色映射
    const statusColors: Record<string, string> = {
        'available': '#22c55e',
        'training': '#f59e0b',
        'error': '#ef4444'
    };

    // 状态文本映射
    const statusText: Record<string, string> = {
        'available': '可用',
        'training': '训练中',
        'error': '错误'
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 搜索框 */}
            <Box sx={{ p: 2 }}>
                <TextField
                    size="small"
                    placeholder="搜索模型..."
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

            <Divider sx={{ borderColor: '#2d3748' }} />

            {/* 模型列表 */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
                <List dense disablePadding>
                    {filteredModels.map((model) => (
                        <ListItem
                            key={model.type}
                            onClick={() => setSelectedModel(model.type)}
                            sx={{
                                px: 2,
                                py: 1.5,
                                borderBottom: '1px solid rgba(45, 55, 72, 0.5)',
                                cursor: 'pointer',
                                background: selectedModel === model.type
                                    ? 'rgba(59, 130, 246, 0.1)'
                                    : 'transparent',
                                '&:hover': {
                                    background: 'rgba(255,255,255,0.03)'
                                }
                            }}
                        >
                            <ListItemIcon sx={{ minWidth: 40 }}>
                                <Box
                                    sx={{
                                        width: 32,
                                        height: 32,
                                        borderRadius: 1.5,
                                        background: `${model.color}20`,
                                        border: `1px solid ${model.color}40`,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        color: model.color
                                    }}
                                >
                                    {model.icon}
                                </Box>
                            </ListItemIcon>
                            <ListItemText
                                primary={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography sx={{ color: '#e2e8f0', fontWeight: 500 }}>
                                            {model.name}
                                        </Typography>
                                        <Chip
                                            label={statusText[model.status]}
                                            size="small"
                                            sx={{
                                                height: 18,
                                                fontSize: '0.65rem',
                                                background: `${statusColors[model.status]}20`,
                                                color: statusColors[model.status]
                                            }}
                                        />
                                    </Box>
                                }
                                secondary={
                                    <Box sx={{ mt: 0.5 }}>
                                        <Typography variant="caption" sx={{ color: '#64748b', display: 'block' }}>
                                            {model.description}
                                        </Typography>
                                        <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
                                            {model.tags.map(tag => (
                                                <Chip
                                                    key={tag}
                                                    label={tag}
                                                    size="small"
                                                    sx={{
                                                        height: 16,
                                                        fontSize: '0.6rem',
                                                        background: 'rgba(255,255,255,0.05)',
                                                        color: '#94a3b8'
                                                    }}
                                                />
                                            ))}
                                        </Box>
                                        {model.status === 'training' && model.progress && (
                                            <Box sx={{ mt: 1 }}>
                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                                    <Typography variant="caption" sx={{ color: '#f59e0b' }}>
                                                        训练进度
                                                    </Typography>
                                                    <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                                                        {model.progress}%
                                                    </Typography>
                                                </Box>
                                                <LinearProgress
                                                    variant="determinate"
                                                    value={model.progress}
                                                    sx={{
                                                        height: 4,
                                                        borderRadius: 2,
                                                        background: 'rgba(255,255,255,0.1)',
                                                        '& .MuiLinearProgress-bar': {
                                                            background: `linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%)`,
                                                            borderRadius: 2
                                                        }
                                                    }}
                                                />
                                            </Box>
                                        )}
                                    </Box>
                                }
                                secondaryTypographyProps={{ component: 'div' }}
                            />
                            <ListItemSecondaryAction>
                                <IconButton size="small" sx={{ color: '#64748b' }}>
                                    <MoreVert fontSize="small" />
                                </IconButton>
                            </ListItemSecondaryAction>
                        </ListItem>
                    ))}
                </List>

                {filteredModels.length === 0 && (
                    <Box sx={{ p: 4, textAlign: 'center' }}>
                        <Typography sx={{ color: '#64748b' }}>
                            没有找到匹配的模型
                        </Typography>
                    </Box>
                )}
            </Box>

            {/* 底部操作 */}
            <Divider sx={{ borderColor: '#2d3748' }} />
            <Box sx={{ p: 2 }}>
                <Typography variant="caption" sx={{ color: '#64748b' }}>
                    点击模型查看详情或拖放到画布使用
                </Typography>
            </Box>
        </Box>
    );
};

export default ModelsPanel;
