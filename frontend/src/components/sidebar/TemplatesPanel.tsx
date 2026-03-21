/**
 * 模板面板组件
 * 显示预设的工作流模板
 */

import React, { useState } from 'react';
import {
    Box,
    Typography,
    TextField,
    InputAdornment,
    Grid,
    Card,
    CardContent,
    CardActions,
    Chip,
    IconButton,
    Tooltip
} from '@mui/material';
import {
    Search,
    PlayArrow,
    Info,
    AutoFixHigh,
    ModelTraining,
    Science,
    Speed,
    Gradient,
    Waves
} from '@mui/icons-material';
import type { WorkflowTemplate } from '../../types';

// 模板数据
const mockTemplates: (WorkflowTemplate & { color: string; difficulty: 'beginner' | 'intermediate' | 'advanced' })[] = [
    {
        id: 'tpl_1',
        name: '基础逆向设计',
        description: '使用神经网络进行基础的逆向设计流程',
        category: '基础',
        icon: 'AutoFixHigh',
        color: '#3b82f6',
        difficulty: 'beginner',
        nodes: [],
        edges: [],
        tags: ['入门', '快速']
    },
    {
        id: 'tpl_2',
        name: 'HiLab 工作流',
        description: 'VAE + 贝叶斯优化的探索式逆向设计',
        category: '高级',
        icon: 'Science',
        color: '#22c55e',
        difficulty: 'advanced',
        nodes: [],
        edges: [],
        tags: ['探索', '优化']
    },
    {
        id: 'tpl_3',
        name: '拓扑优化',
        description: '基于伴随方法的拓扑优化工作流',
        category: '优化',
        icon: 'Gradient',
        color: '#8b5cf6',
        difficulty: 'intermediate',
        nodes: [],
        edges: [],
        tags: ['伴随', '梯度']
    },
    {
        id: 'tpl_4',
        name: '模型训练流程',
        description: '从数据加载到模型训练的完整流程',
        category: '训练',
        icon: 'ModelTraining',
        color: '#f59e0b',
        difficulty: 'intermediate',
        nodes: [],
        edges: [],
        tags: ['训练', '数据']
    },
    {
        id: 'tpl_5',
        name: '快速仿真',
        description: '简化的仿真流程，快速验证设计',
        category: '仿真',
        icon: 'Speed',
        color: '#06b6d4',
        difficulty: 'beginner',
        nodes: [],
        edges: [],
        tags: ['快速', '仿真']
    },
    {
        id: 'tpl_6',
        name: '波分复用器设计',
        description: '完整的波分复用器逆向设计流程',
        category: '应用',
        icon: 'Waves',
        color: '#ef4444',
        difficulty: 'advanced',
        nodes: [],
        edges: [],
        tags: ['WDM', '实用']
    }
];

// 图标映射
const iconMap: Record<string, React.ReactElement> = {
    'AutoFixHigh': <AutoFixHigh />,
    'ModelTraining': <ModelTraining />,
    'Science': <Science />,
    'Speed': <Speed />,
    'Gradient': <Gradient />,
    'Waves': <Waves />
};

// 难度颜色映射
const difficultyColors: Record<string, string> = {
    'beginner': '#22c55e',
    'intermediate': '#f59e0b',
    'advanced': '#ef4444'
};

// 难度文本映射
const difficultyText: Record<string, string> = {
    'beginner': '入门',
    'intermediate': '进阶',
    'advanced': '高级'
};

const TemplatesPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

    // 获取所有分类
    const categories = [...new Set(mockTemplates.map(t => t.category))];

    // 过滤模板
    const filteredTemplates = mockTemplates.filter(template => {
        const matchesSearch = searchQuery === '' ||
            template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
            template.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
        const matchesCategory = selectedCategory === null || template.category === selectedCategory;
        return matchesSearch && matchesCategory;
    });

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 搜索框 */}
            <Box sx={{ p: 2 }}>
                <TextField
                    size="small"
                    placeholder="搜索模板..."
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

            {/* 分类筛选 */}
            <Box sx={{ px: 2, pb: 2, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                <Chip
                    label="全部"
                    size="small"
                    onClick={() => setSelectedCategory(null)}
                    sx={{
                        background: selectedCategory === null ? 'rgba(59, 130, 246, 0.3)' : 'rgba(255,255,255,0.05)',
                        color: selectedCategory === null ? '#93c5fd' : '#94a3b8',
                        border: selectedCategory === null ? '1px solid rgba(59, 130, 246, 0.5)' : '1px solid transparent',
                        '&:hover': { background: 'rgba(59, 130, 246, 0.2)' }
                    }}
                />
                {categories.map(category => (
                    <Chip
                        key={category}
                        label={category}
                        size="small"
                        onClick={() => setSelectedCategory(category)}
                        sx={{
                            background: selectedCategory === category ? 'rgba(139, 92, 246, 0.3)' : 'rgba(255,255,255,0.05)',
                            color: selectedCategory === category ? '#c4b5fd' : '#94a3b8',
                            border: selectedCategory === category ? '1px solid rgba(139, 92, 246, 0.5)' : '1px solid transparent',
                            '&:hover': { background: 'rgba(139, 92, 246, 0.2)' }
                        }}
                    />
                ))}
            </Box>

            {/* 模板网格 */}
            <Box sx={{ flex: 1, overflow: 'auto', p: 2, pt: 0 }}>
                <Grid container spacing={1.5}>
                    {filteredTemplates.map((template) => (
                        <Grid size={{ xs: 12 }} key={template.id}>
                            <Card
                                sx={{
                                    background: 'rgba(255,255,255,0.02)',
                                    border: '1px solid rgba(255,255,255,0.06)',
                                    borderRadius: 2,
                                    transition: 'all 0.2s ease',
                                    '&:hover': {
                                        background: 'rgba(255,255,255,0.05)',
                                        border: `1px solid ${template.color}40`,
                                        transform: 'translateY(-2px)'
                                    }
                                }}
                            >
                                <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
                                        {/* 图标 */}
                                        <Box
                                            sx={{
                                                width: 40,
                                                height: 40,
                                                borderRadius: 1.5,
                                                background: `linear-gradient(135deg, ${template.color}30 0%, ${template.color}15 100%)`,
                                                border: `1px solid ${template.color}30`,
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                color: template.color,
                                                flexShrink: 0
                                            }}
                                        >
                                            {iconMap[template.icon] || <AutoFixHigh />}
                                        </Box>

                                        {/* 内容 */}
                                        <Box sx={{ flex: 1, minWidth: 0 }}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                                                <Typography
                                                    variant="subtitle2"
                                                    sx={{ color: '#e2e8f0', fontWeight: 600 }}
                                                >
                                                    {template.name}
                                                </Typography>
                                                <Chip
                                                    label={difficultyText[template.difficulty]}
                                                    size="small"
                                                    sx={{
                                                        height: 16,
                                                        fontSize: '0.6rem',
                                                        background: `${difficultyColors[template.difficulty]}20`,
                                                        color: difficultyColors[template.difficulty]
                                                    }}
                                                />
                                            </Box>
                                            <Typography
                                                variant="caption"
                                                sx={{ color: '#64748b', display: 'block', mb: 0.5 }}
                                            >
                                                {template.description}
                                            </Typography>
                                            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                                {template.tags?.map(tag => (
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
                                        </Box>
                                    </Box>
                                </CardContent>
                                <CardActions sx={{ p: 1, pt: 0, justifyContent: 'flex-end' }}>
                                    <Tooltip title="查看详情">
                                        <IconButton size="small" sx={{ color: '#64748b' }}>
                                            <Info fontSize="small" />
                                        </IconButton>
                                    </Tooltip>
                                    <Tooltip title="使用模板">
                                        <IconButton
                                            size="small"
                                            sx={{
                                                color: template.color,
                                                background: `${template.color}15`,
                                                '&:hover': { background: `${template.color}25` }
                                            }}
                                        >
                                            <PlayArrow fontSize="small" />
                                        </IconButton>
                                    </Tooltip>
                                </CardActions>
                            </Card>
                        </Grid>
                    ))}
                </Grid>

                {filteredTemplates.length === 0 && (
                    <Box sx={{ p: 4, textAlign: 'center' }}>
                        <Typography sx={{ color: '#64748b' }}>
                            没有找到匹配的模板
                        </Typography>
                    </Box>
                )}
            </Box>
        </Box>
    );
};

export default TemplatesPanel;
