/**
 * 模板面板组件
 * 显示预设的工作流模板
 * 
 * 模板存储在 workflows/templates/ 目录下
 */

import React, { useState, useEffect } from 'react';
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
    Tooltip,
    CircularProgress,
    Divider
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
    Waves,
    Refresh
} from '@mui/icons-material';
import { useResourceStore, type WorkflowTemplate } from '../../store/resourceStore';
import { useWorkflowStore } from '../../store/workflowStore';
import toast from 'react-hot-toast';

// 图标映射
const iconMap: Record<string, React.ReactElement> = {
    'AutoFixHigh': <AutoFixHigh />,
    'ModelTraining': <ModelTraining />,
    'Science': <Science />,
    'Speed': <Speed />,
    'Gradient': <Gradient />,
    'Waves': <Waves />,
    'AccountTree': <AutoFixHigh />,
};

// 默认图标颜色
const defaultColors: string[] = [
    '#3b82f6', '#22c55e', '#8b5cf6', '#f59e0b', '#06b6d4', '#ef4444'
];

const TemplatesPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

    const {
        templates,
        templatesLoading,
        templatesError,
        fetchTemplates,
    } = useResourceStore();

    const {
        loadWorkflow,
        clearWorkflow,
    } = useWorkflowStore();

    // 初始化加载
    useEffect(() => {
        fetchTemplates();
    }, [fetchTemplates]);

    // 获取所有分类
    const categories = [...new Set(templates.map(t => t.category))];

    // 过滤模板
    const filteredTemplates = templates.filter(template => {
        const matchesSearch = searchQuery === '' ||
            template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            template.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            template.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
        const matchesCategory = selectedCategory === null || template.category === selectedCategory;
        return matchesSearch && matchesCategory;
    });

    const handleUseTemplate = (template: WorkflowTemplate) => {
        clearWorkflow();
        loadWorkflow(template.nodes, template.edges);
        toast.success(`已加载模板: ${template.name}`);
    };

    const handleRefresh = () => {
        fetchTemplates();
    };

    // 根据索引获取默认颜色
    const getColor = (index: number) => {
        return defaultColors[index % defaultColors.length];
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 搜索框 */}
            <Box sx={{ p: 1.5, display: 'flex', gap: 1 }}>
                <TextField
                    size="small"
                    placeholder="搜索模板..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <Search sx={{ color: '#64748b', fontSize: 18 }} />
                            </InputAdornment>
                        )
                    }}
                    sx={{
                        flex: 1,
                        '& .MuiOutlinedInput-root': {
                            background: 'rgba(255,255,255,0.05)',
                            borderRadius: 1.5,
                            '& fieldset': { borderColor: '#2d3748' },
                            '&:hover fieldset': { borderColor: '#4a5568' },
                            '& input': { color: '#e2e8f0', fontSize: '0.875rem' }
                        }
                    }}
                />
                <IconButton
                    size="small"
                    onClick={handleRefresh}
                    disabled={templatesLoading}
                    sx={{ color: '#64748b' }}
                >
                    {templatesLoading ? <CircularProgress size={18} /> : <Refresh fontSize="small" />}
                </IconButton>
            </Box>

            {/* 分类筛选 */}
            {categories.length > 0 && (
                <Box sx={{ px: 1.5, pb: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    <Chip
                        label="全部"
                        size="small"
                        onClick={() => setSelectedCategory(null)}
                        sx={{
                            height: 22,
                            fontSize: '0.7rem',
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
                                height: 22,
                                fontSize: '0.7rem',
                                background: selectedCategory === category ? 'rgba(139, 92, 246, 0.3)' : 'rgba(255,255,255,0.05)',
                                color: selectedCategory === category ? '#c4b5fd' : '#94a3b8',
                                border: selectedCategory === category ? '1px solid rgba(139, 92, 246, 0.5)' : '1px solid transparent',
                                '&:hover': { background: 'rgba(139, 92, 246, 0.2)' }
                            }}
                        />
                    ))}
                </Box>
            )}

            <Divider sx={{ borderColor: '#2d3748' }} />

            {/* 错误提示 */}
            {templatesError && (
                <Box sx={{ p: 2, background: 'rgba(239, 68, 68, 0.1)' }}>
                    <Typography variant="caption" sx={{ color: '#ef4444' }}>
                        {templatesError}
                    </Typography>
                </Box>
            )}

            {/* 模板网格 */}
            <Box sx={{ flex: 1, overflow: 'auto', p: 1.5, pt: 1 }}>
                {templatesLoading && filteredTemplates.length === 0 ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                        <CircularProgress size={24} />
                    </Box>
                ) : (
                    <Grid container spacing={1}>
                        {filteredTemplates.map((template, index) => {
                            const color = getColor(index);
                            return (
                                <Grid size={{ xs: 12 }} key={template.id}>
                                    <Card
                                        sx={{
                                            background: 'rgba(255,255,255,0.02)',
                                            border: '1px solid rgba(255,255,255,0.06)',
                                            borderRadius: 1.5,
                                            transition: 'all 0.2s ease',
                                            '&:hover': {
                                                background: 'rgba(255,255,255,0.05)',
                                                border: `1px solid ${color}40`,
                                                transform: 'translateY(-1px)'
                                            }
                                        }}
                                    >
                                        <CardContent sx={{ p: 1.25, '&:last-child': { pb: 1.25 } }}>
                                            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.25 }}>
                                                {/* 图标 */}
                                                <Box
                                                    sx={{
                                                        width: 32,
                                                        height: 32,
                                                        borderRadius: 1,
                                                        background: `linear-gradient(135deg, ${color}25 0%, ${color}10 100%)`,
                                                        border: `1px solid ${color}30`,
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center',
                                                        color: color,
                                                        flexShrink: 0
                                                    }}
                                                >
                                                    {iconMap[template.icon] || <AutoFixHigh sx={{ fontSize: 18 }} />}
                                                </Box>

                                                {/* 内容 */}
                                                <Box sx={{ flex: 1, minWidth: 0 }}>
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.25 }}>
                                                        <Typography
                                                            variant="subtitle2"
                                                            sx={{ color: '#e2e8f0', fontWeight: 600, fontSize: '0.8rem' }}
                                                        >
                                                            {template.name}
                                                        </Typography>
                                                        <Chip
                                                            label={template.category}
                                                            size="small"
                                                            sx={{
                                                                height: 14,
                                                                fontSize: '0.55rem',
                                                                background: `${color}15`,
                                                                color: color
                                                            }}
                                                        />
                                                    </Box>
                                                    {template.description && (
                                                        <Typography
                                                            variant="caption"
                                                            sx={{ color: '#64748b', display: 'block', mb: 0.5, fontSize: '0.7rem' }}
                                                        >
                                                            {template.description}
                                                        </Typography>
                                                    )}
                                                    {template.tags && template.tags.length > 0 && (
                                                        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                                            {template.tags.map(tag => (
                                                                <Chip
                                                                    key={tag}
                                                                    label={tag}
                                                                    size="small"
                                                                    sx={{
                                                                        height: 14,
                                                                        fontSize: '0.55rem',
                                                                        background: 'rgba(255,255,255,0.05)',
                                                                        color: '#94a3b8'
                                                                    }}
                                                                />
                                                            ))}
                                                        </Box>
                                                    )}
                                                </Box>
                                            </Box>
                                        </CardContent>
                                        <CardActions sx={{ p: 0.75, pt: 0, justifyContent: 'flex-end' }}>
                                            <Tooltip title="查看详情">
                                                <IconButton size="small" sx={{ color: '#64748b', p: 0.5 }}>
                                                    <Info fontSize="small" />
                                                </IconButton>
                                            </Tooltip>
                                            <Tooltip title="使用模板">
                                                <IconButton
                                                    size="small"
                                                    onClick={() => handleUseTemplate(template)}
                                                    sx={{
                                                        color: color,
                                                        background: `${color}15`,
                                                        p: 0.5,
                                                        '&:hover': { background: `${color}25` }
                                                    }}
                                                >
                                                    <PlayArrow fontSize="small" />
                                                </IconButton>
                                            </Tooltip>
                                        </CardActions>
                                    </Card>
                                </Grid>
                            );
                        })}
                    </Grid>
                )}

                {!templatesLoading && filteredTemplates.length === 0 && (
                    <Box sx={{ p: 3, textAlign: 'center' }}>
                        <Typography variant="body2" sx={{ color: '#64748b' }}>
                            {searchQuery || selectedCategory ? '没有找到匹配的模板' : '暂无模板'}
                        </Typography>
                    </Box>
                )}
            </Box>

            {/* 底部统计 */}
            <Divider sx={{ borderColor: '#2d3748' }} />
            <Box sx={{ px: 1.5, py: 0.75, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.7rem' }}>
                    共 {templates.length} 个模板
                </Typography>
            </Box>
        </Box>
    );
};

export default TemplatesPanel;