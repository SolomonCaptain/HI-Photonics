/**
 * 模型面板组件
 * 显示逆向设计相关模型（TNN、MDN、CGAN、PINN、GNN、HiLab）
 * 
 * 模型存储在 op_models/ 目录下：
 * - op_models/pretrained/: 预训练模型
 * - op_models/custom/: 自定义模型
 */

import React, { useState, useEffect } from 'react';
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
    Divider,
    CircularProgress,
    Menu,
    MenuItem,
    Tooltip
} from '@mui/material';
import {
    Search,
    AccountTree,
    Functions,
    AutoAwesome,
    Science,
    Share,
    Tune,
    MoreVert,
    Delete,
    Download,
    Refresh,
    CheckCircle
} from '@mui/icons-material';
import { useResourceStore, type ModelInfo, type ModelType } from '../../store/resourceStore';
import toast from 'react-hot-toast';

// 模型图标映射
const modelTypeIcons: Record<ModelType, React.ReactElement> = {
    tnn: <AccountTree />,
    mdn: <Functions />,
    cgan: <AutoAwesome />,
    pinn: <Science />,
    gnn: <Share />,
    hilab: <Tune />
};

// 模型名称映射
const modelTypeNames: Record<ModelType, string> = {
    tnn: 'TNN',
    mdn: 'MDN',
    cgan: 'CGAN',
    pinn: 'PINN',
    gnn: 'GNN',
    hilab: 'HiLab'
};

// 模型描述映射
const modelTypeDescriptions: Record<ModelType, string> = {
    tnn: '串联神经网络，快速逆向设计',
    mdn: '混合密度网络，概率分布输出',
    cgan: '条件生成对抗网络，高质量设计',
    pinn: '物理信息神经网络，约束满足',
    gnn: '图神经网络，结构化设计',
    hilab: 'VAE + 贝叶斯优化，探索式设计'
};

// 模型颜色映射
const modelTypeColors: Record<ModelType, string> = {
    tnn: '#3b82f6',
    mdn: '#8b5cf6',
    cgan: '#06b6d4',
    pinn: '#22c55e',
    gnn: '#f59e0b',
    hilab: '#ef4444'
};

// 挑战名称映射
const challengeNames: Record<string, string> = {
    grating_coupler: '光栅耦合器',
    metagrating: '超构光栅',
    wavelength_demux: '波分复用器'
};

const ModelsPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const [filterPretrained, setFilterPretrained] = useState<boolean | null>(null);

    const {
        models,
        modelsLoading,
        modelsError,
        fetchModels,
        deleteModel,
        downloadModel,
        setSelectedModel,
    } = useResourceStore();

    // 初始化加载
    useEffect(() => {
        fetchModels();
    }, [fetchModels]);

    // 过滤模型
    const filteredModels = models.filter(model => {
        const matchesSearch = searchQuery === '' ||
            model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            modelTypeDescriptions[model.type].toLowerCase().includes(searchQuery.toLowerCase());
        const matchesPretrained = filterPretrained === null || model.isPretrained === filterPretrained;
        return matchesSearch && matchesPretrained;
    });

    // 按类型分组
    const groupedModels = filteredModels.reduce((acc, model) => {
        const type = model.type;
        if (!acc[type]) acc[type] = [];
        acc[type].push(model);
        return acc;
    }, {} as Record<ModelType, ModelInfo[]>);

    const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, modelId: string) => {
        event.stopPropagation();
        setAnchorEl(event.currentTarget);
        setSelectedModelId(modelId);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
        setSelectedModelId(null);
    };

    const handleDelete = async () => {
        if (!selectedModelId) return;
        try {
            await deleteModel(selectedModelId);
            toast.success('模型已删除');
        } catch (error) {
            toast.error('删除失败');
        }
        handleMenuClose();
    };

    const handleDownload = async () => {
        if (!selectedModelId) return;
        try {
            await downloadModel(selectedModelId);
            toast.success('下载已开始');
        } catch (error) {
            toast.error('下载失败');
        }
        handleMenuClose();
    };

    const handleRefresh = () => {
        fetchModels();
    };

    // 格式化文件大小
    const formatSize = (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 搜索框 */}
            <Box sx={{ p: 1.5, display: 'flex', gap: 1 }}>
                <TextField
                    size="small"
                    placeholder="搜索模型..."
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
                    disabled={modelsLoading}
                    sx={{ color: '#64748b' }}
                >
                    {modelsLoading ? <CircularProgress size={18} /> : <Refresh fontSize="small" />}
                </IconButton>
            </Box>

            {/* 筛选 */}
            <Box sx={{ px: 1.5, pb: 1, display: 'flex', gap: 0.5 }}>
                <Chip
                    label="全部"
                    size="small"
                    onClick={() => setFilterPretrained(null)}
                    sx={{
                        height: 22,
                        fontSize: '0.7rem',
                        background: filterPretrained === null ? 'rgba(59, 130, 246, 0.3)' : 'rgba(255,255,255,0.05)',
                        color: filterPretrained === null ? '#93c5fd' : '#94a3b8',
                        '&:hover': { background: 'rgba(59, 130, 246, 0.2)' }
                    }}
                />
                <Chip
                    label="预训练"
                    size="small"
                    onClick={() => setFilterPretrained(true)}
                    sx={{
                        height: 22,
                        fontSize: '0.7rem',
                        background: filterPretrained === true ? 'rgba(34, 197, 94, 0.3)' : 'rgba(255,255,255,0.05)',
                        color: filterPretrained === true ? '#86efac' : '#94a3b8',
                        '&:hover': { background: 'rgba(34, 197, 94, 0.2)' }
                    }}
                />
                <Chip
                    label="自定义"
                    size="small"
                    onClick={() => setFilterPretrained(false)}
                    sx={{
                        height: 22,
                        fontSize: '0.7rem',
                        background: filterPretrained === false ? 'rgba(245, 158, 11, 0.3)' : 'rgba(255,255,255,0.05)',
                        color: filterPretrained === false ? '#fcd34d' : '#94a3b8',
                        '&:hover': { background: 'rgba(245, 158, 11, 0.2)' }
                    }}
                />
            </Box>

            <Divider sx={{ borderColor: '#2d3748' }} />

            {/* 错误提示 */}
            {modelsError && (
                <Box sx={{ p: 2, background: 'rgba(239, 68, 68, 0.1)' }}>
                    <Typography variant="caption" sx={{ color: '#ef4444' }}>
                        {modelsError}
                    </Typography>
                </Box>
            )}

            {/* 模型列表 */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
                {modelsLoading && filteredModels.length === 0 ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                        <CircularProgress size={24} />
                    </Box>
                ) : (
                    <List dense disablePadding>
                        {Object.entries(groupedModels).map(([type, modelsOfType]) => (
                            <Box key={type}>
                                {/* 类型标题 */}
                                <Box sx={{ px: 1.5, py: 0.75, background: 'rgba(255,255,255,0.02)' }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Box
                                            sx={{
                                                width: 20,
                                                height: 20,
                                                borderRadius: 0.75,
                                                background: `${modelTypeColors[type as ModelType]}20`,
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                color: modelTypeColors[type as ModelType]
                                            }}
                                        >
                                            {React.cloneElement(modelTypeIcons[type as ModelType] as React.ReactElement<{ fontSize?: string }>, { fontSize: 'small' })}
                                        </Box>
                                        <Typography variant="caption" sx={{ color: '#94a3b8', fontWeight: 500 }}>
                                            {modelTypeNames[type as ModelType]}
                                        </Typography>
                                        <Typography variant="caption" sx={{ color: '#64748b' }}>
                                            ({modelsOfType.length})
                                        </Typography>
                                    </Box>
                                </Box>

                                {/* 该类型的模型列表 */}
                                {modelsOfType.map((model) => (
                                    <ListItem
                                        key={model.id}
                                        onClick={() => {
                                            setSelectedModel(model);
                                            setSelectedModelId(model.id);
                                        }}
                                        sx={{
                                            px: 1.5,
                                            py: 0.75,
                                            borderBottom: '1px solid rgba(45, 55, 72, 0.3)',
                                            cursor: 'pointer',
                                            background: selectedModelId === model.id
                                                ? 'rgba(59, 130, 246, 0.1)'
                                                : 'transparent',
                                            '&:hover': {
                                                background: 'rgba(255,255,255,0.03)'
                                            }
                                        }}
                                    >
                                        <ListItemIcon sx={{ minWidth: 32 }}>
                                            <Tooltip title={model.isPretrained ? '预训练模型' : '自定义模型'}>
                                                <Box
                                                    sx={{
                                                        width: 24,
                                                        height: 24,
                                                        borderRadius: 1,
                                                        background: model.isPretrained
                                                            ? 'rgba(34, 197, 94, 0.15)'
                                                            : 'rgba(245, 158, 11, 0.15)',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center',
                                                        color: model.isPretrained ? '#22c55e' : '#f59e0b'
                                                    }}
                                                >
                                                    {model.isPretrained ? <CheckCircle sx={{ fontSize: 14 }} /> : <Tune sx={{ fontSize: 14 }} />}
                                                </Box>
                                            </Tooltip>
                                        </ListItemIcon>
                                        <ListItemText
                                            primary={
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                    <Typography sx={{ color: '#e2e8f0', fontSize: '0.8rem', fontWeight: 500 }}>
                                                        {model.name}
                                                    </Typography>
                                                    <Chip
                                                        label={challengeNames[model.challenge] || model.challenge}
                                                        size="small"
                                                        sx={{
                                                            height: 16,
                                                            fontSize: '0.6rem',
                                                            background: 'rgba(139, 92, 246, 0.2)',
                                                            color: '#c4b5fd'
                                                        }}
                                                    />
                                                </Box>
                                            }
                                            secondary={
                                                <Box sx={{ mt: 0.25 }}>
                                                    <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.65rem' }}>
                                                        {model.description || modelTypeDescriptions[model.type]}
                                                    </Typography>
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.25 }}>
                                                        <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.6rem' }}>
                                                            {formatSize(model.fileSize)}
                                                        </Typography>
                                                        {model.metrics && Object.entries(model.metrics).slice(0, 2).map(([key, value]) => (
                                                            <Typography key={key} variant="caption" sx={{ color: '#94a3b8', fontSize: '0.6rem' }}>
                                                                {key}: {typeof value === 'number' ? value.toFixed(3) : value}
                                                            </Typography>
                                                        ))}
                                                    </Box>
                                                </Box>
                                            }
                                            secondaryTypographyProps={{ component: 'div' }}
                                        />
                                        <ListItemSecondaryAction>
                                            <IconButton
                                                size="small"
                                                onClick={(e) => handleMenuOpen(e, model.id)}
                                                sx={{ color: '#64748b', p: 0.5 }}
                                            >
                                                <MoreVert fontSize="small" />
                                            </IconButton>
                                        </ListItemSecondaryAction>
                                    </ListItem>
                                ))}
                            </Box>
                        ))}
                    </List>
                )}

                {!modelsLoading && filteredModels.length === 0 && (
                    <Box sx={{ p: 3, textAlign: 'center' }}>
                        <Typography variant="body2" sx={{ color: '#64748b' }}>
                            {searchQuery ? '没有找到匹配的模型' : '暂无模型'}
                        </Typography>
                    </Box>
                )}
            </Box>

            {/* 底部统计 */}
            <Divider sx={{ borderColor: '#2d3748' }} />
            <Box sx={{ px: 1.5, py: 0.75, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.7rem' }}>
                    共 {models.length} 个模型
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Box sx={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e' }} />
                        <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.6rem' }}>
                            预训练 {models.filter(m => m.isPretrained).length}
                        </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Box sx={{ width: 6, height: 6, borderRadius: '50%', background: '#f59e0b' }} />
                        <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.6rem' }}>
                            自定义 {models.filter(m => !m.isPretrained).length}
                        </Typography>
                    </Box>
                </Box>
            </Box>

            {/* 操作菜单 */}
            <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
                PaperProps={{
                    sx: {
                        background: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: 1.5,
                        minWidth: 120,
                        '& .MuiMenuItem-root': {
                            color: '#e2e8f0',
                            fontSize: '0.8rem',
                            py: 0.75,
                            '&:hover': {
                                background: 'rgba(255,255,255,0.05)'
                            }
                        }
                    }
                }}
            >
                <MenuItem onClick={handleDownload}>
                    <Download sx={{ mr: 1.5, fontSize: 16, color: '#64748b' }} />
                    下载
                </MenuItem>
                <MenuItem onClick={handleDelete} sx={{ color: '#ef4444' }}>
                    <Delete sx={{ mr: 1.5, fontSize: 16 }} />
                    删除
                </MenuItem>
            </Menu>
        </Box>
    );
};

export default ModelsPanel;