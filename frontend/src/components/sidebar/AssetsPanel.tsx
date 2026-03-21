/**
 * 资产面板组件
 * 显示光谱图、GDS版图等资源
 * 
 * 资源分类：
 * - inputs/: datasets, spectra, gds, structures
 * - outputs/: designs, simulations, exports
 * - op_models/: pretrained, custom
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
    Menu,
    MenuItem,
    Divider,
    CircularProgress,
    Tabs,
    Tab
} from '@mui/material';
import {
    Search,
    ShowChart,
    GridOn,
    ViewInAr,
    GraphicEq,
    Storage,
    ModelTraining,
    MoreVert,
    Delete,
    Download,
    Visibility,
    Input as InputIcon,
    Output as OutputIcon,
    Memory as MemoryIcon,
    Refresh
} from '@mui/icons-material';
import { useResourceStore, type ResourceCategory, type AssetType } from '../../store/resourceStore';
import toast from 'react-hot-toast';

// 资产类型图标映射
const assetTypeIcons: Record<AssetType, React.ReactElement> = {
    dataset: <Storage />,
    spectrum: <ShowChart />,
    gds: <GridOn />,
    structure: <ViewInAr />,
    design: <ViewInAr />,
    simulation: <GraphicEq />,
    export: <GridOn />,
    field: <GraphicEq />,
    model_weights: <ModelTraining />
};

// 资产类型名称映射
const assetTypeNames: Record<AssetType, string> = {
    dataset: '数据集',
    spectrum: '光谱图',
    gds: 'GDS版图',
    structure: '结构设计',
    design: '设计结果',
    simulation: '仿真结果',
    export: '导出文件',
    field: '场分布',
    model_weights: '模型权重'
};

// 资产类型颜色映射
const assetTypeColors: Record<AssetType, string> = {
    dataset: '#8b5cf6',
    spectrum: '#22c55e',
    gds: '#3b82f6',
    structure: '#f59e0b',
    design: '#06b6d4',
    simulation: '#ef4444',
    export: '#3b82f6',
    field: '#ef4444',
    model_weights: '#06b6d4'
};

// 分类标签
const categoryTabs: { value: ResourceCategory; label: string; icon: React.ReactElement }[] = [
    { value: 'inputs', label: '输入', icon: <InputIcon fontSize="small" /> },
    { value: 'outputs', label: '输出', icon: <OutputIcon fontSize="small" /> },
    { value: 'models', label: '模型', icon: <MemoryIcon fontSize="small" /> },
];

const AssetsPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedType, setSelectedType] = useState<AssetType | null>(null);
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);
    const [activeCategory, setActiveCategory] = useState<ResourceCategory>('inputs');

    const {
        assets,
        assetsLoading,
        assetsError,
        fetchAssets,
        deleteAsset,
        downloadAsset,
        setSelectedAsset,
    } = useResourceStore();

    // 初始化加载
    useEffect(() => {
        fetchAssets(activeCategory);
    }, [activeCategory, fetchAssets]);

    // 按当前分类过滤资产
    const categoryAssets = assets.filter(a => a.category === activeCategory);

    // 过滤资产
    const filteredAssets = categoryAssets.filter(asset => {
        const matchesSearch = searchQuery === '' ||
            asset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            asset.description?.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesType = selectedType === null || asset.type === selectedType;
        return matchesSearch && matchesType;
    });

    // 格式化文件大小
    const formatSize = (bytes?: number) => {
        if (!bytes) return '';
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, assetId: string) => {
        event.stopPropagation();
        setAnchorEl(event.currentTarget);
        setSelectedAssetId(assetId);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
        setSelectedAssetId(null);
    };

    const handleDelete = async () => {
        if (!selectedAssetId) return;
        try {
            await deleteAsset(selectedAssetId, activeCategory);
            toast.success('资产已删除');
        } catch (error) {
            toast.error('删除失败');
        }
        handleMenuClose();
    };

    const handleDownload = async () => {
        if (!selectedAssetId) return;
        try {
            await downloadAsset(selectedAssetId, activeCategory);
            toast.success('下载已开始');
        } catch (error) {
            toast.error('下载失败');
        }
        handleMenuClose();
    };

    const handleRefresh = () => {
        fetchAssets(activeCategory);
    };

    // 获取当前分类可用的资产类型
    const getAvailableTypes = (): AssetType[] => {
        const types = new Set<AssetType>();
        categoryAssets.forEach(a => types.add(a.type));
        return Array.from(types);
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 分类标签 */}
            <Tabs
                value={activeCategory}
                onChange={(_, value) => setActiveCategory(value)}
                variant="fullWidth"
                sx={{
                    minHeight: 36,
                    borderBottom: '1px solid #2d3748',
                    '& .MuiTabs-indicator': {
                        backgroundColor: '#3b82f6',
                    },
                    '& .MuiTab-root': {
                        minHeight: 36,
                        py: 0.5,
                        color: '#64748b',
                        '&.Mui-selected': {
                            color: '#93c5fd',
                        },
                    },
                }}
            >
                {categoryTabs.map(tab => (
                    <Tab
                        key={tab.value}
                        value={tab.value}
                        icon={tab.icon}
                        label={tab.label}
                        iconPosition="start"
                        sx={{ flexDirection: 'row', gap: 0.5 }}
                    />
                ))}
            </Tabs>

            {/* 搜索框 */}
            <Box sx={{ p: 1.5, display: 'flex', gap: 1 }}>
                <TextField
                    size="small"
                    placeholder="搜索资产..."
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
                    disabled={assetsLoading}
                    sx={{ color: '#64748b' }}
                >
                    {assetsLoading ? <CircularProgress size={18} /> : <Refresh fontSize="small" />}
                </IconButton>
            </Box>

            {/* 类型筛选 */}
            <Box sx={{ px: 1.5, pb: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                <Chip
                    label="全部"
                    size="small"
                    onClick={() => setSelectedType(null)}
                    sx={{
                        height: 22,
                        fontSize: '0.7rem',
                        background: selectedType === null ? 'rgba(59, 130, 246, 0.3)' : 'rgba(255,255,255,0.05)',
                        color: selectedType === null ? '#93c5fd' : '#94a3b8',
                        border: selectedType === null ? '1px solid rgba(59, 130, 246, 0.5)' : '1px solid transparent',
                        '&:hover': { background: 'rgba(59, 130, 246, 0.2)' }
                    }}
                />
                {getAvailableTypes().map(type => (
                    <Chip
                        key={type}
                        label={assetTypeNames[type]}
                        size="small"
                        onClick={() => setSelectedType(type)}
                        sx={{
                            height: 22,
                            fontSize: '0.7rem',
                            background: selectedType === type ? `${assetTypeColors[type]}30` : 'rgba(255,255,255,0.05)',
                            color: selectedType === type ? assetTypeColors[type] : '#94a3b8',
                            border: selectedType === type ? `1px solid ${assetTypeColors[type]}50` : '1px solid transparent',
                            '&:hover': { background: `${assetTypeColors[type]}20` }
                        }}
                    />
                ))}
            </Box>

            <Divider sx={{ borderColor: '#2d3748' }} />

            {/* 错误提示 */}
            {assetsError && (
                <Box sx={{ p: 2, background: 'rgba(239, 68, 68, 0.1)' }}>
                    <Typography variant="caption" sx={{ color: '#ef4444' }}>
                        {assetsError}
                    </Typography>
                </Box>
            )}

            {/* 资产列表 */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
                {assetsLoading && filteredAssets.length === 0 ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                        <CircularProgress size={24} />
                    </Box>
                ) : (
                    <List dense disablePadding>
                        {filteredAssets.map((asset) => (
                            <ListItem
                                key={asset.id}
                                sx={{
                                    px: 1.5,
                                    py: 0.75,
                                    borderBottom: '1px solid rgba(45, 55, 72, 0.5)',
                                    cursor: 'pointer',
                                    '&:hover': {
                                        background: 'rgba(255,255,255,0.03)'
                                    }
                                }}
                                onClick={() => setSelectedAsset(asset)}
                            >
                                <ListItemIcon sx={{ minWidth: 32, color: assetTypeColors[asset.type] }}>
                                    {assetTypeIcons[asset.type]}
                                </ListItemIcon>
                                <ListItemText
                                    primary={asset.name}
                                    secondary={
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.25 }}>
                                            <Chip
                                                label={assetTypeNames[asset.type]}
                                                size="small"
                                                sx={{
                                                    height: 16,
                                                    fontSize: '0.6rem',
                                                    background: `${assetTypeColors[asset.type]}20`,
                                                    color: assetTypeColors[asset.type]
                                                }}
                                            />
                                            <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.65rem' }}>
                                                {formatSize(asset.fileSize)}
                                            </Typography>
                                        </Box>
                                    }
                                    primaryTypographyProps={{
                                        sx: { color: '#e2e8f0', fontSize: '0.8rem' }
                                    }}
                                    secondaryTypographyProps={{
                                        component: 'div'
                                    }}
                                />
                                <ListItemSecondaryAction>
                                    <IconButton
                                        size="small"
                                        onClick={(e) => handleMenuOpen(e, asset.id)}
                                        sx={{ color: '#64748b', p: 0.5 }}
                                    >
                                        <MoreVert fontSize="small" />
                                    </IconButton>
                                </ListItemSecondaryAction>
                            </ListItem>
                        ))}
                    </List>
                )}

                {!assetsLoading && filteredAssets.length === 0 && (
                    <Box sx={{ p: 3, textAlign: 'center' }}>
                        <Typography variant="body2" sx={{ color: '#64748b' }}>
                            {searchQuery || selectedType ? '没有找到匹配的资产' : '暂无资产'}
                        </Typography>
                    </Box>
                )}
            </Box>

            {/* 底部统计 */}
            <Divider sx={{ borderColor: '#2d3748' }} />
            <Box sx={{ px: 1.5, py: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.7rem' }}>
                    共 {categoryAssets.length} 项
                </Typography>
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
                <MenuItem onClick={handleMenuClose}>
                    <Visibility sx={{ mr: 1.5, fontSize: 16, color: '#64748b' }} />
                    预览
                </MenuItem>
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

export default AssetsPanel;