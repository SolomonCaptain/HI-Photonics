/**
 * 资产面板组件
 * 显示光谱图、GDS版图等资源
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
    Menu,
    MenuItem,
    Divider
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
    Add
} from '@mui/icons-material';
import { AssetType, type Asset } from '../../types/nodes';

// 资产类型图标映射
const assetTypeIcons: Record<AssetType, React.ReactElement> = {
    [AssetType.SPECTRUM]: <ShowChart />,
    [AssetType.GDS]: <GridOn />,
    [AssetType.STRUCTURE]: <ViewInAr />,
    [AssetType.FIELD]: <GraphicEq />,
    [AssetType.DATASET]: <Storage />,
    [AssetType.MODEL_WEIGHTS]: <ModelTraining />
};

// 资产类型名称映射
const assetTypeNames: Record<AssetType, string> = {
    [AssetType.SPECTRUM]: '光谱图',
    [AssetType.GDS]: 'GDS版图',
    [AssetType.STRUCTURE]: '结构设计',
    [AssetType.FIELD]: '场分布',
    [AssetType.DATASET]: '数据集',
    [AssetType.MODEL_WEIGHTS]: '模型权重'
};

// 资产类型颜色映射
const assetTypeColors: Record<AssetType, string> = {
    [AssetType.SPECTRUM]: '#22c55e',
    [AssetType.GDS]: '#3b82f6',
    [AssetType.STRUCTURE]: '#f59e0b',
    [AssetType.FIELD]: '#ef4444',
    [AssetType.DATASET]: '#8b5cf6',
    [AssetType.MODEL_WEIGHTS]: '#06b6d4'
};

// 模拟资产数据
const mockAssets: Asset[] = [
    {
        id: 'asset_1',
        name: '光栅耦合器光谱',
        type: AssetType.SPECTRUM,
        description: '1.55μm 波长光栅耦合器透射谱',
        createdAt: '2024-01-15T10:30:00Z',
        updatedAt: '2024-01-15T10:30:00Z',
        size: 128 * 1024
    },
    {
        id: 'asset_2',
        name: '波分复用器 GDS',
        type: AssetType.GDS,
        description: '4通道波分复用器版图',
        createdAt: '2024-01-14T14:20:00Z',
        updatedAt: '2024-01-14T14:20:00Z',
        size: 2.5 * 1024 * 1024
    },
    {
        id: 'asset_3',
        name: '超构光栅结构',
        type: AssetType.STRUCTURE,
        description: '超构光栅逆向设计结构',
        createdAt: '2024-01-13T09:15:00Z',
        updatedAt: '2024-01-13T09:15:00Z',
        size: 512 * 1024
    },
    {
        id: 'asset_4',
        name: '训练数据集 v2',
        type: AssetType.DATASET,
        description: '10000 样本光栅耦合器训练数据',
        createdAt: '2024-01-12T16:45:00Z',
        updatedAt: '2024-01-12T16:45:00Z',
        size: 150 * 1024 * 1024
    },
    {
        id: 'asset_5',
        name: 'HiLab 模型权重',
        type: AssetType.MODEL_WEIGHTS,
        description: '训练完成的 HiLab 逆向设计模型',
        createdAt: '2024-01-11T11:00:00Z',
        updatedAt: '2024-01-11T11:00:00Z',
        size: 25 * 1024 * 1024
    }
];

const AssetsPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedType, setSelectedType] = useState<AssetType | null>(null);
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const [selectedAsset, setSelectedAsset] = useState<string | null>(null);

    // 过滤资产
    const filteredAssets = mockAssets.filter(asset => {
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
        setAnchorEl(event.currentTarget);
        setSelectedAsset(assetId);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
        setSelectedAsset(null);
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 搜索框 */}
            <Box sx={{ p: 2 }}>
                <TextField
                    size="small"
                    placeholder="搜索资产..."
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

            {/* 类型筛选 */}
            <Box sx={{ px: 2, pb: 2, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                <Chip
                    label="全部"
                    size="small"
                    onClick={() => setSelectedType(null)}
                    sx={{
                        background: selectedType === null ? 'rgba(59, 130, 246, 0.3)' : 'rgba(255,255,255,0.05)',
                        color: selectedType === null ? '#93c5fd' : '#94a3b8',
                        border: selectedType === null ? '1px solid rgba(59, 130, 246, 0.5)' : '1px solid transparent',
                        '&:hover': { background: 'rgba(59, 130, 246, 0.2)' }
                    }}
                />
                {Object.values(AssetType).map(type => (
                    <Chip
                        key={type}
                        label={assetTypeNames[type]}
                        size="small"
                        onClick={() => setSelectedType(type)}
                        sx={{
                            background: selectedType === type ? `${assetTypeColors[type]}30` : 'rgba(255,255,255,0.05)',
                            color: selectedType === type ? assetTypeColors[type] : '#94a3b8',
                            border: selectedType === type ? `1px solid ${assetTypeColors[type]}50` : '1px solid transparent',
                            '&:hover': { background: `${assetTypeColors[type]}20` }
                        }}
                    />
                ))}
            </Box>

            <Divider sx={{ borderColor: '#2d3748' }} />

            {/* 资产列表 */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
                <List dense disablePadding>
                    {filteredAssets.map((asset) => (
                        <ListItem
                            key={asset.id}
                            sx={{
                                px: 2,
                                py: 1,
                                borderBottom: '1px solid rgba(45, 55, 72, 0.5)',
                                '&:hover': {
                                    background: 'rgba(255,255,255,0.03)'
                                }
                            }}
                        >
                            <ListItemIcon sx={{ minWidth: 36, color: assetTypeColors[asset.type] }}>
                                {assetTypeIcons[asset.type]}
                            </ListItemIcon>
                            <ListItemText
                                primary={asset.name}
                                secondary={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                                        <Chip
                                            label={assetTypeNames[asset.type]}
                                            size="small"
                                            sx={{
                                                height: 18,
                                                fontSize: '0.65rem',
                                                background: `${assetTypeColors[asset.type]}20`,
                                                color: assetTypeColors[asset.type]
                                            }}
                                        />
                                        <Typography variant="caption" sx={{ color: '#64748b' }}>
                                            {formatSize(asset.size)}
                                        </Typography>
                                    </Box>
                                }
                                primaryTypographyProps={{
                                    sx: { color: '#e2e8f0', fontSize: '0.875rem' }
                                }}
                                secondaryTypographyProps={{
                                    component: 'div'
                                }}
                            />
                            <ListItemSecondaryAction>
                                <IconButton
                                    size="small"
                                    onClick={(e) => handleMenuOpen(e, asset.id)}
                                    sx={{ color: '#64748b' }}
                                >
                                    <MoreVert fontSize="small" />
                                </IconButton>
                            </ListItemSecondaryAction>
                        </ListItem>
                    ))}
                </List>

                {filteredAssets.length === 0 && (
                    <Box sx={{ p: 4, textAlign: 'center' }}>
                        <Typography sx={{ color: '#64748b' }}>
                            没有找到匹配的资产
                        </Typography>
                    </Box>
                )}
            </Box>

            {/* 操作菜单 */}
            <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
                PaperProps={{
                    sx: {
                        background: '#1a1a2e',
                        border: '1px solid #2d3748',
                        '& .MuiMenuItem-root': {
                            color: '#e2e8f0',
                            '&:hover': {
                                background: 'rgba(255,255,255,0.05)'
                            }
                        }
                    }
                }}
            >
                <MenuItem onClick={handleMenuClose}>
                    <Visibility sx={{ mr: 1, fontSize: 18 }} />
                    预览
                </MenuItem>
                <MenuItem onClick={handleMenuClose}>
                    <Download sx={{ mr: 1, fontSize: 18 }} />
                    下载
                </MenuItem>
                <MenuItem onClick={handleMenuClose} sx={{ color: '#ef4444' }}>
                    <Delete sx={{ mr: 1, fontSize: 18 }} />
                    删除
                </MenuItem>
            </Menu>
        </Box>
    );
};

export default AssetsPanel;
