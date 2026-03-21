/**
 * 工作流面板组件
 * 显示已保存的工作流
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
    Divider,
    Menu,
    MenuItem
} from '@mui/material';
import {
    Search,
    AccountTree,
    MoreVert,
    PlayArrow,
    ContentCopy,
    Delete,
    Download,
    Share,
    AccessTime
} from '@mui/icons-material';
import type { Workflow } from '../../types';

// 模拟工作流数据
const mockWorkflows: (Workflow & { status: 'active' | 'draft' | 'archived'; nodeCount: number })[] = [
    {
        id: 'wf_1',
        name: '光栅耦合器逆向设计',
        description: '使用 HiLab 进行光栅耦合器逆向设计',
        nodes: [],
        edges: [],
        createdAt: '2024-01-15T10:30:00Z',
        updatedAt: '2024-01-15T14:20:00Z',
        status: 'active',
        nodeCount: 7
    },
    {
        id: 'wf_2',
        name: '波分复用器优化',
        description: '4通道波分复用器拓扑优化',
        nodes: [],
        edges: [],
        createdAt: '2024-01-14T09:00:00Z',
        updatedAt: '2024-01-14T16:45:00Z',
        status: 'active',
        nodeCount: 9
    },
    {
        id: 'wf_3',
        name: '超构光栅设计',
        description: '超构光栅逆向设计实验',
        nodes: [],
        edges: [],
        createdAt: '2024-01-13T11:20:00Z',
        updatedAt: '2024-01-13T11:20:00Z',
        status: 'draft',
        nodeCount: 4
    },
    {
        id: 'wf_4',
        name: 'MDN 训练工作流',
        description: '混合密度网络训练流程',
        nodes: [],
        edges: [],
        createdAt: '2024-01-12T14:00:00Z',
        updatedAt: '2024-01-12T18:30:00Z',
        status: 'archived',
        nodeCount: 5
    }
];

const WorkflowsPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

    // 过滤工作流
    const filteredWorkflows = mockWorkflows.filter(workflow =>
        searchQuery === '' ||
        workflow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        workflow.description.toLowerCase().includes(searchQuery.toLowerCase())
    );

    // 格式化日期
    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

        if (diffDays === 0) return '今天';
        if (diffDays === 1) return '昨天';
        if (diffDays < 7) return `${diffDays} 天前`;
        return date.toLocaleDateString('zh-CN');
    };

    // 状态颜色映射
    const statusColors: Record<string, string> = {
        'active': '#22c55e',
        'draft': '#f59e0b',
        'archived': '#64748b'
    };

    // 状态文本映射
    const statusText: Record<string, string> = {
        'active': '活跃',
        'draft': '草稿',
        'archived': '已归档'
    };

    const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, workflowId: string) => {
        event.stopPropagation();
        setAnchorEl(event.currentTarget);
        setSelectedWorkflow(workflowId);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
        setSelectedWorkflow(null);
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 搜索框 */}
            <Box sx={{ p: 2 }}>
                <TextField
                    size="small"
                    placeholder="搜索工作流..."
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

            {/* 工作流列表 */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
                <List dense disablePadding>
                    {filteredWorkflows.map((workflow) => (
                        <ListItem
                            key={workflow.id}
                            onClick={() => setSelectedWorkflow(workflow.id)}
                            sx={{
                                px: 2,
                                py: 1.5,
                                borderBottom: '1px solid rgba(45, 55, 72, 0.5)',
                                cursor: 'pointer',
                                background: selectedWorkflow === workflow.id
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
                                        width: 36,
                                        height: 36,
                                        borderRadius: 1.5,
                                        background: `linear-gradient(135deg, ${statusColors[workflow.status]}20 0%, ${statusColors[workflow.status]}10 100%)`,
                                        border: `1px solid ${statusColors[workflow.status]}30`,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        color: statusColors[workflow.status]
                                    }}
                                >
                                    <AccountTree fontSize="small" />
                                </Box>
                            </ListItemIcon>
                            <ListItemText
                                primary={
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography sx={{ color: '#e2e8f0', fontWeight: 500 }}>
                                            {workflow.name}
                                        </Typography>
                                        <Chip
                                            label={statusText[workflow.status]}
                                            size="small"
                                            sx={{
                                                height: 18,
                                                fontSize: '0.65rem',
                                                background: `${statusColors[workflow.status]}20`,
                                                color: statusColors[workflow.status]
                                            }}
                                        />
                                    </Box>
                                }
                                secondary={
                                    <Box sx={{ mt: 0.5 }}>
                                        <Typography variant="caption" sx={{ color: '#64748b', display: 'block' }}>
                                            {workflow.description}
                                        </Typography>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 0.5 }}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                <AccessTime sx={{ fontSize: 12, color: '#64748b' }} />
                                                <Typography variant="caption" sx={{ color: '#64748b' }}>
                                                    {formatDate(workflow.updatedAt)}
                                                </Typography>
                                            </Box>
                                            <Typography variant="caption" sx={{ color: '#64748b' }}>
                                                {workflow.nodeCount} 个节点
                                            </Typography>
                                        </Box>
                                    </Box>
                                }
                                secondaryTypographyProps={{ component: 'div' }}
                            />
                            <ListItemSecondaryAction>
                                <IconButton
                                    size="small"
                                    onClick={(e) => handleMenuOpen(e, workflow.id)}
                                    sx={{ color: '#64748b' }}
                                >
                                    <MoreVert fontSize="small" />
                                </IconButton>
                            </ListItemSecondaryAction>
                        </ListItem>
                    ))}
                </List>

                {filteredWorkflows.length === 0 && (
                    <Box sx={{ p: 4, textAlign: 'center' }}>
                        <Typography sx={{ color: '#64748b' }}>
                            没有找到匹配的工作流
                        </Typography>
                    </Box>
                )}
            </Box>

            {/* 底部统计 */}
            <Divider sx={{ borderColor: '#2d3748' }} />
            <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="caption" sx={{ color: '#64748b' }}>
                    共 {mockWorkflows.length} 个工作流
                </Typography>
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                    {Object.entries(statusText).map(([status, text]) => (
                        <Chip
                            key={status}
                            label={`${text}: ${mockWorkflows.filter(w => w.status === status).length}`}
                            size="small"
                            sx={{
                                height: 18,
                                fontSize: '0.6rem',
                                background: `${statusColors[status]}15`,
                                color: statusColors[status]
                            }}
                        />
                    ))}
                </Box>
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
                    <PlayArrow sx={{ mr: 1, fontSize: 18 }} />
                    打开
                </MenuItem>
                <MenuItem onClick={handleMenuClose}>
                    <ContentCopy sx={{ mr: 1, fontSize: 18 }} />
                    复制
                </MenuItem>
                <MenuItem onClick={handleMenuClose}>
                    <Download sx={{ mr: 1, fontSize: 18 }} />
                    导出
                </MenuItem>
                <MenuItem onClick={handleMenuClose}>
                    <Share sx={{ mr: 1, fontSize: 18 }} />
                    分享
                </MenuItem>
                <MenuItem onClick={handleMenuClose} sx={{ color: '#ef4444' }}>
                    <Delete sx={{ mr: 1, fontSize: 18 }} />
                    删除
                </MenuItem>
            </Menu>
        </Box>
    );
};

export default WorkflowsPanel;
