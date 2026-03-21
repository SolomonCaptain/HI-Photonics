/**
 * 工作流面板组件
 * 显示已保存的工作流
 * 
 * 工作流存储在 workflows/saved/ 目录下
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
    Menu,
    MenuItem,
    CircularProgress,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    TextField as MuiTextField
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
    AccessTime,
    Refresh,
    Save
} from '@mui/icons-material';
import { useResourceStore, type SavedWorkflow } from '../../store/resourceStore';
import { useWorkflowStore } from '../../store/workflowStore';
import toast from 'react-hot-toast';

const WorkflowsPanel: React.FC = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedWorkflowId, setSelectedWorkflowId] = useState<string | null>(null);
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const [saveDialogOpen, setSaveDialogOpen] = useState(false);
    const [workflowName, setWorkflowName] = useState('');
    const [workflowDescription, setWorkflowDescription] = useState('');

    const {
        workflows,
        workflowsLoading,
        workflowsError,
        fetchWorkflows,
        saveWorkflow,
        deleteWorkflow,
    } = useResourceStore();

    const {
        nodes,
        edges,
        loadWorkflow,
    } = useWorkflowStore();

    // 初始化加载
    useEffect(() => {
        fetchWorkflows();
    }, [fetchWorkflows]);

    // 过滤工作流
    const filteredWorkflows = workflows.filter(workflow =>
        searchQuery === '' ||
        workflow.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        workflow.description?.toLowerCase().includes(searchQuery.toLowerCase())
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

    const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, workflowId: string) => {
        event.stopPropagation();
        setAnchorEl(event.currentTarget);
        setSelectedWorkflowId(workflowId);
    };

    const handleMenuClose = () => {
        setAnchorEl(null);
        setSelectedWorkflowId(null);
    };

    const handleOpenWorkflow = (workflow: SavedWorkflow) => {
        loadWorkflow(workflow.nodes, workflow.edges);
        toast.success(`已加载工作流: ${workflow.name}`);
        handleMenuClose();
    };

    const handleDeleteWorkflow = async () => {
        if (!selectedWorkflowId) return;
        try {
            await deleteWorkflow(selectedWorkflowId);
            toast.success('工作流已删除');
        } catch (error) {
            toast.error('删除失败');
        }
        handleMenuClose();
    };

    const handleSaveCurrentWorkflow = async () => {
        if (!workflowName.trim()) {
            toast.error('请输入工作流名称');
            return;
        }
        try {
            await saveWorkflow({
                name: workflowName,
                nodes,
                edges,
                description: workflowDescription || undefined,
            });
            toast.success('工作流已保存');
            setSaveDialogOpen(false);
            setWorkflowName('');
            setWorkflowDescription('');
        } catch (error) {
            toast.error('保存失败');
        }
    };

    const handleRefresh = () => {
        fetchWorkflows();
    };

    const selectedWorkflow = workflows.find(w => w.id === selectedWorkflowId);

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* 搜索框和保存按钮 */}
            <Box sx={{ p: 1.5, display: 'flex', gap: 1 }}>
                <TextField
                    size="small"
                    placeholder="搜索工作流..."
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
                    onClick={() => setSaveDialogOpen(true)}
                    disabled={nodes.length === 0}
                    sx={{
                        color: '#22c55e',
                        background: 'rgba(34, 197, 94, 0.1)',
                        '&:hover': { background: 'rgba(34, 197, 94, 0.2)' }
                    }}
                    title="保存当前工作流"
                >
                    <Save fontSize="small" />
                </IconButton>
                <IconButton
                    size="small"
                    onClick={handleRefresh}
                    disabled={workflowsLoading}
                    sx={{ color: '#64748b' }}
                >
                    {workflowsLoading ? <CircularProgress size={18} /> : <Refresh fontSize="small" />}
                </IconButton>
            </Box>

            <Divider sx={{ borderColor: '#2d3748' }} />

            {/* 错误提示 */}
            {workflowsError && (
                <Box sx={{ p: 2, background: 'rgba(239, 68, 68, 0.1)' }}>
                    <Typography variant="caption" sx={{ color: '#ef4444' }}>
                        {workflowsError}
                    </Typography>
                </Box>
            )}

            {/* 工作流列表 */}
            <Box sx={{ flex: 1, overflow: 'auto' }}>
                {workflowsLoading && filteredWorkflows.length === 0 ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                        <CircularProgress size={24} />
                    </Box>
                ) : (
                    <List dense disablePadding>
                        {filteredWorkflows.map((workflow) => (
                            <ListItem
                                key={workflow.id}
                                onClick={() => setSelectedWorkflowId(workflow.id)}
                                sx={{
                                    px: 1.5,
                                    py: 0.75,
                                    borderBottom: '1px solid rgba(45, 55, 72, 0.5)',
                                    cursor: 'pointer',
                                    background: selectedWorkflowId === workflow.id
                                        ? 'rgba(59, 130, 246, 0.1)'
                                        : 'transparent',
                                    '&:hover': {
                                        background: 'rgba(255,255,255,0.03)'
                                    }
                                }}
                            >
                                <ListItemIcon sx={{ minWidth: 36 }}>
                                    <Box
                                        sx={{
                                            width: 28,
                                            height: 28,
                                            borderRadius: 1,
                                            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.1) 100%)',
                                            border: '1px solid rgba(59, 130, 246, 0.3)',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            color: '#3b82f6'
                                        }}
                                    >
                                        <AccountTree fontSize="small" />
                                    </Box>
                                </ListItemIcon>
                                <ListItemText
                                    primary={
                                        <Typography sx={{ color: '#e2e8f0', fontSize: '0.8rem', fontWeight: 500 }}>
                                            {workflow.name}
                                        </Typography>
                                    }
                                    secondary={
                                        <Box sx={{ mt: 0.25 }}>
                                            {workflow.description && (
                                                <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.65rem', display: 'block' }}>
                                                    {workflow.description}
                                                </Typography>
                                            )}
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 0.25 }}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                    <AccessTime sx={{ fontSize: 10, color: '#64748b' }} />
                                                    <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.6rem' }}>
                                                        {formatDate(workflow.updatedAt)}
                                                    </Typography>
                                                </Box>
                                                <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.6rem' }}>
                                                    {workflow.nodes?.length || 0} 个节点
                                                </Typography>
                                            </Box>
                                            {workflow.tags && workflow.tags.length > 0 && (
                                                <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
                                                    {workflow.tags.map(tag => (
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
                                    }
                                    secondaryTypographyProps={{ component: 'div' }}
                                />
                                <ListItemSecondaryAction>
                                    <IconButton
                                        size="small"
                                        onClick={(e) => handleMenuOpen(e, workflow.id)}
                                        sx={{ color: '#64748b', p: 0.5 }}
                                    >
                                        <MoreVert fontSize="small" />
                                    </IconButton>
                                </ListItemSecondaryAction>
                            </ListItem>
                        ))}
                    </List>
                )}

                {!workflowsLoading && filteredWorkflows.length === 0 && (
                    <Box sx={{ p: 3, textAlign: 'center' }}>
                        <Typography variant="body2" sx={{ color: '#64748b' }}>
                            {searchQuery ? '没有找到匹配的工作流' : '暂无已保存的工作流'}
                        </Typography>
                    </Box>
                )}
            </Box>

            {/* 底部统计 */}
            <Divider sx={{ borderColor: '#2d3748' }} />
            <Box sx={{ px: 1.5, py: 0.75, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="caption" sx={{ color: '#64748b', fontSize: '0.7rem' }}>
                    共 {workflows.length} 个工作流
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
                        minWidth: 140,
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
                <MenuItem onClick={() => selectedWorkflow && handleOpenWorkflow(selectedWorkflow)}>
                    <PlayArrow sx={{ mr: 1.5, fontSize: 16, color: '#22c55e' }} />
                    打开
                </MenuItem>
                <MenuItem onClick={handleMenuClose}>
                    <ContentCopy sx={{ mr: 1.5, fontSize: 16, color: '#64748b' }} />
                    复制
                </MenuItem>
                <MenuItem onClick={handleMenuClose}>
                    <Download sx={{ mr: 1.5, fontSize: 16, color: '#64748b' }} />
                    导出
                </MenuItem>
                <MenuItem onClick={handleMenuClose}>
                    <Share sx={{ mr: 1.5, fontSize: 16, color: '#64748b' }} />
                    分享
                </MenuItem>
                <MenuItem onClick={handleDeleteWorkflow} sx={{ color: '#ef4444' }}>
                    <Delete sx={{ mr: 1.5, fontSize: 16 }} />
                    删除
                </MenuItem>
            </Menu>

            {/* 保存对话框 */}
            <Dialog
                open={saveDialogOpen}
                onClose={() => setSaveDialogOpen(false)}
                PaperProps={{
                    sx: {
                        background: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: 2,
                        minWidth: 320
                    }
                }}
            >
                <DialogTitle sx={{ color: '#e2e8f0', pb: 1 }}>
                    保存工作流
                </DialogTitle>
                <DialogContent>
                    <MuiTextField
                        autoFocus
                        margin="dense"
                        label="工作流名称"
                        fullWidth
                        value={workflowName}
                        onChange={(e) => setWorkflowName(e.target.value)}
                        sx={{
                            '& .MuiInputLabel-root': { color: '#64748b' },
                            '& .MuiOutlinedInput-root': {
                                '& fieldset': { borderColor: '#334155' },
                                '& input': { color: '#e2e8f0' }
                            }
                        }}
                    />
                    <MuiTextField
                        margin="dense"
                        label="描述（可选）"
                        fullWidth
                        multiline
                        rows={2}
                        value={workflowDescription}
                        onChange={(e) => setWorkflowDescription(e.target.value)}
                        sx={{
                            '& .MuiInputLabel-root': { color: '#64748b' },
                            '& .MuiOutlinedInput-root': {
                                '& fieldset': { borderColor: '#334155' },
                                '& textarea': { color: '#e2e8f0' }
                            }
                        }}
                    />
                    <Typography variant="caption" sx={{ color: '#64748b', mt: 1, display: 'block' }}>
                        当前工作流包含 {nodes.length} 个节点和 {edges.length} 条连接
                    </Typography>
                </DialogContent>
                <DialogActions sx={{ px: 3, pb: 2 }}>
                    <Button
                        onClick={() => setSaveDialogOpen(false)}
                        sx={{ color: '#94a3b8' }}
                    >
                        取消
                    </Button>
                    <Button
                        onClick={handleSaveCurrentWorkflow}
                        variant="contained"
                        sx={{
                            background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
                            '&:hover': {
                                background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)'
                            }
                        }}
                    >
                        保存
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default WorkflowsPanel;