/**
 * 顶部工具栏组件
 */

import React from "react";
import {
    Box,
    Typography,
    Button,
    ButtonGroup,
    Divider,
    Tooltip,
    IconButton,
    Chip
} from "@mui/material";
import {
    PlayArrow,
    Stop,
    Save,
    FolderOpen,
    Undo,
    Redo,
    Settings,
    Help,
    AutoAwesome,
    Github
} from "@mui/icons-material";
import { useWorkflowStore } from "../store/workflowStore";

const Header: React.FC = () => {
    const { executeAll, isExecuting, clearWorkflow, nodes } = useWorkflowStore();

    return (
        <Box
            sx={{
                height: 56,
                background: 'linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
                borderBottom: '1px solid #2d3748',
                display: 'flex',
                alignItems: 'center',
                px: 2,
                gap: 2
            }}
        >
            {/* Logo */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AutoAwesome sx={{ color: '#3b82f6', fontSize: 28 }} />
                <Typography
                    variant="h6"
                    sx={{
                        fontWeight: 700,
                        background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent'
                    }}
                >
                    HI-Photonics Studio
                </Typography>
            </Box>

            <Divider orientation="vertical" flexItem sx={{ borderColor: '#2d3748' }} />

            {/* 工作流操作 */}
            <ButtonGroup variant="outlined" size="small">
                <Tooltip title="新建工作流">
                    <Button
                        startIcon={<FolderOpen />}
                        sx={{ borderColor: '#2d3748', color: '#94a3b8' }}
                        onClick={clearWorkflow}
                    >
                        新建
                    </Button>
                </Tooltip>
                <Tooltip title="保存工作流">
                    <Button
                        startIcon={<Save />}
                        sx={{ borderColor: '#2d3748', color: '#94a3b8' }}
                    >
                        保存
                    </Button>
                </Tooltip>
            </ButtonGroup>

            {/* 编辑操作 */}
            <ButtonGroup variant="outlined" size="small">
                <Tooltip title="撤销">
                    <IconButton sx={{ borderColor: '#2d3748', color: '#64748b' }}>
                        <Undo fontSize="small" />
                    </IconButton>
                </Tooltip>
                <Tooltip title="重做">
                    <IconButton sx={{ borderColor: '#2d3748', color: '#64748b' }}>
                        <Redo fontSize="small" />
                    </IconButton>
                </Tooltip>
            </ButtonGroup>

            <Box sx={{ flex: 1 }} />

            {/* 节点状态 */}
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <Chip
                    label={`${nodes.length} 节点`}
                    size="small"
                    sx={{
                        background: 'rgba(59, 130, 246, 0.2)',
                        color: '#93c5fd',
                        border: '1px solid rgba(59, 130, 246, 0.3)'
                    }}
                />
            </Box>

            {/* 执行控制 */}
            <Button
                variant="contained"
                size="small"
                startIcon={isExecuting ? <Stop /> : <PlayArrow />}
                onClick={executeAll}
                disabled={nodes.length === 0}
                sx={{
                    backgroud: isExecuting
                        ? 'linear-gradient(135deg, #ef4444, #dc2626)'
                        : 'linear-gradient(135deg, #22c55e, #16a34a)',
                    '&:hover': {
                        background: isExecuting
                            ? 'linear-gradient(135deg, #dc2626, #b91c1c)'
                            : 'linear-gradient(135deg, #16a34a, #15803d)'
                    }
                }}
            >
                {isExecuting ? '停止' : '执行工作流'}
            </Button>

            <Divider orientation="vertical" flexItem sx={{ borderColor: '#2d3748' }} />

            {/* 帮助链接 */}
            <Tooltip title="GitHub">
                <IconButton
                    component="a"
                    href="https://github.com/SolomonCaptain/HI-Photonics"
                    target="_blank"
                    sx={{ color: '#64748b' }}
                >
                    <Github />
                </IconButton>
            </Tooltip>
            <Tooltip title="帮助文档">
                <IconButton sx={{ color: '#64748b' }}>
                    <Help />
                </IconButton>
            </Tooltip>
            <Tooltip title="设置">
                <IconButton sx={{ color: '#64748b' }}>
                    <Settings />
                </IconButton>
            </Tooltip>
        </Box>
    );
};

export default Header;