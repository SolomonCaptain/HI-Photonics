/**
 * 主侧边栏组件
 * 整合导航栏和面板，类似 ComfyUI 的布局
 */

import React from 'react';
import { Box, Typography, IconButton } from '@mui/material';
import { ChevronLeft } from '@mui/icons-material';
import { SidebarPanelType } from '../../types';
import { useWorkflowStore } from '../../store/workflowStore';
import {
    SidebarNav,
    AssetsPanel,
    NodesPanel,
    ModelsPanel,
    WorkflowsPanel,
    TemplatesPanel
} from '../sidebar';

// 面板标题映射
const panelTitles: Record<string, string> = {
    [SidebarPanelType.ASSETS]: '资产',
    [SidebarPanelType.NODES]: '节点库',
    [SidebarPanelType.MODELS]: '模型',
    [SidebarPanelType.WORKFLOWS]: '工作流',
    [SidebarPanelType.TEMPLATES]: '模板'
};

// 面板颜色映射
const panelColors: Record<string, string> = {
    [SidebarPanelType.ASSETS]: '#f59e0b',
    [SidebarPanelType.NODES]: '#3b82f6',
    [SidebarPanelType.MODELS]: '#8b5cf6',
    [SidebarPanelType.WORKFLOWS]: '#22c55e',
    [SidebarPanelType.TEMPLATES]: '#06b6d4'
};

const MainSidebar: React.FC = () => {
    const { sidebarOpen, currentPanel, setSidebarOpen } = useWorkflowStore();

    // 渲染当前面板
    const renderPanel = () => {
        switch (currentPanel) {
            case SidebarPanelType.ASSETS:
                return <AssetsPanel />;
            case SidebarPanelType.NODES:
                return <NodesPanel />;
            case SidebarPanelType.MODELS:
                return <ModelsPanel />;
            case SidebarPanelType.WORKFLOWS:
                return <WorkflowsPanel />;
            case SidebarPanelType.TEMPLATES:
                return <TemplatesPanel />;
            default:
                return <NodesPanel />;
        }
    };

    return (
        <Box
            sx={{
                display: 'flex',
                height: '100%',
                background: 'linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)'
            }}
        >
            {/* 左侧图标导航栏 */}
            <SidebarNav />

            {/* 右侧面板区域 */}
            {sidebarOpen && (
                <Box
                    sx={{
                        width: 280,
                        borderRight: '1px solid #2d3748',
                        display: 'flex',
                        flexDirection: 'column',
                        overflow: 'hidden',
                        animation: 'slideIn 0.2s ease-out',
                        '@keyframes slideIn': {
                            '0%': { opacity: 0, transform: 'translateX(-10px)' },
                            '100%': { opacity: 1, transform: 'translateX(0)' }
                        }
                    }}
                >
                    {/* 面板标题栏 */}
                    <Box
                        sx={{
                            p: 2,
                            borderBottom: '1px solid #2d3748',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            background: 'rgba(0,0,0,0.2)'
                        }}
                    >
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Box
                                sx={{
                                    width: 8,
                                    height: 8,
                                    borderRadius: '50%',
                                    background: panelColors[currentPanel] || '#64748b',
                                    boxShadow: `0 0 8px ${panelColors[currentPanel] || '#64748b'}50`
                                }}
                            />
                            <Typography
                                variant="h6"
                                sx={{ color: '#e2e8f0', fontWeight: 600, fontSize: '1rem' }}
                            >
                                {panelTitles[currentPanel] || '面板'}
                            </Typography>
                        </Box>
                        <IconButton
                            onClick={() => setSidebarOpen(false)}
                            sx={{
                                color: '#64748b',
                                '&:hover': {
                                    color: '#94a3b8',
                                    background: 'rgba(255,255,255,0.05)'
                                }
                            }}
                        >
                            <ChevronLeft fontSize="small" />
                        </IconButton>
                    </Box>

                    {/* 面板内容 */}
                    <Box sx={{ flex: 1, overflow: 'hidden' }}>
                        {renderPanel()}
                    </Box>
                </Box>
            )}
        </Box>
    );
};

export default MainSidebar;
