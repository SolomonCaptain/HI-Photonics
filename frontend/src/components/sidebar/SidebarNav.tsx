/**
 * 侧边栏导航组件
 * 类似 ComfyUI 的左侧图标导航栏
 */

import React from 'react';
import { Box, IconButton, Tooltip } from '@mui/material';
import {
    FolderOpen,
    Extension,
    Psychology,
    AccountTree,
    Dashboard,
    SmartToy,
} from '@mui/icons-material';
import { SidebarPanelType } from '../../types';
import type { SidebarPanelType as SidebarPanelTypeType } from '../../types';
import { useWorkflowStore } from '../../store/workflowStore';

interface NavItem {
    type: SidebarPanelTypeType;
    name: string;
    icon: React.ReactElement;
    description: string;
}

const navItems: NavItem[] = [
    {
        type: SidebarPanelType.ASSETS,
        name: '资产',
        icon: <FolderOpen />,
        description: '光谱图、GDS版图等资源'
    },
    {
        type: SidebarPanelType.NODES,
        name: '节点',
        icon: <Extension />,
        description: '工作流节点库'
    },
    {
        type: SidebarPanelType.MODELS,
        name: '模型',
        icon: <Psychology />,
        description: '逆向设计模型'
    },
    {
        type: SidebarPanelType.WORKFLOWS,
        name: '工作流',
        icon: <AccountTree />,
        description: '已保存的工作流'
    },
    {
        type: SidebarPanelType.TEMPLATES,
        name: '模板',
        icon: <Dashboard />,
        description: '预设工作流模板'
    },
    {
        type: SidebarPanelType.AI_ASSISTANT,
        name: 'AI 助手',
        icon: <SmartToy />,
        description: 'LLM 智能设计助手'
    }
];

const SidebarNav: React.FC = () => {
    const { currentPanel, togglePanel, sidebarOpen } = useWorkflowStore();

    return (
        <Box
            sx={{
                width: 48,
                background: 'linear-gradient(180deg, #0d0d1a 0%, #12122a 100%)',
                borderRight: '1px solid #2d3748',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                py: 2,
                gap: 1
            }}
        >
            {navItems.map((item) => {
                const isActive = currentPanel === item.type && sidebarOpen;
                return (
                    <Tooltip
                        key={item.type}
                        title={`${item.name}: ${item.description}`}
                        placement="right"
                        arrow
                        sx={{
                            '& .MuiTooltip-tooltip': {
                                background: '#1a1a2e',
                                border: '1px solid #2d3748'
                            }
                        }}
                    >
                        <IconButton
                            onClick={() => togglePanel(item.type)}
                            sx={{
                                width: 36,
                                height: 36,
                                borderRadius: 1.5,
                                color: isActive ? '#3b82f6' : '#64748b',
                                background: isActive
                                    ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%)'
                                    : 'transparent',
                                border: isActive ? '1px solid rgba(59, 130, 246, 0.4)' : '1px solid transparent',
                                transition: 'all 0.2s ease',
                                '&:hover': {
                                    color: '#3b82f6',
                                    background: 'rgba(59, 130, 246, 0.1)',
                                    border: '1px solid rgba(59, 130, 246, 0.3)'
                                },
                                '& .MuiSvgIcon-root': {
                                    fontSize: 20
                                }
                            }}
                        >
                            {item.icon}
                        </IconButton>
                    </Tooltip>
                );
            })}
        </Box>
    );
};

export default SidebarNav;
