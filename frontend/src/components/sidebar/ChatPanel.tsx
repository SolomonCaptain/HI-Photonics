/**
 * AI 助手对话面板
 * 提供与 LLM 助手的交互界面
 */

import React, { useState, useRef, useEffect } from 'react';
import {
    Box,
    Typography,
    TextField,
    IconButton,
    Paper,
    Chip,
    Fade,
    CircularProgress,
    Tooltip,
} from '@mui/material';
import {
    Send,
    Clear,
    Lightbulb,
    Psychology,
} from '@mui/icons-material';
import { useLLMStore } from '../../store/llmStore';
import type { ChatMessage } from '../../utils/llmApi';

// 快捷提示词
const QUICK_PROMPTS = [
    { label: '设计光栅耦合器', prompt: '帮我设计一个1550nm波长的光栅耦合器，效率要大于70%' },
    { label: '推荐模型', prompt: '对于光栅耦合器逆向设计，应该选择哪种模型？TNN还是MDN？' },
    { label: '解释结果', prompt: '如何解读逆向设计的输出结果？' },
    { label: '优化参数', prompt: '训练模型时，如何调整超参数以获得更好的性能？' },
];

// 渲染消息内容（支持简单的 Markdown）
const MessageContent: React.FC<{ content: string }> = ({ content }) => {
    // 简单的 Markdown 渲染
    const renderContent = (text: string) => {
        const lines = text.split('\n');
        return lines.map((line, idx) => {
            // 粗体
            line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            // 列表项
            if (line.startsWith('• ') || line.startsWith('- ')) {
                return (
                    <Box key={idx} sx={{ display: 'flex', gap: 1, mb: 0.5 }}>
                        <Box component="span" sx={{ color: '#3b82f6' }}>•</Box>
                        <Box 
                            component="span" 
                            dangerouslySetInnerHTML={{ __html: line.substring(2) }}
                            sx={{ flex: 1 }}
                        />
                    </Box>
                );
            }
            // 标题
            if (line.startsWith('# ')) {
                return (
                    <Typography key={idx} variant="h6" sx={{ mt: 1, mb: 0.5, color: '#e2e8f0' }}>
                        {line.substring(2)}
                    </Typography>
                );
            }
            // 普通文本
            return (
                <Box 
                    key={idx} 
                    component="span" 
                    sx={{ display: 'block', mb: 0.5 }}
                    dangerouslySetInnerHTML={{ __html: line || '&nbsp;' }}
                />
            );
        });
    };

    return (
        <Box sx={{ 
            fontSize: '0.875rem', 
            lineHeight: 1.6,
            '& strong': { color: '#60a5fa' }
        }}>
            {renderContent(content)}
        </Box>
    );
};

// 单条消息组件
const ChatMessageItem: React.FC<{ message: ChatMessage }> = ({ message }) => {
    const isUser = message.role === 'user';

    return (
        <Fade in timeout={300}>
            <Box
                sx={{
                    display: 'flex',
                    justifyContent: isUser ? 'flex-end' : 'flex-start',
                    mb: 2,
                }}
            >
                <Paper
                    elevation={0}
                    sx={{
                        maxWidth: '85%',
                        p: 1.5,
                        borderRadius: 2,
                        background: isUser
                            ? 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)'
                            : 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
                        border: isUser
                            ? 'none'
                            : '1px solid rgba(59, 130, 246, 0.2)',
                        color: '#e2e8f0',
                    }}
                >
                    {!isUser && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <Psychology sx={{ fontSize: 16, color: '#3b82f6' }} />
                            <Typography variant="caption" sx={{ color: '#94a3b8' }}>
                                AI 助手
                            </Typography>
                        </Box>
                    )}
                    <MessageContent content={message.content} />
                </Paper>
            </Box>
        </Fade>
    );
};

const ChatPanel: React.FC = () => {
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    const { messages, isLoading, sendMessage, clearMessages } = useLLMStore();

    // 自动滚动到底部
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // 发送消息
    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const message = input.trim();
        setInput('');
        await sendMessage(message);
    };

    // 快捷提示词点击
    const handleQuickPrompt = (prompt: string) => {
        setInput(prompt);
        inputRef.current?.focus();
    };

    // 键盘事件处理
    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <Box
            sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                background: 'transparent',
            }}
        >
            {/* 头部工具栏 */}
            <Box
                sx={{
                    p: 1.5,
                    borderBottom: '1px solid #2d3748',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    background: 'rgba(0, 0, 0, 0.2)',
                }}
            >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Lightbulb sx={{ fontSize: 18, color: '#f59e0b' }} />
                    <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                        快捷提问
                    </Typography>
                </Box>
                <Tooltip title="清空对话">
                    <IconButton
                        size="small"
                        onClick={clearMessages}
                        sx={{ color: '#64748b', '&:hover': { color: '#f87171' } }}
                    >
                        <Clear fontSize="small" />
                    </IconButton>
                </Tooltip>
            </Box>

            {/* 快捷提示词 */}
            <Box
                sx={{
                    p: 1,
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 0.5,
                    borderBottom: '1px solid #2d3748',
                    background: 'rgba(0, 0, 0, 0.1)',
                }}
            >
                {QUICK_PROMPTS.map((item) => (
                    <Chip
                        key={item.label}
                        label={item.label}
                        size="small"
                        onClick={() => handleQuickPrompt(item.prompt)}
                        sx={{
                            background: 'rgba(59, 130, 246, 0.1)',
                            border: '1px solid rgba(59, 130, 246, 0.3)',
                            color: '#60a5fa',
                            fontSize: '0.75rem',
                            '&:hover': {
                                background: 'rgba(59, 130, 246, 0.2)',
                                border: '1px solid rgba(59, 130, 246, 0.5)',
                            },
                        }}
                    />
                ))}
            </Box>

            {/* 消息列表 */}
            <Box
                sx={{
                    flex: 1,
                    overflow: 'auto',
                    p: 2,
                    '&::-webkit-scrollbar': {
                        width: '4px',
                    },
                    '&::-webkit-scrollbar-track': {
                        background: 'transparent',
                    },
                    '&::-webkit-scrollbar-thumb': {
                        background: '#2d3748',
                        borderRadius: '4px',
                    },
                }}
            >
                {messages.map((message, index) => (
                    <ChatMessageItem
                        key={index}
                        message={message}
                    />
                ))}

                {/* 加载状态 */}
                {isLoading && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 2 }}>
                        <CircularProgress size={16} sx={{ color: '#3b82f6' }} />
                        <Typography variant="body2" sx={{ color: '#64748b' }}>
                            AI 正在思考...
                        </Typography>
                    </Box>
                )}

                <div ref={messagesEndRef} />
            </Box>

            {/* 输入区域 */}
            <Box
                sx={{
                    p: 1.5,
                    borderTop: '1px solid #2d3748',
                    background: 'rgba(0, 0, 0, 0.2)',
                }}
            >
                <Box
                    sx={{
                        display: 'flex',
                        gap: 1,
                        alignItems: 'flex-end',
                    }}
                >
                    <TextField
                        inputRef={inputRef}
                        fullWidth
                        multiline
                        maxRows={3}
                        placeholder="描述你的设计需求..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={isLoading}
                        sx={{
                            '& .MuiOutlinedInput-root': {
                                background: '#1e293b',
                                borderRadius: 2,
                                border: '1px solid #2d3748',
                                '&:hover': {
                                    border: '1px solid #3b82f6',
                                },
                                '&.Mui-focused': {
                                    border: '1px solid #3b82f6',
                                    boxShadow: '0 0 0 2px rgba(59, 130, 246, 0.1)',
                                },
                            },
                            '& .MuiInputBase-input': {
                                color: '#e2e8f0',
                                fontSize: '0.875rem',
                                '&::placeholder': {
                                    color: '#64748b',
                                },
                            },
                        }}
                    />
                    <IconButton
                        onClick={handleSend}
                        disabled={!input.trim() || isLoading}
                        sx={{
                            background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
                            color: '#fff',
                            borderRadius: 2,
                            p: 1.2,
                            '&:hover': {
                                background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
                            },
                            '&.Mui-disabled': {
                                background: '#2d3748',
                                color: '#64748b',
                            },
                        }}
                    >
                        {isLoading ? (
                            <CircularProgress size={20} sx={{ color: '#fff' }} />
                        ) : (
                            <Send />
                        )}
                    </IconButton>
                </Box>

                {/* 提示 */}
                <Typography
                    variant="caption"
                    sx={{
                        display: 'block',
                        mt: 1,
                        color: '#64748b',
                        textAlign: 'center',
                    }}
                >
                    按 Enter 发送 · Shift + Enter 换行
                </Typography>
            </Box>
        </Box>
    );
};

export default ChatPanel;
