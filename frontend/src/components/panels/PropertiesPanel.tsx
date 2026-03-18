/**
 * 属性面板
 * 显示选中节点的参数配置
 */

import React from 'react';
import {
  Box,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  FormControlLabel,
  Switch,
  Slider,
  Divider,
  IconButton,
  Chip,
  Button,
  Alert,
  LinearProgress
} from '@mui/material';
import {
  Delete,
  PlayArrow,
  Refresh,
  Close,
  CheckCircle,
  Error,
  HourglassEmpty
} from '@mui/icons-material';
import { useWorkflowStore, NODE_DEFINITIONS } from '../../store/workflowStore';
import { NodeParameter } from '../../types';

const PropertiesPanel: React.FC = () => {
  const { 
    nodes, 
    selectedNodeId, 
    updateNodeParams, 
    removeNode,
    executeNode,
    propertiesOpen,
    setPropertiesOpen
  } = useWorkflowStore();

  const selectedNode = nodes.find(n => n.id === selectedNodeId);

  if (!propertiesOpen) return null;

  const definition = selectedNode ? NODE_DEFINITIONS[selectedNode.type] : null;

  const renderParamInput = (param: NodeParameter, value: any) => {
    switch (param.type) {
      case 'number':
        if (param.options) {
          return (
            <Select
              size="small"
              value={value ?? param.default}
              onChange={(e) => updateNodeParams(selectedNodeId!, { [param.key]: e.target.value })}
              sx={{
                background: 'rgba(255,255,255,0.05)',
                '& .MuiOutlinedInput-notchedOutline': { borderColor: '#2d3748' },
                '& .MuiSvgIcon-root': { color: '#64748b' },
                '& .MuiSelect-select': { color: '#e2e8f0' }
              }}
            >
              {param.options.map(opt => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>
          );
        }
        return (
          <Box sx={{ px: 1 }}>
            <Slider
              value={value ?? param.default}
              min={param.min ?? 0}
              max={param.max ?? 1}
              step={param.step ?? 0.01}
              onChange={(_, v) => updateNodeParams(selectedNodeId!, { [param.key]: v })}
              valueLabelDisplay="auto"
              sx={{
                '& .MuiSlider-track': { background: '#3b82f6' },
                '& .MuiSlider-thumb': { background: '#3b82f6' },
                '& .MuiSlider-rail': { background: '#2d3748' }
              }}
            />
            <TextField
              size="small"
              type="number"
              value={value ?? param.default}
              onChange={(e) => updateNodeParams(selectedNodeId!, { [param.key]: parseFloat(e.target.value) })}
              sx={{
                mt: 1,
                '& .MuiOutlinedInput-root': {
                  background: 'rgba(255,255,255,0.05)',
                  '& fieldset': { borderColor: '#2d3748' },
                  '& input': { color: '#e2e8f0', textAlign: 'center' }
                }
              }}
              inputProps={{ step: param.step, min: param.min, max: param.max }}
            />
          </Box>
        );

      case 'string':
        return (
          <TextField
            size="small"
            value={value ?? param.default}
            onChange={(e) => updateNodeParams(selectedNodeId!, { [param.key]: e.target.value })}
            fullWidth
            sx={{
              '& .MuiOutlinedInput-root': {
                background: 'rgba(255,255,255,0.05)',
                '& fieldset': { borderColor: '#2d3748' },
                '& input': { color: '#e2e8f0' }
              }
            }}
          />
        );

      case 'select':
        return (
          <FormControl size="small" fullWidth>
            <Select
              value={value ?? param.default}
              onChange={(e) => updateNodeParams(selectedNodeId!, { [param.key]: e.target.value })}
              sx={{
                background: 'rgba(255,255,255,0.05)',
                '& .MuiOutlinedInput-notchedOutline': { borderColor: '#2d3748' },
                '& .MuiSvgIcon-root': { color: '#64748b' },
                '& .MuiSelect-select': { color: '#e2e8f0' }
              }}
            >
              {param.options?.map(opt => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>
          </FormControl>
        );

      case 'boolean':
        return (
          <FormControlLabel
            control={
              <Switch
                checked={value ?? param.default}
                onChange={(e) => updateNodeParams(selectedNodeId!, { [param.key]: e.target.checked })}
                sx={{
                  '& .MuiSwitch-switchBase.Mui-checked': { color: '#22c55e' },
                  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { background: '#22c55e' }
                }}
              />
            }
            label=""
          />
        );

      case 'array':
        return (
          <TextField
            size="small"
            value={Array.isArray(value) ? JSON.stringify(value) : JSON.stringify(param.default)}
            onChange={(e) => {
              try {
                const arr = JSON.parse(e.target.value);
                updateNodeParams(selectedNodeId!, { [param.key]: arr });
              } catch {}
            }}
            fullWidth
            sx={{
              '& .MuiOutlinedInput-root': {
                background: 'rgba(255,255,255,0.05)',
                '& fieldset': { borderColor: '#2d3748' },
                '& input': { color: '#e2e8f0', fontFamily: 'monospace' }
              }
            }}
          />
        );

      default:
        return null;
    }
  };

  return (
    <Box
      sx={{
        width: 300,
        background: 'linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)',
        borderLeft: '1px solid #2d3748',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden'
      }}
    >
      {/* 标题栏 */}
      <Box
        sx={{
          p: 2,
          borderBottom: '1px solid #2d3748',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}
      >
        <Typography variant="h6" sx={{ color: '#e2e8f0', fontWeight: 600 }}>
          属性
        </Typography>
        <IconButton onClick={() => setPropertiesOpen(false)} sx={{ color: '#64748b' }}>
          <Close />
        </IconButton>
      </Box>

      {!selectedNode ? (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography sx={{ color: '#64748b' }}>
            选择一个节点以查看属性
          </Typography>
        </Box>
      ) : (
        <>
          {/* 节点信息 */}
          <Box sx={{ p: 2, borderBottom: '1px solid #2d3748' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle1" sx={{ color: '#e2e8f0', fontWeight: 600 }}>
                {definition?.name}
              </Typography>
              <Chip
                label={selectedNode.data.status}
                size="small"
                icon={
                  selectedNode.data.status === 'success' ? <CheckCircle /> :
                  selectedNode.data.status === 'error' ? <Error /> :
                  selectedNode.data.status === 'running' ? <HourglassEmpty /> : undefined
                }
                sx={{
                  background: 
                    selectedNode.data.status === 'success' ? 'rgba(34, 197, 94, 0.2)' :
                    selectedNode.data.status === 'error' ? 'rgba(239, 68, 68, 0.2)' :
                    selectedNode.data.status === 'running' ? 'rgba(59, 130, 246, 0.2)' :
                    'rgba(100, 116, 139, 0.2)',
                  color:
                    selectedNode.data.status === 'success' ? '#86efac' :
                    selectedNode.data.status === 'error' ? '#fca5a5' :
                    selectedNode.data.status === 'running' ? '#93c5fd' :
                    '#94a3b8'
                }}
              />
            </Box>
            <Typography variant="caption" sx={{ color: '#64748b' }}>
              {definition?.description}
            </Typography>
            
            {selectedNode.data.status === 'running' && (
              <LinearProgress 
                sx={{ 
                  mt: 2, 
                  borderRadius: 1,
                  '& .MuiLinearProgress-bar': { background: '#3b82f6' },
                  background: '#2d3748'
                }} 
              />
            )}
          </Box>

          {/* 参数配置 */}
          <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
            <Typography variant="subtitle2" sx={{ color: '#94a3b8', mb: 2, textTransform: 'uppercase' }}>
              参数
            </Typography>
            
            {definition?.params.map(param => (
              <Box key={param.key} sx={{ mb: 2 }}>
                <Typography
                  variant="body2"
                  sx={{ color: '#e2e8f0', mb: 0.5, display: 'flex', alignItems: 'center' }}
                >
                  {param.label}
                  {param.description && (
                    <Typography component="span" variant="caption" sx={{ color: '#64748b', ml: 1 }}>
                      ({param.description})
                    </Typography>
                  )}
                </Typography>
                {renderParamInput(param, selectedNode.data.params[param.key])}
              </Box>
            ))}
          </Box>

          {/* 操作按钮 */}
          <Divider sx={{ borderColor: '#2d3748' }} />
          <Box sx={{ p: 2, display: 'flex', gap: 1 }}>
            <Button
              variant="contained"
              startIcon={<PlayArrow />}
              onClick={() => executeNode(selectedNodeId!)}
              disabled={selectedNode.data.status === 'running'}
              sx={{
                flex: 1,
                background: 'linear-gradient(135deg, #3b82f6, #2563eb)',
                '&:hover': { background: 'linear-gradient(135deg, #2563eb, #1d4ed8)' }
              }}
            >
              执行
            </Button>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              sx={{
                borderColor: '#2d3748',
                color: '#94a3b8',
                '&:hover': { borderColor: '#4a5568', background: 'rgba(255,255,255,0.05)' }
              }}
            >
              重置
            </Button>
            <IconButton
              onClick={() => removeNode(selectedNodeId!)}
              sx={{ color: '#ef4444', '&:hover': { background: 'rgba(239, 68, 68, 0.1)' } }}
            >
              <Delete />
            </IconButton>
          </Box>
        </>
      )}
    </Box>
  );
};

export default PropertiesPanel;
