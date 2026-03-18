/**
 * HI-Photonics Studio 主应用
 * 光子学逆向设计可视化工作流平台
 */

import React from 'react';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { Toaster } from 'react-hot-toast';
import { Header, WorkflowCanvas } from './components';
import { NodeSidebar, PropertiesPanel } from './components/panels';

// 创建暗色主题
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#3b82f6',
      light: '#60a5fa',
      dark: '#2563eb'
    },
    secondary: {
      main: '#8b5cf6',
      light: '#a78bfa',
      dark: '#7c3aed'
    },
    background: {
      default: '#0f0f1a',
      paper: '#1a1a2e'
    },
    text: {
      primary: '#e2e8f0',
      secondary: '#94a3b8'
    }
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h6: {
      fontWeight: 600
    }
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8
        }
      }
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none'
        }
      }
    }
  }
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: '#1a1a2e',
            color: '#e2e8f0',
            border: '1px solid #2d3748'
          }
        }}
      />
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          overflow: 'hidden'
        }}
      >
        <Header />
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          <NodeSidebar />
          <WorkflowCanvas />
          <PropertiesPanel />
        </div>
      </div>
    </ThemeProvider>
  );
};

export default App;
