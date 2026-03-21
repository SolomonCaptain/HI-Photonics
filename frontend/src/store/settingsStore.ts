/**
 * 设置状态管理
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsState {
    // API 设置
    apiUrl: string;
    setApiUrl: (url: string) => void;

    // 主题设置
    theme: 'dark' | 'light';
    setTheme: (theme: 'dark' | 'light') => void;

    // 编辑器设置
    snapToGrid: boolean;
    gridSize: number;
    setSnapToGrid: (snap: boolean) => void;
    setGridSize: (size: number) => void;

    // 执行设置
    autoExecute: boolean;
    executionTimeout: number;
    setAutoExecute: (auto: boolean) => void;
    setExecutionTimeout: (timeout: number) => void;
}

export const useSettingsStore = create<SettingsState>()(
    persist(
        (set) => ({
            // API 设置
            apiUrl: 'http://localhost:8080',
            setApiUrl: (url) => set({ apiUrl: url }),

            // 主题设置
            theme: 'dark',
            setTheme: (theme) => set({ theme }),

            // 编辑器设置
            snapToGrid: true,
            gridSize: 15,
            setSnapToGrid: (snap) => set({ snapToGrid: snap }),
            setGridSize: (size) => set({ gridSize: size }),

            // 执行配置
            autoExecute: false,
            executionTimeout: 300,
            setAutoExecute: (auto) => set({ autoExecute: auto }),
            setExecutionTimeout: (timeout) => set({ executionTimeout: timeout })
        }),
        {
            name: 'hi-photonics-settings'
        }
    )
);