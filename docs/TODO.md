# HI-Photonics 开发待办清单

> 最后更新: 2026-03-18

## 项目状态概览

| 模块 | 实现进度 | 测试覆盖 | 示例 | 文档 |
|------|----------|----------|------|------|
| TNN | 95% | ✅ 完整 | ✅ 有 | ❌ 缺失 |
| MDN | 95% | ✅ 完整 | ✅ 有 | ❌ 缺失 |
| CGAN | 90% | ✅ 完整 | ✅ 有 | ❌ 缺失 |
| PINN | 85% | ✅ 完整 | ✅ 有 | ❌ 缺失 |
| GNN | 80% | ✅ 完整 | ❌ 无 | ❌ 缺失 |
| HiLab | 0% | ❌ 无 | ⚠️ 有 | ❌ 缺失 |

---

## 🔴 高优先级 (P0)

### 1. HiLab 模型实现
- [ ] 实现 `models/inverse/hilab.py`
  - [ ] 定义 HiLabConfig 配置类
  - [ ] 实现 HiLab 核心模型架构
  - [ ] 实现与 HiLab API 的接口
  - [ ] 添加 `create_hilab_for_challenge` 便捷函数
- [ ] 添加测试 `tests/test_models/test_hilab.py`
- [ ] 验证 `examples/03_hilab_workflow.ipynb` 可正常运行

**备注**: 存在示例 notebook 但模型为空文件，需要优先处理

### 2. GNN 模型完善
- [ ] 添加 GNN 示例脚本 `examples/09_gnn_inverse_design.py`
- [ ] 完善 GNN 模型实现 (当前 80%)
  - [ ] 添加图池化层 (Global Pooling)
  - [ ] 实现边特征更新机制
  - [ ] 添加预训练模型支持
- [ ] 添加 GNN 模型文档字符串

### 3. 核心文档编写
- [ ] 编写 `docs/architecture.md`
  - [ ] 系统架构图
  - [ ] 核心概念说明
  - [ ] 数据流图
- [ ] 编写 `docs/api_reference.md`
  - [ ] models 模块 API
  - [ ] core 模块 API
  - [ ] interfaces 模块 API

---

## 🟠 中优先级 (P1)

### 4. 仿真器接口完善
- [ ] 检查并完善 `interfaces/simulators/rcwa.py`
- [ ] 检查并完善 `interfaces/simulators/lumerical.py`
- [ ] 添加仿真器性能基准测试
- [ ] 实现仿真结果缓存机制

### 5. 代码清理
- [ ] 处理 `core/nodes/base.py` 空文件
  - 确认功能是否已在 `core/node.py` 实现
  - 决定是删除还是合并
- [ ] 统一代码风格检查 (添加 flake8/black 配置)
- [ ] 添加类型检查配置 (pyright/mypy)

### 6. 训练流程优化
- [ ] 实现 `models/training/trainer.py` 统一训练器
  - [ ] 支持分布式训练
  - [ ] 支持混合精度训练
  - [ ] 支持梯度累积
- [ ] 添加学习率查找器
- [ ] 实现模型自动超参数优化

### 7. 数据管道增强
- [ ] 实现 `data/generators/active_learning.py` 主动学习数据生成
- [ ] 实现 `data/generators/multi_fidelity.py` 多保真度数据生成
- [ ] 完善 `data/loaders/hdf5_loader.py` HDF5 数据加载
- [ ] 添加数据验证和清洗工具

---

## 🟡 低优先级 (P2)

### 8. 新增设计挑战
- [ ] 实现 `challenges/waveguide_bend.py` 波导弯曲设计
- [ ] 实现 `challenges/splitter.py` 分束器设计
- [ ] 实现 `challenges/resonator.py` 谐振器设计
- [ ] 添加挑战性能基准

### 9. 代理模型扩展
- [ ] 完善 `models/surrogates/cnn_surrogate.py`
- [ ] 完善 `models/surrogates/deeponet.py`
- [ ] 完善 `models/surrogates/pino.py`
- [ ] 添加代理模型对比评估脚本

### 10. 可视化增强
- [ ] 实现 `interfaces/visualization/field.py` 场分布可视化
- [ ] 实现 `interfaces/visualization/structure.py` 结构可视化
- [ ] 添加交互式设计预览工具
- [ ] 支持 GDS 文件可视化

### 11. 代工厂接口
- [ ] 完善 `interfaces/foundry/design_rules.py` 设计规则检查
- [ ] 完善 `interfaces/foundry/gds.py` GDS 导出
- [ ] 添加 DRC (设计规则检查) 验证
- [ ] 支持多代工厂工艺文件

### 12. 优化算法扩展
- [ ] 完善 `optimization/solvers/bayesian.py` 贝叶斯优化
- [ ] 完善 `optimization/solvers/evolutionary.py` 进化算法
- [ ] 完善 `optimization/solvers/gradient_based.py` 梯度优化
- [ ] 添加多目标优化支持

---

## 🔵 持续改进 (P3)

### 13. 测试覆盖率
- [ ] 提高单元测试覆盖率至 80%+
- [ ] 添加集成测试
- [ ] 添加性能回归测试
- [ ] 设置 Codecov 集成

### 14. CI/CD 增强
- [ ] 添加代码风格检查到 CI
- [ ] 添加类型检查到 CI
- [ ] 添加自动发布流程
- [ ] 添加文档自动部署

### 15. 性能优化
- [ ] GPU 内存优化
- [ ] 数据加载性能优化
- [ ] 模型推理加速
- [ ] 大规模仿真并行化

### 16. 文档完善
- [ ] 添加贡献指南 `CONTRIBUTING.md`
- [ ] 添加变更日志 `CHANGELOG.md`
- [ ] 添加常见问题 `docs/faq.md`
- [ ] 添加教程文档

---

## 📋 版本规划

### v0.2.0 (下一个版本)
- [x] TNN 模型实现
- [x] MDN 模型实现
- [x] CGAN 模型实现
- [x] PINN 模型实现
- [x] GNN 模型实现
- [ ] HiLab 模型实现
- [ ] 核心文档
- [ ] GNN 示例

### v0.3.0
- [ ] 统一训练器
- [ ] 数据管道完善
- [ ] 仿真器缓存
- [ ] 新增设计挑战

### v0.4.0
- [ ] 代工厂接口
- [ ] 可视化工具
- [ ] 代理模型
- [ ] 多目标优化

### v1.0.0
- [ ] 完整文档
- [ ] 80%+ 测试覆盖率
- [ ] 性能优化
- [ ] 稳定 API

---

## 🛠️ 技术债务

| 项目 | 描述 | 影响 | 建议处理时间 |
|------|------|------|--------------|
| hilab.py 空文件 | 有示例无实现 | 高 | 立即 |
| docs/ 空文档 | 缺乏使用指南 | 高 | 本周 |
| 无类型检查配置 | 代码质量风险 | 中 | 本月 |
| 无统一训练器 | 代码重复 | 中 | 下周 |
| GNN 无示例 | 功能不可用 | 中 | 本周 |

---

## 📝 备注

- 使用 GitHub Issues 跟踪具体任务
- 大型任务应拆分为多个 PR
- 每个 PR 应包含对应测试
- 遵循语义化版本控制
