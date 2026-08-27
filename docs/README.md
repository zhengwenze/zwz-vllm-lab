# Nano-vLLM 文档索引

本目录集中保存环境复现、在线调度器设计、实验协议和面试材料。文档中的状态遵循统一证据边界：

- **Implemented**：代码已经实现，但不等于已经完成目标硬件验证；
- **CPU Verified**：已通过不依赖 CUDA 的契约测试；
- **GPU Verified**：已有固定环境、工作负载和原始结果支持；
- **GPU Pending**：尚未获得目标 RTX 4060 的真实在线实验数据。

## 环境与离线基线

| 文档 | 用途 |
| --- | --- |
| [environment.md](environment.md) | WSL2、Python、PyTorch、CUDA、Triton 与 FlashAttention 环境 |
| [milestone-00-baseline.md](milestone-00-baseline.md) | 离线 GPU 基线目标、步骤与验收标准 |
| [baseline-rtx4060.md](baseline-rtx4060.md) | 已验证的 RTX 4060 离线基线、方法和限制 |

## 在线调度器

| 文档 | 用途 |
| --- | --- |
| [开发文档](online_scheduler/DEV_DOCUMENT.md) | 架构、状态机、不变量、配置和测试矩阵 |
| [在线 API 契约](online_scheduler/NANOVLLM_ONLINE_API.md) | Python/SSE 字段、时序、状态码和错误语义 |
| [实现规范](online_scheduler/NANOVLLM_SCHEDULER_STYLE.md) | 线程所有权、编码约束、异常与清理规范 |
| [开发日志](online_scheduler/DEVELOPMENT_LOG.md) | 七天实施过程、节点证据和当前门禁 |
| [RTX 4060 实验报告模板](online_scheduler/EXPERIMENT_REPORT_RTX4060.md) | 三策略 A/B 实验设计与待填结果 |
| [WSL2 + RTX 4060 复现手册](online_scheduler/WSL2_RTX4060_RUNBOOK.md) | GPU 冒烟、正式实验和排障流程 |
| [原始数据规范](online_scheduler/results/README.md) | manifest、请求、token、step 和 summary schema |

## 项目表达与复盘

| 文档 | 用途 |
| --- | --- |
| [项目展示与面试讲解](online_scheduler/PROJECT_SHOWCASE_ZH.md) | 贡献边界、核心链路、难点和演示脚本 |
| [模拟面试 Q&A](interview/NANOVLLM_MOCK_INTERVIEW_QA.md) | 推理框架、KV Cache、调度、算子和多卡高频题 |
| [项目问题清单与下一步路线](PROJECT_GAPS_AND_NEXT_STEPS_ZH.md) | 当前证据缺口、简历收敛与后续优先级 |

真实性原则：离线吞吐不能替代在线 TTFT/TPOT 结论；CPU 测试数量不能替代 GPU 性能证据；上游 Nano-vLLM 能力与本仓库新增实现必须分别表述。
