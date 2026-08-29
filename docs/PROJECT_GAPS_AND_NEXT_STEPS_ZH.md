# Nano-vLLM 项目问题清单与下一步路线

> 状态：在线调度实现与 CPU 契约测试已完成；RTX 4060 在线三策略 A/B 已于 2026-08-28 升级为 **GPU Verified**（[15-run 报告](../reports/nanovllm-online-rtx4060-20260828.md)）。
>
> 用途：本文主体是 GPU 实验前的历史问题清单，用于复盘哪些缺口已关闭；不作为当前性能结果报告。

当前 **Nano-vLLM 在线调度二次开发项目** 已经具备较完整的 AI Infra 项目骨架，但仍有几个关键问题。以下判断基于仓库实现、README 的状态标签和现阶段项目表述。

## 1. 最大问题：核心创新已经实现，但还没有被真实 GPU 实验闭环证明

你现在项目最有价值的部分其实不是“跑通 Nano-vLLM”，而是你新增的：

```text
Prefill First
Decode First
Bounded Decode First
```

尤其是：

```text
Bounded Decode First
= Decode 优先
+ 连续 Decode 步数上限
+ 强制 Prefill
+ 防止新请求饥饿
```

这是整个项目最值得面试官追问的地方。

但目前简历真正能量化写出来的 GPU 数据只有：

> RTX 4060 / Qwen3-0.6B，256 个序列、133,966 Token、1241.94 Output Token/s。

问题在于：**这只是上游离线吞吐 baseline，不是你三种调度策略的在线 A/B 实验结果。**

因此目前证据链是：

```text
设计策略
↓
写代码
↓
CPU 契约测试通过
↓
？？？
↓
真实 GPU 在线实验
↓
三策略结果
↓
结论
```

最关键的中后半段还没闭环。

这是当前项目的 **第一优先级问题**。

---

## 2. 你的“核心贡献”和“上游已有能力”很容易混在一起

简历技术栈现在写了：

```text
Python、PyTorch、CUDA、Triton、FlashAttention、
Paged KV Cache、FastAPI、SSE
```

这会有一个风险。

面试官很可能问：

> “Paged KV Cache 是你实现的吗？”
>
> “FlashAttention 是你写的吗？”
>
> “Triton kernel 你改了什么？”
>
> “CUDA Graph 是你加的吗？”

但这些东西大部分是 **上游 Nano-vLLM 已有能力**，不是你的原创实现。

你真正做的是：

```text
Scheduler 策略
Async Engine
动态请求
Streaming
Abort
Backpressure
Benchmark
Metrics
```

所以目前项目存在一个“贡献边界容易产生误解”的问题。

这个问题在 README 里其实处理得很好，因为 README 已明确区分：

```text
上游：
Paged KV
Prefix Cache
FlashAttention
TP
CUDA Graph
基础 Chunked Prefill

你的二次开发：
三种调度策略
AsyncLLMEngine
SSE
请求取消
背压
Benchmark
```

但简历上这个边界还没有完全体现出来。

### 以后你讲项目时必须始终保持：

> **“我不是从零实现 Nano-vLLM，我是在已有推理执行框架上做在线调度和 Serving 二次开发。”**

这是可信度问题，不只是措辞问题。

---

## 3. 简历里的 Nano-vLLM 描述太密，像技术报告，不像简历

比如你第一条现在写的是：

> 在上游 Prefill First 基线上新增 Decode First、Bounded Decode First，形成三种可切换策略；以连续 Decode 步数为预算，在等待请求可执行时强制插入 Prefill，并记录抢占、KV Block 占用、分配阻塞和 Forced Prefill，建立 TTFT/TPOT 权衡的可解释依据。

这里技术含量没问题。

问题是信息量太大。

一条 bullet 同时出现：

```text
Prefill First
Decode First
Bounded Decode First
Decode budget
Forced Prefill
preemption
KV block
allocation blocked
TTFT
TPOT
```

面试官扫简历的时候很难第一眼抓住：

> **你到底解决了什么问题？**

你的简历应该变成：

```text
问题
↓
方案
↓
结果
```

而不是：

```text
实现了 A/B/C/D/E/F/G
```

比如最核心的逻辑实际上非常简单：

> 持续 Prefill 会干扰 Decode，纯 Decode First 又可能让新请求饥饿，所以设计 Bounded Decode First，在连续 K 个 Decode step 后强制 Prefill，以平衡 TTFT 和 TPOT。

这才是你项目的“故事”。

---

## 4. 项目现在“工程工作量很大”，但“优化成果感”还不够强

你其实已经写了很多工程内容。

例如：

```text
单 CUDA Worker
ADD/ABORT/SHUTDOWN
逐 Token SSE
请求取消
背压
慢消费者保护
KV Block 回收
Fake Runner
46 项 CPU 测试
```

从软件工程角度看，工作量很大。

但如果你投：

> **大模型推理优化实习**

面试官更关注：

```text
你优化了什么？
为什么慢？
瓶颈在哪里？
改完之后快了多少？
尾延迟改善多少？
吞吐损失多少？
显存变化多少？
```

你现在项目更像：

> “我把一个离线推理引擎改造成了更完整的在线推理系统。”

这是很好的 **Inference Serving Engineering**。

但“Performance Optimization”的标签还不够硬。

你需要最终出现类似这种真实结论：

```text
Bounded Decode First, K=8

相较 Prefill First：
P95 TPOT ↓ 18.7%
P95 TTFT ↑ 6.3%

相较 Decode First：
P95 TTFT ↓ 41.2%
P95 TPOT ↑ 3.8%

Goodput 提升 XX%
```

数字只是举例，不能写假数据。

一旦你有真实数据，项目才真正从：

> **“做了调度器”**

升级成：

> **“做了调度优化，并验证了收益。”**

---

## 5. 目前项目比较偏 Scheduler / Serving，底层 Kernel 深度仍然有限

这个问题要客观看待。

你现在 Nano-vLLM 涉及：

```text
Scheduler
Sequence
BlockManager
ModelRunner
Paged KV Cache
Prefill
Decode
CUDA Worker
```

所以已经非常适合：

```text
Inference Serving
推理框架
Scheduler
Runtime
```

但是如果岗位特别偏：

```text
CUDA Kernel
Triton
FlashAttention
算子融合
Memory Access
Tensor Core
```

那么这个项目本身并不能证明你真正做过 kernel 优化。

因为目前 Triton / FlashAttention 主要来自上游。

所以这个项目很适合作为：

> **“推理框架 / 调度”主项目**

但不应该试图同时包装成：

> 调度 + CUDA Kernel + 分布式 + Serving + 算子优化全都会。

这也是我为什么之前建议你再补一个独立 Triton 项目的原因。

---

## 6. 当前项目范围有点太宽，需要建立一个真正的“主线”

现在项目包含：

```text
Nano-vLLM
│
├─ Scheduler
├─ KV Cache
├─ Async Engine
├─ FastAPI
├─ SSE
├─ Cancel
├─ Backpressure
├─ Metrics
├─ Poisson workload
├─ Coordinated Omission
├─ GPU Telemetry
├─ Benchmark artifacts
└─ Goodput
```

工作量确实大。

但面试官可能问：

> **“你这个项目最核心的技术问题到底是什么？”**

如果你回答：

> “做了在线服务、SSE、背压、调度、benchmark……”

会显得散。

你应该把全部东西收敛到一个核心研究问题：

> **Prefill 和 Decode 竞争 GPU 时，如何在 TTFT 和 TPOT 之间做调度权衡？**

其他内容全部为这个问题服务：

```text
Async Engine
→ 为了接在线动态请求

SSE
→ 为了观察真实逐 token 延迟

Poisson
→ 为了模拟动态 arrival

Benchmark
→ 为了测 TTFT / TPOT

Step Stats
→ 为了验证调度行为

Backpressure
→ 为了保证在线服务稳定

Bounded Decode First
→ 核心解决方案
```

这样整个项目一下就“立住了”。

---

## 7. `46 passed` 是工程证据，不是性能证据

你现在简历写：

> 使用 Fake Runner 完成调度、并发与故障路径的 46 项 CPU 契约测试。

这个很好，必须保留某种程度的体现。

但是一定要理解它证明什么。

它证明：

```text
状态机正确
调度规则正确
取消逻辑正确
并发逻辑正确
异常路径正确
```

它不能证明：

```text
GPU 推理正确
性能变快
TTFT 改善
TPOT 改善
显存降低
吞吐提升
```

所以你现在的证据结构其实是：

```text
Correctness
★★★★☆

Performance
★★☆☆☆
```

后面的 GPU 实验就是补这个缺口。

---

## 最后压缩成 5 个最关键的问题

如果今天只记住这几个，就记：

1. **三种 Scheduler 策略还缺真实 RTX 4060 在线 A/B 数据，这是当前最大缺口。**
2. **上游 Nano-vLLM 能力和你自己的二次开发贡献必须明确区分。**
3. **简历描述太密，技术点很多，但“问题 → 方案 → 结果”的故事不够突出。**
4. **目前工程化工作量很强，但真正的“性能优化结果”还不够硬。**
5. **这个项目应该明确定位成 Scheduler / Inference Runtime / Serving 项目，不要强行包装成 CUDA Kernel 项目。**

所以我对这个项目的判断不是“有大问题”，而是：

> **现在已经有 70%～80% 的优秀推理实习项目形态了，真正缺的是最后那 20%：真实 GPU 在线实验、量化结论，以及把项目叙事进一步收敛。**

一旦三策略的真实 TTFT/TPOT/Goodput 数据跑出来，这个项目就会比现在完整很多。
