# Nano-vLLM 在线调度器项目展示（中文）

> **状态标签**：`Implemented`（在线调度扩展） · `CPU Verified`（46 tests passed） · `GPU Pending`
>
> **面试原则**：只讲能指向代码、测试或原始实验数据的事实。

## 1. 一句话介绍

我基于 Nano-vLLM 的离线批处理引擎，扩展一个支持在线请求、逐 token 流式输出、请求取消、队列背压和可观测调度策略的单机实验型推理服务，并计划在 RTX 4060 上用固定负载比较 Prefill/Decode 调度对 TTFT 与 TPOT 的影响。

在线扩展已通过纯 CPU 契约测试；面试时可以说“已实现并完成 CPU 验证”，但 GPU 证据补齐前不能说已经优化了多少。

## 2. 为什么值得做

离线 `generate()` 把全部请求先塞入队列，再等所有请求完成。真实在线服务更难：

- 新请求随时到达，不能等整批结束。
- 在途请求希望 Decode 连续，减少 token 间抖动。
- 新请求希望尽快 Prefill，降低首 token 等待。
- KV Cache 容量有限，请求取消或完成必须及时释放。
- 并发调用不能让多个线程同时操作同一个 CUDA 上下文。

这个项目的核心不是包一层 HTTP，而是把上述冲突变成可解释的状态机、调度策略和实验。

## 3. 架构讲解

```text
客户端
  │ POST /generate，消费 SSE
  ▼
命令通道 + 活跃请求总量上限 ── 负责接入与背压
  │
  ▼
单 worker 线程 ── 独占 CUDA 和所有可变引擎状态
  │ step 边界处理 ADD / ABORT / SHUTDOWN
  ▼
Scheduler ── 选择 Prefill 或 Decode
  ├─ Sequence 状态机
  ├─ BlockManager / Prefix Cache
  └─ ModelRunner / CUDA
  │
  ▼
RequestOutput + StepStats ── 分发给对应请求流和指标
```

## 4. 我能讲清楚的三条核心链路

### 4.1 新请求到首 token

`POST /generate → AsyncLLMEngine.submit → ADD 命令 → worker step 边界 drain → LLMEngine.add_request → Scheduler.waiting → Prefill → sample token → RequestOutput → SSE token`

TTFT 覆盖从服务接收请求到第一个 token 事件，不只是 GPU Prefill kernel。

### 4.2 Decode 与 KV Cache

Prefill 后序列进入 running。Decode 每步只为每个选中序列调度一个 token；`prepare_decode` 使用最后 token、当前位置、上下文长度和 block table，FlashAttention 从分页 KV Cache 读取历史上下文。生成新 token 后，序列长度增加；跨 block 边界前分配新块。

### 4.3 取消请求

`DELETE → ABORT 命令 → worker 下一个 step 边界 → 从 waiting/running 移除 → BlockManager.deallocate → 发布 finish_reason=abort → 关闭该请求 stream`。取消不是中断正在运行的 CUDA kernel。

## 5. 三种策略怎样解释

| 策略 | 直觉 | 关注指标 |
|---|---|---|
| Prefill First | 新请求来了优先做首轮计算 | TTFT、在途请求 TPOT 抖动 |
| Decode First | 先让已有请求继续吐 token | TPOT、新请求尾部 TTFT |
| Bounded Decode First | Decode 连续达到上限后插一次 Prefill | TTFT/TPOT 折中、强制 Prefill 次数 |

`max_consecutive_decode_steps=8` 是可控实验变量，不是“理论最优值”。我要通过固定负载 A/B 实验回答它在 RTX 4060 和指定模型上的行为。

## 6. 上游与本人贡献边界

### 上游 Nano-vLLM 已有

- Qwen3 模型结构和权重加载。
- FlashAttention、Triton KV Cache 写入。
- Paged KV Cache 与 Prefix Cache。
- Tensor Parallel 和 CUDA Graph。
- waiting/running 基础调度、Chunked Prefill、抢占。
- 离线 `LLM.generate()`。

### 本阶段本人已实现并完成 CPU 验证

- 可切换的 Prefill/Decode 调度策略与有界 Decode。
- 外部请求 ID、逐步输出与取消语义。
- 单 worker 异步引擎、命令队列和背压。
- SSE 生成/取消、健康和指标接口。
- CPU 契约测试与 RTX 4060 可复现实验材料。

面试表达：

> 我没有把上游引擎包装成自己从零实现。我先读懂它的 Scheduler、Sequence、BlockManager 和 ModelRunner 调用链，然后把离线调度扩展为可取消、可观测的在线调度，并用测试和原始实验数据证明我修改的边界。

## 7. 难点与取舍

### 难点一：Prefill 与 Decode 的目标冲突

持续 Prefill 会干扰在途 token 间隔；持续 Decode 会让新请求迟迟拿不到首 token。有界 Decode 通过显式上限避免策略无限偏向一侧，并用 `forced_prefill` 让行为可观测。

### 难点二：并发不等于多线程调用 CUDA

HTTP 请求可以并发，但 CUDA 引擎状态需要单一所有者。我选择单 worker + 命令队列，使并发发生在提交和消费层，模型执行仍串行地产生动态批次。

### 难点三：取消与资源回收

取消不仅是给客户端返回成功，还必须从调度队列删除序列、释放引用计数块、只发一个 abort 终态，并处理与完成事件同时到达的竞态。

### 难点四：性能数字的可信度

只跑一次吞吐没有意义。正式报告要固定模型、软件、输入输出、并发和策略；warmup 后重复运行，保留逐请求与逐 step 原始数据，并先通过正确性门禁。

## 8. 两分钟面试话术

> 这个项目基于 Nano-vLLM。上游已经有模型执行、Paged KV Cache、Prefix Cache 和离线调度，我的工作边界是在线调度扩展。原来的 generate 会先加入整批请求，然后同步 step 到全部结束；我把它拆成带 request ID 的逐步输出，并设计 AsyncLLMEngine，让单 worker 线程独占 CUDA，外部并发请求通过命令通道进入，Scheduler 的 waiting+running 总量上限负责 admission 背压。
>
> 调度上我保留 Prefill First，又增加 Decode First 和 Bounded Decode First。后者优先保障在途请求的 token 间隔，但连续 Decode 达到上限且存在 waiting 请求时，强制执行一次 Prefill，避免新请求饥饿。每一步都输出 batch 类型、KV block、抢占和 forced prefill 等统计，所以我能解释指标变化来自什么调度行为。
>
> 取消也在 step 边界处理：序列从队列移除，KV block 释放，然后发唯一的 abort 终态。CPU 测试会用 fake runner 覆盖状态机、背压和竞态；GPU 部分会在 RTX 4060 固定负载下比较 TTFT、TPOT 和吞吐。没有原始数据前我不会声称具体提升。

## 9. 高频追问与短答

### 为什么不让每个 HTTP 请求直接调用 `step()`？

因为 Scheduler、KV Block 和 CUDA 上下文都是共享可变状态。多个 handler 直接调用会产生竞态，也难以保证 CUDA 线程所有权。命令队列把接入并发与模型执行串行化分开。

### Bounded Decode First 能保证绝对公平吗？

它只保证“存在可执行 waiting 请求时，连续 Decode 不超过配置上限”。KV 分配失败、超长 Prefill 或请求尺寸差异仍会影响公平性，因此需要 `allocation_blocked` 和分位数指标观察。

### 为什么取消不是立刻生效？

已发射的 CUDA kernel 通常不在这里安全中断。取消命令在下一个 step 边界被 drain，边界清晰，资源状态也更容易证明正确。

### Prefix Cache 和普通 KV Cache 有什么区别？

普通 KV Cache 保存单个请求已经计算的历史 K/V，Decode 避免重复算历史 token；Prefix Cache 用完整块哈希复用多个请求共享的前缀块，并通过引用计数管理共享生命周期。

### 为什么 `text` 要累计解码？

子词 token 单独解码再字符串拼接可能与一次性解码累计 token 不一致。累计 token 是事实源，文本是它的投影。

### 怎样证明没有 KV 泄漏？

在完成、长度结束、取消、抢占重跑和 worker 异常路径都断言块集合与引用计数不变量；一轮测试结束后 `kv_used_blocks` 应回落到预期基线。

## 10. 演示脚本

1. 展示三个并发 SSE 请求逐 token 返回。
2. 生成期间取消一个请求，观察唯一的 abort 终态。
3. 设置很小队列触发背压，展示稳定错误码。
4. 切换三种策略，展示 `batch_kind` 和 `forced_prefill` 序列。
5. 打开 `/metrics`，解释 waiting/running、KV block 和抢占。
6. 展示 `results/` 原始 JSONL 如何生成实验汇总。

当前可演示代码、纯 CPU 测试，以及 benchmark 原始产物的生成链路与 schema；真实模型 SSE/CUDA 性能演示仍等待 WSL2 + RTX 4060。

## 11. 简历表述模板

只有对应证据完成后才采用：

- 实现 Nano-vLLM 在线调度扩展：基于单 CUDA worker、命令通道与活跃请求总量上限支持并发请求接入、逐 token SSE、请求取消与背压，并以 CPU 契约测试覆盖状态机和资源回收。
- 设计 Prefill First、Decode First、Bounded Decode First 三种策略，记录逐 step 批次、KV Block、抢占与强制 Prefill，构建 TTFT/TPOT/吞吐 A/B 实验链路。
- 在 RTX 4060 上固定模型与负载完成重复实验并归档原始 JSONL；具体指标必须从最终报告填写，不预写提升百分比。

## 12. 当前诚实状态

- `Implemented`：上游离线核心与本人在线调度、异步 worker、SSE 适配、benchmark 代码均可定位。
- `CPU Verified`：`python -m pytest -q` 为 `46 passed`；Ruff、`compileall` 与 `git diff --check` 通过。
- `GPU Pending`：RTX 4060 表格保持空白，无个人 GPU 性能结论。

开发主机 Python 3.13 只用于纯 CPU 测试；项目声明 `>=3.10,<3.13`，正式 WSL 运行使用 Python 3.11。这一差异必须主动说明，不能把 CPU 测试解释成受支持版本上的 GPU 运行。
