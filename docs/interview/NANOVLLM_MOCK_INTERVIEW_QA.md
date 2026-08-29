# Nano-vLLM 推理优化实习模拟面试 Q&A 清单

> 适用岗位：大模型推理框架优化、AI Infra、推理加速、高性能算子、训推框架实习。
>
> 文档基线：上游基线 `bb823b3` + 当前在线调度二次开发分支。整理日期：2026-08-22。

> 2026-08-28 更新：在线 GPU 实验已完成，面试数字以[15-run 正式报告](../../reports/nanovllm-online-rtx4060-20260828.md)为准；下文原 GPU Pending 口径已同步更新。

## 0. 使用说明与事实边界

这不是一份只背结论的“八股题库”。每道重点题都按下面四层准备：

1. **第一句话先给结论**：10～20 秒内回答核心问题。
2. **解释原理**：说清数据结构、计算过程或资源瓶颈。
3. **落到本项目源码**：指出类、方法和状态如何变化。
4. **说明边界和验证方法**：避免把假设说成事实、把能运行说成性能提升。

### 当前项目事实边界

当前仓库基于上游 `GeeeekExplorer/nano-vllm`。上游基线已实现：

- 离线批量生成接口；
- `waiting/running` 双队列调度；
- Chunked Prefill；
- Paged KV Cache 和 Prefix Cache；
- KV Cache 不足时的 Recompute 抢占；
- Tensor Parallel；
- Decode CUDA Graph；
- FlashAttention 变长 Prefill 和 KV Cache Decode；
- Triton KV Cache 写入内核。

当前分支在上游之上新增并完成纯 CPU 契约验证：

- `prefill_first`、`decode_first`、`bounded_decode_first` 三种调度策略；
- 外部请求 ID、逐 token 累计输出、取消与背压；
- 单 worker `AsyncLLMEngine`，由同一线程独占 CUDA 和可变引擎状态；
- FastAPI/SSE 生成、取消、健康与指标接口；
- fixed、Poisson 和 first-token barrier 在线负载及原始产物归档；
- 46 项纯 CPU 契约测试。

Milestone 0 已在 RTX 4060 上完成上游离线 `bench.py` 基线；在线调度的 CUDA/SSE 正确性和 TTFT/TPOT A/B 已完成三策略 × 5 次重复并公开证据。不能把上游已有机制说成“我从零实现”，也不能把离线吞吐写成在线优化结果。安全表达是：

> 我先读懂 Nano-vLLM 上游的 Scheduler、Sequence、BlockManager 和 ModelRunner 链路，再完成在线调度二次开发：新增 Decode First 与 Bounded Decode First、动态请求、逐 token 流、取消、背压和 SSE，并用 46 项纯 CPU 测试验证状态机和资源回收。RTX 4060 离线基线已经完成，但在线三策略 A/B 仍等待原始 GPU 数据，所以我不声明具体在线性能提升。

### 60 分钟模拟一面节奏

| 时间 | 环节 | 面试官主要判断 |
|---|---|---|
| 0～5 分钟 | 自我介绍、项目归属 | 表达能力、项目真实性 |
| 5～18 分钟 | Nano-vLLM 架构和源码链路 | 是否真正读过代码 |
| 18～33 分钟 | KV Cache、PagedAttention、调度 | 推理框架核心基本功 |
| 33～43 分钟 | FlashAttention、CUDA、算子 | 是否理解性能来自哪里 |
| 43～50 分钟 | 量化、多卡 | 知识宽度和推导能力 |
| 50～60 分钟 | 手写题 | 编码下限、边界意识 |

---

## 1. 自我介绍与项目真实性

### Q1：请用 90 秒做自我介绍

**参考回答：**

> 面试官您好，我目前主要学习大模型推理部署和推理框架优化。为了理解 vLLM 的核心链路，我选择了代码规模较小但结构完整的 Nano-vLLM，先分析请求从 `LLMEngine` 进入 `Scheduler`，经过 Paged KV Cache、Prefill/Decode、Attention 和采样，再回到调度器更新状态的全过程。在此基础上，我完成了面向在线请求的调度二次开发：增加 Decode First 和 Bounded Decode First，通过单 CUDA worker 支持动态请求、逐 token 输出、取消、背压和 SSE，并记录逐请求、逐 token、逐 step 的实验数据。46 项纯 CPU 契约测试已经通过；RTX 4060 上三策略各 5 次重复显示，在固定 workload 下 Bounded(K=8) 相对严格 Decode First 将 TTFT P50 从 108.9 s 降至 4.58 s、输出吞吐从 50.8 提升至 446.6 tok/s。该结论只对应公开证据包中的模型和负载。我希望在推理框架岗位继续加强 CUDA Profiling、算子和多卡能力。

**继续追问：**

- 为什么从推理部署切入，而不是先做训练框架？
- 你能在白板上画出完整请求链路吗？
- 哪部分是你真正独立完成的？
- 最近解决的一个具体问题是什么？

**注意：**不要说没有证据的性能数字，不要把“读懂”说成“实现”。

### Q2：为什么选择 Nano-vLLM，而不是直接读 vLLM？

**参考回答：**

上游 Nano-vLLM 用较小的 Python 代码量保留了推理引擎的主要骨架，适合先建立完整心智模型；vLLM 的生产代码包含更多模型、后端、分布式执行器和兼容逻辑，初学时容易只看到局部。先读 Nano-vLLM 可以把 Scheduler、BlockManager、ModelRunner、Attention 串起来，再带着问题进入 vLLM。

上游基线的公开接口主要是离线同步生成；当前分支虽然补齐了动态接入、流式输出、取消、背压和 SSE，但仍是单机实验型服务。模型覆盖、超时治理、鉴权限流、多租户、分布式执行和可观测性仍比生产级 vLLM 简化。

### Q3：项目中最难的地方是什么？

**当前阶段的诚实回答：**

> 第一处难点是 Sequence 中“token 已经追加”和“这个 token 的 KV 已经写入”不是同一时刻，它直接影响新块申请、`num_cached_tokens`、`slot_mapping` 和 Decode 边界。第二处难点是 Decode 优先不能简单写成永远先跑 running，否则 waiting 会饥饿；我增加连续 Decode 计数和 `forced_prefill/allocation_blocked` 统计，在 waiting 可分配时用 K 步上限约束偏向，在资源不可分配时显式回退。第三处难点是并发所有权，我用单 worker 在 step 边界统一处理 ADD、ABORT 和 SHUTDOWN，避免 HTTP 线程直接操作 CUDA 或释放 KV。

验证上，我用 fake runner 覆盖调度边界、取消、重复 ID、背压、慢消费者和 worker 故障；代价是当前仍是 step-level phase scheduling，同一次模型调用不混合 Prefill 与 Decode。

### Q4：你如何证明自己不是只把项目跑起来？

**参考回答：**

- 能脱离代码讲出请求生命周期和关键状态；
- 能用一个跨块边界的例子手算 Block Table 和 Slot Mapping；
- 能解释为什么 `can_append()` 在特定余数时申请块；
- 能设计 Prefix Cache、抢占重算和 cached Decode 的正确性测试；
- 能指出当前调度策略的能力边界及指标冲突；
- 性能结论必须保留固定环境、warmup、重复实验和原始结果。

### Q5：如果面试官问“哪些代码是你写的”，怎么回答？

**参考回答模板：**

> 上游已经提供 ModelRunner、BlockManager、Paged KV Cache、Prefix Cache、Chunked Prefill、CUDA Graph 和 Tensor Parallel，我不会把它们说成从零实现。我的代码边界是：扩展 Scheduler 的三种策略和有界 Decode；在 LLMEngine 增加外部请求 ID、逐步输出与取消；实现单 worker AsyncLLMEngine、SSE 服务、背压和在线 benchmark；并补充 46 项纯 CPU 契约测试。RTX 4060 离线基线有真实报告，在线性能仍等待正式实验。

---

## 2. Nano-vLLM 架构与完整调用链

### Q6：请介绍 Nano-vLLM 的整体架构

**参考回答：**

可以分成五层：

1. **接口层**：`nanovllm/llm.py` 暴露 `LLM`，实际继承 `LLMEngine`。
2. **引擎层**：`LLMEngine` 负责 tokenizer、请求加入、迭代执行和结果汇总。
3. **调度与内存层**：`Scheduler` 管 waiting/running；`BlockManager` 管物理 KV 块、引用计数和 Prefix Cache。
4. **执行层**：`ModelRunner` 准备 Prefill/Decode 输入、分配 KV Cache、捕获 CUDA Graph，并驱动模型执行。
5. **模型与算子层**：Qwen3 模型、张量并行 Linear、RoPE、RMSNorm、Attention、Sampler。

### Q7：一个 prompt 的完整调用链是什么？

**参考回答：**

```text
LLM.generate
  → LLMEngine.add_request
  → tokenizer.encode（如果输入是字符串）
  → Sequence(prompt, sampling_params)
  → Scheduler.add，进入 waiting
  → 循环 LLMEngine.step
      → Scheduler.schedule
      → ModelRunner.run
          → prepare_prefill / prepare_decode
          → Qwen3ForCausalLM.forward
          → Attention.forward
          → compute_logits
          → Sampler
      → Scheduler.postprocess
          → 更新 cache/token 状态
          → append_token
          → 完成时释放 KV block
  → 按 seq_id 排序并 decode 文本
```

**源码落点：**

- `nanovllm/engine/llm_engine.py::generate/add_request/step`
- `nanovllm/engine/scheduler.py::schedule/postprocess`
- `nanovllm/engine/model_runner.py::run`
- `nanovllm/layers/attention.py::Attention.forward`

### Q8：Sequence 保存了哪些关键状态？

**参考回答：**

- `seq_id`：全局递增请求编号；
- `status`：WAITING、RUNNING、FINISHED；
- `token_ids`、`num_tokens`、`num_prompt_tokens`；
- `num_cached_tokens`：已经存在有效 KV 的 token 数；
- `num_scheduled_tokens`：本 iteration 计划计算的 token 数；
- `is_prefill`；
- `block_table`：该请求的逻辑块到物理块映射；
- 采样和停止参数：temperature、max_tokens、ignore_eos。

`num_tokens` 不一定等于 `num_cached_tokens`：刚采样并追加到 Sequence 的新 token，要到下一轮 Decode 才生成并写入自己的 KV。

### Q9：Prefill 和 Decode 的输入准备有什么区别？

**参考回答：**

Prefill 可以为每个请求安排多个 token，通过扁平化的 `input_ids` 和 `positions`，配合 `cu_seqlens_q/cu_seqlens_k` 描述变长序列；如果命中 Prefix Cache，还会传入 Block Table，使新 Q 能访问缓存中的历史 K/V。

Decode 每个请求只输入 `last_token`，同时传 `context_lens`、`slot_mapping` 和 Block Table。Attention 读取历史分页 KV，并把当前 token 的 K/V 写到物理槽位。

### Q10：`cu_seqlens_q` 和 `cu_seqlens_k` 是什么？

**参考回答：**

它们是各序列 Q 和 K 的累积长度边界。例如两条 Q 长度为 3、2，则 `cu_seqlens_q=[0,3,5]`。FlashAttention 用它在一个扁平 Tensor 中定位不同变长序列，避免把所有请求 padding 到相同长度。

Prefix Cache 命中时，Q 只包含未缓存 token，而 K 的有效历史包括缓存前缀，所以 `seqlen_k` 可能大于 `seqlen_q`。

### Q11：为什么 Prefill 只为每条序列采样一个 token？

**参考回答：**

模型会为 Prefill 的每个位置产生 hidden state，但自回归生成只需要 prompt 最后一个有效位置的 logits 来采样第一个输出 token。`ParallelLMHead.forward()` 在 Prefill 情况下用 `cu_seqlens_q[1:] - 1` 选取每条序列最后一个 Q 位置。

### Q12：Sampler 使用的是什么采样方法？

**参考回答：**

先用 temperature 缩放 logits 并做 Softmax，然后利用指数随机变量实现 Gumbel-max 等价采样：`probs / Exp(1)` 后取 argmax。项目明确禁止 temperature 近似为 0，因此没有提供 greedy sampling，也没有 top-k/top-p。

### Q13：为什么 ModelRunner 要在初始化时 warmup？

**参考回答：**

一是触发模型、编译或库的初始化开销；二是测量模型执行峰值显存，以估算剩余显存能放多少 KV blocks；三是在捕获 CUDA Graph 前让相关路径稳定。否则首次运行开销和显存峰值会污染容量估算和 benchmark。

### Q14：KV Cache 块数怎样计算？

**参考回答：**

单个物理块的字节数近似为：

```text
2 × num_layers × block_size × local_num_kv_heads × head_dim × dtype_bytes
```

其中 `2` 是 K 和 V。`local_num_kv_heads = total_num_kv_heads / tensor_parallel_size`。项目在 warmup 后读取总显存、当前占用和峰值，再用可用于 KV Cache 的字节数除以单块字节数，得到 `num_kvcache_blocks`。

### Q15：为什么 KV Cache 是 `[2, layers, blocks, block_size, kv_heads, head_dim]`？

**参考回答：**

- 第一维区分 K/V；
- 每层 Attention 有独立 K/V；
- blocks 是分页后的物理块池；
- block_size 是每块 token 数；
- 最后两维是本 TP rank 的 KV heads 和 head dimension。

随后每个 Attention 层拿到 `kv_cache[0, layer_id]` 和 `kv_cache[1, layer_id]`。

### Q16：CUDA Graph 为什么主要用于 Decode？

**参考回答：**

Decode 每轮每请求通常一个 token，形状变化主要来自 batch size，比较容易按若干 batch bucket 捕获固定图。Prefill 的 token 数和变长边界变化更大，直接捕获会需要大量图或复杂 padding。

项目为 Decode 捕获多个 batch size；实际 batch 会选择不小于它的最小 graph bucket，并用 `slot_mapping=-1`、`context_lens=0` 等方式屏蔽 padding lane。

**继续追问：**CUDA Graph减少的是重复 kernel launch 和 CPU 调度开销，不会减少模型本身的算术量。

---

## 3. KV Cache、PagedAttention 与 Prefix Cache

### Q17：KV Cache 为什么能加速自回归 Decode？

**参考回答：**

自回归第 `t` 步只新增一个 token。历史 token 在每一层产生的 K/V 不会变化，因此可以缓存。下一步只计算新 token 的 Q/K/V，让新 Q 与历史 K 做注意力，再加上新 V；无需为历史 token 重算 K/V。

它减少重复计算，但代价是 KV Cache 随层数、上下文长度、并发数线性增长，并且 Decode 每步需要读取大量历史 KV，容易受显存带宽限制。

### Q18：KV Cache 显存公式是什么？

**参考回答：**

对标准全注意力 Decoder，单请求近似为：

```text
KV bytes = 2 × L × S × H_kv × D × bytes_per_element
```

- `L`：层数；
- `S`：当前序列长度；
- `H_kv`：KV head 数；
- `D`：head dimension；
- `2`：K 和 V。

总显存还要乘并发请求数。GQA/MQA 用更少的 KV heads，因此不能误用 Query head 数计算。TP 下每个 rank 通常只保存本地 KV heads。

### Q19：PagedAttention 解决了什么？没有解决什么？

**参考回答：**

它把每条序列逻辑上连续的 KV Cache 切成固定大小逻辑块，再通过 Block Table 映射到不连续的物理块。物理块按需分配，因此不必为最大序列长度预留连续显存，并支持快速回收、共享前缀和 Copy-on-Write 类机制。

它主要解决 KV 内存管理和利用率问题，**不会把标准 Attention 的计算复杂度从 `O(n²)` 变成 `O(n)`**。分页还会引入 Block Table 查询和非连续访存，需要专门内核处理。

### Q20：手算一次分页地址映射

**题目：**

```text
block_size = 4
block_table = [7, 2, 11]
token_position = 9
```

**参考回答：**

```text
logical_block = 9 // 4 = 2
offset = 9 % 4 = 1
physical_block = block_table[2] = 11
```

因此访问物理块 11 的块内位置 1。若将整个物理块池展平，token 级 slot 可写为 `11 × 4 + 1 = 45`。

### Q21：Block Table 和 Slot Mapping 的区别是什么？

**参考回答：**

- Block Table 面向整条请求，记录“逻辑块编号 → 物理块编号”，Attention读取历史分页KV时使用。
- Slot Mapping 面向本轮将要计算的每个 token，直接给出它的 K/V 应写入物理 Cache 的扁平槽位。

一个用于读历史映射，一个用于写当前 token；它们相关但不等价。

### Q22：block size 越小越好吗？

**参考回答：**

不是。

- 块大：每条序列最后一个块的内部浪费上升，Prefix Cache 命中粒度更粗。
- 块小：Block Table 更长，元数据、分配管理和地址查询开销增加，也可能影响内核访存效率。

应在内部碎片、管理开销、Prefix复用粒度和Attention内核效率之间折中，并用真实请求长度分布测量。

### Q23：Prefix Cache 怎样匹配相同前缀？

**参考回答：**

项目只对完整块建立哈希。每块 hash 同时依赖上一块 hash 和当前块 token IDs，形成链式前缀身份。新请求从第一个完整块开始逐块计算 hash，并在 `hash_to_block_id` 中查找；连续命中多少块，就跳过多少块的 Prefill。

不仅哈希要相同，代码还比较 `block.token_ids`，降低哈希碰撞导致错误复用的风险。

### Q24：为什么 hash 要包含前序块 hash？

**参考回答：**

相同的当前块 token 出现在不同前文后，产生的 KV 并不相同，因为它的隐藏状态依赖此前上下文。链式 hash 把完整前缀身份包含进来，避免只看局部 token 块而错误复用。

### Q25：为什么通常只缓存完整块？

**参考回答：**

完整块内容稳定且具有明确边界，便于建立不可变的 hash 和跨请求共享。部分块还可能继续追加 token；如果直接共享，后续写入就需要 Copy-on-Write 或更复杂的版本管理。当前实现选择只复用完整块，换取简单和安全。

### Q26：引用计数怎样保证共享块安全？

**参考回答：**

当新请求复用一个仍被其他请求使用的物理块时，`ref_count += 1`。请求结束或被抢占时逐块减一，只有降到零时才回到 free queue。这样一个请求结束不会释放另一个请求仍在读取的共享块。

空闲但保留 hash 的块可继续作为 Prefix Cache；真正重新分配该块时，旧 hash 映射才会被移除并重置。

### Q27：Prefix Cache 的哈希碰撞会有什么后果？

**参考回答：**

如果只依赖非加密 hash，碰撞可能让请求读取错误前缀 KV，导致错误输出；多租户场景还可能形成信息泄漏风险。本项目额外比较 token IDs，但生产系统仍需要根据性能与安全要求选择 hash、加入租户/LoRA/多模态等额外身份，并测试碰撞处理。

### Q28：显存不足时 Nano-vLLM 怎么抢占？

**参考回答：**

Decode 请求需要新块但空闲块不足时，Scheduler 从 running 队尾选择请求抢占。`preempt()` 把它改回 WAITING/PREFILL，释放全部 Block Table，并放到 waiting 队首。恢复时重新分配并重新 Prefill，因此属于 Recompute，而不是把 KV Swap 到 CPU。

代价是省下 CPU-GPU 传输和 CPU 内存，但重新计算会增加延迟和算力消耗；频繁抢占可能出现抖动或饥饿。

### Q29：为什么 `can_append()` 在 `len(seq) % block_size == 1` 时需要新块？

**参考回答：**

这是一个容易误判的源码题。`postprocess()` 先把刚采样的 token 追加进 Sequence，但这个新 token 的 K/V 尚未计算。下一轮 Decode 才拿它作为输入并写 KV。

如果旧块原本刚好填满，采样追加后 `len(seq)` 会变成 `old_multiple + 1`，新 token 是新逻辑块中的第一个 token，因此此时必须先申请物理块。不能机械地改为余数 `0`。

### Q30：怎样测试 Paged KV Cache 正确性？

**参考回答：**

至少覆盖：

1. cached Decode logits 与完整历史重新 Prefill 的最后位置 logits 对齐；
2. prompt 和生成跨越 block 边界；
3. 两请求共享多个完整前缀块，输出与禁用 Prefix Cache 一致；
4. 同 hash 不同 token 不错误复用；
5. 请求结束后 ref count、used/free blocks恢复；
6. 抢占后 Recompute 输出一致；
7. 最后一个部分块不进入 Prefix Cache；
8. EOS、max_tokens和 ignore_eos 结束路径都释放内存。

---

## 4. 调度、Continuous Batching 与在线服务

### Q31：Static Batching 和 Continuous Batching 有什么区别？

**参考回答：**

Static Batching 通常等待一批请求组成固定 batch，直到整批完成才替换；短请求完成后留下空槽，受到最长请求拖累。

Continuous Batching 在每个模型 iteration 重新组织 batch：完成的请求立即退出，新请求可以加入，运行中的请求继续 Decode，因此更适合输出长度不一致的在线生成。

### Q32：当前 Nano-vLLM 到底有没有 Continuous Batching？

**参考回答：**

需要区分上游接口和当前分支：

- 上游 Scheduler 每轮重新从 waiting/running 选择序列，已有迭代级动态组批核心；但公开 `generate()` 一次性加入 prompts 并同步等待全部完成。
- 当前分支增加 `add_request/step_stream/abort_request`、单 worker `AsyncLLMEngine` 和 SSE，使请求可以在其他请求运行期间动态进入、逐 token 返回并被取消。
- 当前仍是单机实验型服务，没有生产级鉴权、持久化、跨进程 API worker、超时治理和多租户资源隔离。

所以当前分支已经实现在线 Continuous Batching 的完整实验闭环，但不能表述为生产级 vLLM serving。

### Q33：当前 Scheduler 的 Prefill/Decode 策略是什么？

**参考回答：**

当前 Scheduler 可配置三种策略：

- `prefill_first` 保留上游行为，有可执行 waiting 时优先 Prefill；
- `decode_first` 有 running 时优先 Decode，但可能让新请求 TTFT 饥饿；
- `bounded_decode_first` 优先 Decode，连续达到 K 步且 waiting 可执行时强制一次 Prefill。

`Config` 默认仍是 `prefill_first` 以保持离线兼容，在线服务 CLI 默认 `bounded_decode_first`、K=8。K 是实验变量，不是理论最优。当前一次 `ModelRunner.run()` 仍只有全局 `is_prefill`，所以同一模型调用不会混合两类输入。

### Q34：为什么 Chunked Prefill 有用？

**参考回答：**

长 prompt 如果一次占满整个 token budget，会长时间阻塞 Decode。Chunked Prefill 把长 prompt 拆成多轮，每轮只处理一部分 token，使调度器可以在中间插入其他请求并限制单轮工作量。

它改善调度灵活性，但会增加多轮调度和 kernel 调用，也必须正确维护 `num_cached_tokens`、Q/K长度、Slot Mapping和部分块边界。

### Q35：`max_num_batched_tokens` 和 `max_num_seqs` 分别限制什么？

**参考回答：**

- `max_num_batched_tokens`：一次 Prefill iteration 最多处理的 token budget，也是 Chunked Prefill 的切分依据。
- `max_num_seqs`：一次可调度的最大序列数，Decode 时近似限制 batch size。

增大 token budget 可能提高 Prefill 吞吐或 TTFT，但可能挤压 Decode、增加单轮时长；增大 seq 数可提高并发和吞吐，但需要更多 KV Cache，并可能提高单 token延迟。

### Q36：如果让你做在线 Continuous Batching，你会怎么改？

**参考回答：**

我已经先完成不改 Attention 内核的第一阶段闭环：

1. `submit/generate/abort` 异步接口和逐 token 单请求 stream；
2. 单 worker 独占 CUDA，在 step 边界 drain ADD/ABORT/SHUTDOWN；
3. Prefill First、Decode First、Bounded Decode First 三种策略；
4. Scheduler 活跃请求总量背压、慢消费者保护和统一 KV 回收；
5. `RequestOutput/StepStats` 记录 token、队列、KV、抢占和强制 Prefill；
6. SSE 的生成、取消、健康和指标端点；
7. fixed、Poisson 和 barrier interference workload 及原始 JSONL。

下一阶段是在 RTX 4060 上补齐在线 GPU 正确性和三策略重复实验，再根据 profiler 结果决定是否做混合 Prefill/Decode metadata、两个 micro-batch 或更深的 Attention 内核改造。

### Q37：为什么不能只把 Prefill 和 Decode 两个 list 拼起来？

**参考回答：**

当前 `Context` 只有一个全局 `is_prefill`，`Attention.forward()` 据此选择 `flash_attn_varlen_func` 或 `flash_attn_with_kvcache`。两类请求的 Q长度、K历史、采样位置、CUDA Graph形状和元数据不同，因此混合执行不是简单拼 Tensor；要设计能描述每类 token 的统一 metadata，或在一个 iteration 内执行两个 micro-batch。

### Q38：如何处理调度公平性？

**参考回答：**

仅 FCFS 容易被超长 prompt 形成 head-of-line blocking；仅 Decode优先又可能让新请求长期拿不到首 token。可以组合：

- Decode基础优先级；
- Prefill Chunking；
- 等待时间老化；
- long-prefill并发上限；
- 每轮为 Prefill 保留最小budget；
- 按SLO违约风险动态调整。

策略必须用 TTFT、TPOT/P99、吞吐和饥饿次数共同评价，不能只看总吞吐。

### Q39：Preemption 应选择哪个请求？

**参考回答：**

没有无条件最优策略。可以比较：

- 抢占队尾/最新请求：保护FCFS，逻辑简单；
- 抢占已生成较少的请求：减少重算成本；
- 抢占低优先级或SLO更宽松请求；
- 考虑其已占blocks、剩余长度预测和Prefix可复用程度。

评价指标包括释放块数、重算token数、被重复抢占次数、公平性和尾延迟。

### Q40：怎样设计调度 A/B 实验？

**参考回答：**

固定模型、权重版本、GPU、软件版本、dtype、block size、输入/输出长度分布、请求数和随机种子。分别测试固定并发、固定速率和 Poisson 到达；先 warmup，再重复多轮。记录：

- 请求吞吐、输入/输出 token吞吐；
- TTFT mean/P50/P95/P99；
- TPOT/ITL mean/P99；
- E2E延迟；
- Goodput；
- KV块利用率、Prefix命中率、抢占/重算次数；
- OOM、失败请求和实际输出token数。

结论必须附原始结果，不能只保存汇总表。

---

## 5. Attention、FlashAttention 与并行效率

### Q41：标准 Self-Attention 的计算过程是什么？

**参考回答：**

```text
Q = XWq, K = XWk, V = XWv
S = QK^T / sqrt(d)
P = softmax(S)
O = PV
```

对序列长度 `N`、head dimension `D`，`QK^T` 和 `PV` 的主要计算量是 `O(N²D)`。朴素实现还会将 `N×N` score/probability中间结果写入和读回HBM，造成大量显存访问和 `O(N²)` 中间存储。

### Q42：Attention 如何提升并行计算效率？

**参考回答：**

可以从多个维度并行：batch、attention head、query token tile、key/value tile和head dimension。工程优化重点包括：

- 将 Q/K/V projection 合并成一次更大的 GEMM；
- 用 Tensor Core 友好的数据类型和尺寸；
- 对变长序列使用 packed layout，减少 padding；
- 分块计算 QK、Softmax 和 PV，减少HBM中间读写；
- 融合 scale、mask、Softmax 等操作；
- Decode 对多个请求和 heads组批，提高并行度；
- GQA/MQA 减少KV存储和读取；
- 分页场景用专门内核高效遍历 Block Table。

不能只回答“多开线程”，必须说明并行维度、数据复用和瓶颈。

### Q43：FlashAttention 为什么快？

**参考回答：**

FlashAttention 是 IO-aware 的精确 Attention。它把 Q/K/V 分块搬到片上存储，在片上完成局部 `QK^T`、在线 Softmax 和 `PV` 累加，不把完整 `N×N` score/probability矩阵写回HBM，从而显著减少HBM访问和中间显存。

它没有把标准全Attention的渐进计算复杂度从 `O(N²D)` 改掉，也不是稀疏或近似Attention；主要收益来自IO复杂度和kernel融合。

### Q44：Online Softmax 怎么合并不同 tile？

**参考回答：**

对每一行维护运行最大值 `m`、归一化和 `l`、加权Value累积 `acc`。新tile分数为 `s`：

```text
m_new = max(m, max(s))
alpha = exp(m - m_new)
p = exp(s - m_new)
l_new = alpha * l + sum(p)
acc_new = alpha * acc + p @ V_tile
```

遍历完全部K/V tiles后输出 `acc / l`。当最大值更新时，用 `alpha` 重缩放旧累积量，保证数值稳定。

### Q45：FlashAttention 和 PagedAttention 有什么区别？

**参考回答：**

- FlashAttention关注Attention计算过程中的IO和中间矩阵物化。
- PagedAttention关注推理服务中KV Cache的分页分配、地址映射、共享和回收。

两者可以组合：KV按块存储，Attention kernel根据Block Table读取分页KV，同时采用分块和在线Softmax减少HBM流量。

### Q46：Prefill 为什么通常更像 compute-bound，Decode 为什么更像 memory-bound？

**参考回答：**

Prefill一次处理许多token，矩阵乘法维度较大，权重能被多个token复用，计算强度较高。Decode每请求每步只有一个token，GEMM退化为更窄的矩阵乘，且每步需要读取模型权重和不断增长的KV Cache，单位计算对应的数据搬运更多。

这只是典型规律。具体仍取决于batch、上下文长度、模型结构、量化、GPU和kernel实现，应结合Roofline和Profiler判断。

### Q47：GQA、MQA 和 MHA 有什么区别？

**参考回答：**

- MHA：Query、Key、Value通常拥有相同数量的heads。
- MQA：所有Query heads共享一组K/V heads。
- GQA：若干Query heads共享一组K/V heads，处于二者之间。

GQA/MQA显著减少KV Cache体积和Decode读取量，但会改变Query head到KV head的映射。计算KV显存时必须使用 `num_key_value_heads`，不能使用 `num_attention_heads`。

### Q48：本项目 Attention 的 Prefill 和 Decode 分别调用什么？

**参考回答：**

- Prefill：`flash_attn_varlen_func`，支持 packed变长序列；Prefix Cache命中时传分页K/V和Block Table。
- Decode：`flash_attn_with_kvcache`，Query增加长度维后读取分页KV Cache。
- 两条路径之前都通过 `store_kvcache()` 把当前K/V写入物理槽位。

源码：`nanovllm/layers/attention.py::Attention.forward`。

---

## 6. Triton、CUDA 与算子优化

### Q49：解释 `store_kvcache_kernel`

**参考回答：**

grid大小为本轮token数 `N`，每个Triton program处理一个token。它从 `slot_mapping[idx]` 得到该token在分页KV池中的扁平物理槽位；然后把该token所有本地KV heads和head dimension展平为 `D`，从输入K/V连续读取，再写入 `slot × D` 开始的Cache位置。

`slot == -1` 用来屏蔽CUDA Graph padding lane。当前内核结构简单，是否需要优化必须测量其执行时间和占比。

### Q50：这个 KV 写入内核可以怎样继续优化？

**参考回答：**

先用Profiler确认它是否是瓶颈，再考虑：

- 根据D选择block size和`num_warps`；
- 对非2次幂或边界加入mask；
- 使用向量化、保证对齐和合并访问；
- 融合KV量化/scale计算；
- 评估与RoPE或其他布局变换融合；
- 避免寄存器压力过高；
- 对不同head_dim、KV heads和dtype做autotune。

融合不是越多越好：可能增加寄存器生命周期、降低occupancy，并让代码难以复用。

### Q51：什么是 Memory Coalescing？

**参考回答：**

同一warp中的线程访问相邻、对齐的全局内存地址时，硬件可用较少内存事务完成访问。若线程跨较大stride或地址分散，会产生更多事务、浪费带宽。分析时要先明确“哪个线程访问哪个地址”，不能只看数组在逻辑上是否连续。

### Q52：什么是 Shared Memory Bank Conflict？

**参考回答：**

Shared Memory分成多个bank。同一warp中多个线程在一次指令访问同一bank的不同地址时，访问可能串行化；访问同一地址的广播通常是特殊情况。常见解决办法是改变布局、加入padding或转置访问模式。

### Q53：为什么调整 `BLOCK_M/BLOCK_N/BLOCK_K` 会改变 GEMM 性能？

**参考回答：**

tile决定：

- A/B从HBM载入后能复用多少次；
- 每个program产生多少工作，kernel数量和调度开销；
- Shared Memory和寄存器占用；
- 可同时驻留多少blocks/warps，即occupancy；
- Tensor Core利用率和边界mask比例；
- 访存是否连续、对齐。

tile太小，数据复用差、调度开销高；太大，可能寄存器溢出、Shared Memory超限、occupancy下降。正确方法是在目标GPU和真实shape上autotune，并结合Nsight Compute指标解释结果。

### Q54：M、N、K分别很小时，GEMM怎么优化？

**参考回答：**

小矩阵通常难以填满GPU，kernel launch和框架调度开销占比更高。可以：

- 将多个独立小GEMM做batched/grouped GEMM；
- 融合相邻的bias、activation、residual等操作；
- 为小shape选择较小tile和合适warp数；
- 减少padding和中间Tensor；
- 若数据规模过小，评估CPU或其他执行路径；
- 对固定shape使用专门kernel或CUDA Graph。

具体还要分别讨论：K小意味着归约短；M/N小意味着输出并行度不足。不能用一个tile覆盖所有shape。

### Q55：为什么 block size 常见 128 或 256 threads？

**参考回答：**

它们是warp大小32的整数倍，通常能提供多个warps帮助隐藏延迟，同时不给单block占用过多资源。但这只是起始经验，不是定律。最终选择取决于寄存器、Shared Memory、每SM最大blocks/warps和真实benchmark；更高occupancy也不一定更快。

### Q56：怎样判断一个 kernel 是 compute-bound 还是 memory-bound？

**参考回答：**

先看算术强度，即每搬运一字节做多少计算，再结合目标GPU的峰值算力和带宽做Roofline判断；然后用Nsight Compute观察：

- DRAM/L2吞吐是否接近峰值；
- SM、Tensor Core利用率；
- 指令和内存stall；
- occupancy、register spilling；
- kernel launch和执行时长。

仅凭“这是矩阵乘”或“这是Decode”不能断言瓶颈。

### Q57：Argmax kernel怎么写？

**参考回答：**

每个program负责一行或一个切片：分块加载值和索引，对越界位置mask为负无穷；先在线程私有/向量范围内求局部最大值及索引，再进行warp/block reduction，最终写出全局索引。

必须定义：

- tie时返回第一个还是任意位置；
- NaN如何处理；
- 输入是否连续；
- 行长度是否超过单program能力；
- 索引类型和超长维度；
- 与 `torch.argmax` 的正确性对比。

### Q58：Argmax还能怎样优化？

**参考回答：**

- 将Softmax后采样或其他前后处理融合，避免中间写回；
- 针对小维度使用单warp；
- 针对大维度使用两阶段归约；
- 调整每线程处理元素数、warps和block size；
- 使用向量化加载并确保合并访问；
- 减少同步和共享内存；
- 若只需要最大logit，不要先计算完整Softmax。

最后一条对greedy decoding尤其重要；本项目当前不是greedy，而是温度采样，不能直接删除Softmax等价步骤。

### Q59：写一个最朴素 CUDA Vector Add

**参考答案：**

```cpp
__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int threads = 256;
int blocks = (n + threads - 1) / threads;
vector_add<<<blocks, threads>>>(a, b, c, n);
```

**继续追问：**

- `i < n`防止最后一个不完整block越界；
- 不需要Shared Memory，因为每个输入只读一次，没有block内复用；
- 相邻线程访问相邻float，便于合并访问；
- 正确计时需要warmup、CUDA event和同步；
- 需要检查launch和异步执行错误。

### Q60：写一个朴素 GEMM，再说如何优化

**第一版参考答案：**

```cpp
__global__ void naive_gemm(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

**优化路径：**

1. A/B按tile加载到Shared Memory；
2. block内复用并减少HBM读取；
3. 保证global load合并和对齐；
4. register tiling，让每线程计算多个输出；
5. double buffering/异步拷贝隐藏访存；
6. 使用Tensor Core和适配的数据布局；
7. 处理非整除边界；
8. 针对shape autotune。

---

## 7. 量化与部署链路

### Q61：完整的大模型部署链路是什么？

**参考回答：**

```text
模型与Tokenizer准备
→ 权重格式/量化与正确性检查
→ 引擎加载模型
→ GPU/TP进程与通信初始化
→ warmup、显存profiling、KV Cache分配
→ CUDA Graph或kernel准备
→ 启动服务接口
→ 请求tokenize、排队、调度
→ Prefill、Decode、流式返回
→ 指标、日志、限流、超时和错误处理
→ 压测、SLO验证和容量规划
```

如果面试官问“你部署过vLLM吗”，应给出真实模型、GPU、dtype/量化方式、关键参数、请求协议、压测方法和遇到的问题；没有实际部署就不要虚构。

### Q62：vLLM服务常见启动参数有哪些类别？

**参考回答：**

不用死背所有参数，但要按类别组织：

- 模型/tokenizer、dtype、最大上下文；
- Tensor/Pipeline/Data Parallel；
- GPU memory utilization、KV Cache dtype、block/cache配置；
- max_num_seqs、max_num_batched_tokens、调度策略；
- Prefix Cache、Chunked Prefill、CUDA Graph/eager；
- 量化方式；
- API host/port、并发、鉴权等服务参数。

并说明版本变化很快，正式使用应查对应版本文档，不凭记忆复制参数。

### Q63：量化为什么能加速推理？

**参考回答：**

量化可减少权重和/或激活的存储体积与内存带宽，并在硬件支持时使用更高吞吐的低精度计算。但端到端加速取决于是否存在高效kernel、量化/反量化开销、shape、batch和瓶颈位置。仅仅把权重文件变小，不保证延迟同比下降。

### Q64：对称 INT8 量化的 scale 怎么算？

**参考回答：**

对signed INT8，常用范围约为 `[-127,127]`。对称量化可取：

```text
scale = max(abs(x)) / 127
q = clamp(round(x / scale), -127, 127)
x_hat = q * scale
```

按tensor简单但粗；按channel/group能适应不同分布，通常误差更小，但需要更多scale和更复杂kernel。校准数据要代表真实激活分布。

### Q65：PTQ 和 QAT 有什么区别？

**参考回答：**

- PTQ：模型训练完成后，用校准和量化算法直接转换，成本低但低bit精度更难保持。
- QAT：训练/微调期间模拟量化误差，让权重适应低精度，成本更高但通常更有精度潜力。

不能把GPTQ错误地叫成QAT：GPTQ是使用近似二阶信息的一次性权重量化方法，属于PTQ。

### Q66：AWQ 的核心思想是什么？

**参考回答：**

AWQ是低bit weight-only PTQ。它通过激活统计识别重要权重通道；不是简单保留混合精度，而是对显著通道做等价缩放，使权重量化误差降低，同时保持硬件友好的低bit执行。论文强调保护很少比例的显著权重即可显著降低误差。

### Q67：AWQ、GPTQ、SmoothQuant怎么区分？

**参考回答：**

- **AWQ**：activation-aware的低bit weight-only PTQ，通过激活统计和等价缩放保护显著通道。
- **GPTQ**：使用近似二阶信息，逐层/逐块进行一次性权重量化和误差补偿。
- **SmoothQuant**：把激活离群值的量化难度通过等价变换迁移到权重，主要目标是硬件友好的W8A8 PTQ。

回答时同时说明bit数、是否量化激活、是否需要校准/重构、kernel支持和目标硬件。

### Q68：为什么 INT4 权重不一定带来 4 倍速度提升？

**参考回答：**

- FP16到INT4是权重存储理论缩小4倍，不等于端到端计算4倍；
- 可能需要解包和反量化；
- kernel或Tensor Core路径未充分优化；
- Attention、KV Cache、采样和CPU调度未被加速；
- 小batch可能受launch、访存或其他部分限制；
- 量化group scale等元数据也占空间和带宽。

### Q69：怎样验证量化结果？

**参考回答：**

同时做正确性、质量和性能验证：

- 权重加载、层输出和异常值检查；
- PPL、任务集准确率或业务评测；
- 长上下文和不同prompt分布；
- TTFT、TPOT、吞吐、显存；
- 与同模型FP16/BF16基线对比；
- 固定版本、参数、随机性和输出长度；
- 检查量化kernel是否真正生效，而不是回退到高精度路径。

---

## 8. Tensor Parallel 与多卡推理

### Q70：什么是 Tensor Parallel？

**参考回答：**

TP把同一层的大矩阵沿输出或输入维度切到多张卡上，每张卡计算部分结果，再通过通信得到完整逻辑结果。它能分摊模型权重和部分中间状态，但每层可能产生集合通信，因此速度取决于计算/通信比和互联带宽。

### Q71：Column Parallel 和 Row Parallel 有什么区别？

**参考回答：**

- Column Parallel按权重输出维切分，每个rank得到一部分输出，若下一层能继续使用分片，当前不必立即通信。
- Row Parallel按输入维切分，每个rank得到部分和，通常需要All-Reduce求完整输出。

本项目QKV和MLP gate/up使用Column Parallel；Attention输出投影和MLP down投影使用Row Parallel并执行All-Reduce。

### Q72：一层 Qwen3 大致有哪些 TP 通信？

**参考回答：**

主干中通常至少有：

1. Attention `o_proj` 的Row Parallel输出All-Reduce；
2. MLP `down_proj` 的Row Parallel输出All-Reduce。

Embedding在词表分片后通过All-Reduce组合；最终LM Head在rank 0通过gather拼完整vocab logits。具体通信次数还要以代码、是否融合通信和模型结构为准。

### Q73：本项目多进程怎样协同？

**参考回答：**

rank 0在主进程中执行；其他rank由spawn创建。NCCL负责GPU集合通信。rank 0把方法名和参数pickle后写入Shared Memory，再用Event通知worker；worker读取后调用同名方法。模型权重按TP rank加载分片。

这是一种精简控制面，不等同于生产RPC或多节点容错方案。

### Q74：TP 为什么不一定线性加速？

**参考回答：**

- 每层All-Reduce/Gather增加通信；
- 小模型或小batch计算量不足，通信占比高；
- PCIe/NVLink/NVSwitch带宽差异；
- 切分后每卡GEMM变小，GPU利用率下降；
- load imbalance、同步和Python控制开销；
- KV Cache和Attention布局可能限制扩展。

TP首先解决“模型装不下”，性能收益需要实测。

### Q75：TP 和 PP 怎么选？

**参考回答：**

- TP切单层矩阵，层内通信频繁，但单请求可以多卡共同计算。
- PP按连续层切stage，层间传activation，通信频率较低，但需要micro-batch降低pipeline bubble。

单机高速互联常优先TP；跨节点或层本身不能方便整除时可考虑PP；大模型常组合TP×PP。选择取决于模型大小、并发、延迟目标和互联拓扑。

### Q76：如果完全没做过多卡，面试时怎么答？

**参考回答：**

> 我目前没有完成真实多卡性能实验，所以不会给出加速结论。但我读过本项目的QKV/MLP切分和通信路径：Column Parallel保留分片输出，Row Parallel通过All-Reduce合并；我能根据模型是否装得下、每层通信量和互联带宽分析TP/PP选择。后续需要在真实多卡环境中验证通信占比和扩展效率。

这比只说“不会”更好，也没有伪造实践。

---

## 9. 性能指标、Benchmark 与问题定位

### Q77：TTFT、TPOT、ITL、E2E分别是什么？

**参考回答：**

- TTFT：请求发送到收到第一个输出token；包含排队和Prefill等时间。
- ITL：相邻流式输出之间的间隔样本。
- TPOT：通常按请求计算，`(E2E - TTFT) / (output_tokens - 1)`。
- E2E：请求开始到完整输出结束。

不同工具口径可能不同，比较时要写清测量点和公式，而不是只比较同名指标。

### Q78：吞吐高是否代表用户体验好？

**参考回答：**

不代表。极端增大batch可能提高tokens/s，却让排队、TTFT或TPOT/P99恶化。在线服务应在SLO约束下看Goodput，即满足延迟目标的有效请求/令牌吞吐，而不是只追求总吞吐。

### Q79：当前 `bench.py` 有哪些局限？

**参考回答：**

- 一次性生成随机token IDs，是offline吞吐测试；
- 只统计总输出tokens/总时间；
- 没有TTFT、TPOT、ITL、P99和请求时间线；
- 没有动态请求到达；
- 没有多轮重复、置信区间或结果归档；
- 没有显式验证输出一致性和失败请求；
- README结果依赖特定4070 Laptop和模型，不能直接外推到其他环境。

它适合smoke/粗吞吐对比，不足以证明调度优化或生产SLO改善。

当前分支因此新增 `benchmarks/online_scheduler/`：它支持动态到达、first-token barrier 干扰负载，并保存逐请求、逐 token、逐 step 数据来复算 TTFT、TPOT、ITL、E2E 和 Goodput。但在线 GPU 原始 run 尚未归档，不能只凭 benchmark 代码存在就声称优化成功。

### Q80：怎样正确给 CUDA kernel 计时？

**参考回答：**

- 先warmup，排除编译和首次初始化；
- 使用CUDA Event记录GPU时间；
- 在读取结果前同步；
- 重复多次并报告中位数/分位数；
- 固定输入shape、dtype、布局和设备；
- 与正确参考实现对比；
- 避免把Host数据准备和kernel时间混淆，除非目标就是端到端；
- 检查异步错误。

### Q81：优化项目应该怎样描述结果？

**参考回答模板：**

> 在固定的模型、GPU、dtype、输入/输出长度分布、并发和软件版本下，先warmup并重复N轮。相对基线，某指标从A变为B；同时另一个指标从C变为D，显存/正确性结果为E。原始结果和命令均已归档。该结论只适用于上述环境，不外推到其他GPU和模型。

如果没有GPU正式实验，只能写“实现/CPU smoke/正确性测试”，不能写GPU加速百分比。

### Q82：遇到性能下降，你会怎样定位？

**参考回答：**

1. 复现并固定输入、版本和环境；
2. 检查正确性、实际batch/token数量和回退路径；
3. 分解排队、Prefill、Decode、采样、通信和Host开销；
4. 用Profiler定位时间占比和瓶颈；
5. 提出单一假设；
6. 一次只改变一个变量；
7. 重复测量并分析收益、代价和适用范围。

---

## 10. 手写题与伪代码

### Q83：用伪代码实现单请求、单Decode token的 PagedAttention

**题目约定：**

- `q`形状为 `[num_q_heads, head_dim]`；
- `k_cache/v_cache`形状为 `[num_blocks, block_size, num_kv_heads, head_dim]`；
- `block_table`给出逻辑块到物理块映射；
- `context_len`包含当前可见的KV token数；
- 支持GQA，并使用数值稳定online softmax。

**参考伪代码：**

```python
def paged_attention_decode(q, k_cache, v_cache,
                           block_table, context_len,
                           block_size, scale):
    num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    assert num_q_heads % num_kv_heads == 0
    group_size = num_q_heads // num_kv_heads

    out = zeros_like(q, dtype=float32)

    for q_head in range(num_q_heads):
        kv_head = q_head // group_size
        running_max = -inf
        running_sum = 0.0
        running_acc = zeros(head_dim, dtype=float32)

        for token_pos in range(context_len):
            logical_block = token_pos // block_size
            block_offset = token_pos % block_size
            physical_block = block_table[logical_block]

            k = k_cache[physical_block, block_offset, kv_head].float()
            v = v_cache[physical_block, block_offset, kv_head].float()
            score = dot(q[q_head].float(), k) * scale

            new_max = max(running_max, score)
            old_scale = exp(running_max - new_max)
            new_weight = exp(score - new_max)

            running_acc = running_acc * old_scale + new_weight * v
            running_sum = running_sum * old_scale + new_weight
            running_max = new_max

        out[q_head] = running_acc / running_sum

    return out.to(q.dtype)
```

**继续追问：**

- 实际GPU实现不会逐token串行，而会按KV tiles并行，再归约局部max/sum/acc；
- 最后一块只处理 `token_pos < context_len`；
- Block Table访问必须防止越界和无效块；
- 可与先gather成连续KV后的PyTorch Attention对比输出；
- FP16/BF16输入通常使用FP32累加提高稳定性。

### Q84：实现 BlockManager 的 allocate/free 伪代码

**参考伪代码：**

```python
def allocate(seq, required_blocks):
    if len(free_blocks) < required_blocks:
        return False
    for _ in range(required_blocks):
        block_id = free_blocks.pop_left()
        blocks[block_id].ref_count = 1
        used_blocks.add(block_id)
        seq.block_table.append(block_id)
    return True

def share_prefix(seq, cached_block_ids):
    for block_id in cached_block_ids:
        if block_id not in used_blocks:
            free_blocks.remove(block_id)
            used_blocks.add(block_id)
        blocks[block_id].ref_count += 1
        seq.block_table.append(block_id)

def free(seq):
    for block_id in reversed(seq.block_table):
        blocks[block_id].ref_count -= 1
        if blocks[block_id].ref_count == 0:
            used_blocks.remove(block_id)
            free_blocks.append(block_id)
    seq.block_table.clear()
```

**继续追问：**真实实现还要处理Prefix hash、空闲但可缓存块、重复释放、异常安全、并发访问和一致性断言。

### Q85：手写数值稳定 Softmax

**参考答案：**

```python
def stable_softmax(x):
    m = max(x)
    exps = [exp(v - m) for v in x]
    denom = sum(exps)
    return [v / denom for v in exps]
```

减去最大值防止指数上溢。GPU并行版需要先做max reduction，再做exp/sum reduction；FlashAttention则在分块遍历时使用online softmax。

### Q86：怎样验证手写 GEMM？

**参考回答：**

- 覆盖M/N/K不整除tile、1、小矩阵和大矩阵；
- 与cuBLAS/PyTorch参考对比；
- 根据dtype使用合理 `atol/rtol`；
- 检查NaN/Inf和越界；
- 分离正确性与性能测试；
- 性能比较必须包含warmup、CUDA Event和多次重复；
- 不期望教学kernel普遍击败cuBLAS，只分析特定shape。

---

## 11. 高频追问题与一句话回答

### Q87：PagedAttention 的 “利用率” 是什么利用率？

是KV Cache显存分配利用率：减少为最大长度预留和连续分配造成的浪费，并支持非连续物理块复用；不是GPU算力利用率，也不是Attention复杂度降低。

### Q88：Prefix Cache 能优化 Decode 吗？

主要节省共享前缀的Prefill计算；正常Decode仍需读取已有KV并生成新token。它可能缩短TTFT，但不会自动加速后续每个Decode step。

### Q89：KV Cache量化和权重量化有什么区别？

权重量化减少模型权重存储/读取并可能使用低精度GEMM；KV Cache量化减少随上下文和并发增长的K/V存储与读取。二者作用对象、误差传播和kernel路径不同。

### Q90：Continuous Batching一定降低单请求延迟吗？

不一定。它通常提高整体吞吐和资源利用率，但更大的动态batch可能增加单step时间；需要在TTFT、TPOT和吞吐之间调节。

### Q91：Chunked Prefill一定提高TTFT吗？

不一定。单个长prompt被拆成多轮可能增加其完成Prefill的时间；主要价值是限制长Prefill阻塞Decode并改善整体调度公平性。

### Q92：CUDA Graph 能处理任意动态shape吗？

图内地址和执行结构需要稳定。本项目为多个Decode batch bucket分别捕获，并用padding适配实际batch；任意shape会导致大量图或回退eager。

### Q93：高 occupancy 一定更快吗？

不一定。足够occupancy有助于隐藏延迟，但追求更高occupancy可能减少每线程寄存器、导致spill，或牺牲tile复用。必须结合瓶颈和实测。

### Q94：算子融合一定更快吗？

不一定。融合减少launch和中间HBM读写，但可能增加寄存器/Shared Memory压力、降低occupancy，并放大编译和维护成本。

### Q95：为什么不能直接用CPU计时包住异步CUDA调用？

CUDA kernel launch通常异步返回，CPU计时可能只测到launch开销。需要CUDA Event或在计时边界同步；端到端计时则要明确包含哪些Host工作。

### Q96：为什么不能只报告平均延迟？

在线服务的排队、长prompt、抢占和资源竞争常体现在尾部。平均值会掩盖少量严重慢请求，因此至少报告P50/P95/P99和失败率。

### Q97：为什么不能宣称“和 vLLM 一样快”？

性能依赖模型、GPU、版本、参数和负载。README中的单次环境结果不能外推；需要在当前代码和固定实验协议下复现，并同时验证输出、延迟、吞吐和显存。

---

## 12. 面试中的典型错误与修正

| 错误表达 | 正确表达 |
|---|---|
| Page Attention | PagedAttention |
| PagedAttention把计算从 `O(n²)` 降到 `O(n)` | 它优化KV内存管理，不改变标准Attention渐进计算量 |
| FlashAttention是近似Attention | 它是IO-aware的精确Attention算法 |
| KV Cache减少Attention对历史KV的读取 | 它避免重算历史K/V，但Decode仍要读取历史KV |
| Nano-vLLM完全没有Continuous Batching | 上游已有迭代级动态调度核心；当前分支已补在线接入与流式闭环，但不是生产级 serving |
| 原项目Scheduler/Prefix Cache是我实现的 | 明确上游能力，只描述已验证的个人改动 |
| Prefill永远compute-bound，Decode永远memory-bound | 这是常见规律，仍依赖shape、batch、模型、dtype和硬件 |
| INT4一定比FP16快4倍 | 存储理论缩小不等于端到端加速，取决于kernel和瓶颈 |
| block越大occupancy越高 | occupancy还受寄存器、Shared Memory和硬件上限影响 |
| 吞吐越高方案越好 | 在线服务要同时看TTFT、TPOT/P99、Goodput和正确性 |

---

## 13. 可以反问面试官的问题

优先选择能判断岗位真实工作内容的问题：

1. 团队当前主要做推理引擎的调度、KV Cache、模型适配，还是CUDA/国产芯片算子？
2. 实习生进入后第一个月通常负责什么类型的任务？
3. 优化目标更偏吞吐、TTFT/TPOT、成本，还是特定模型的端到端性能？
4. 目前主要使用或改造vLLM、SGLang、TensorRT-LLM，还是自研框架？
5. 性能优化如何做正确性、回归和线上流量验证？
6. 单机多卡、跨节点、PD分离和投机解码分别处于什么阶段？
7. 对实习生的CUDA要求是能读/改Triton kernel，还是需要直接写CUDA/CUTLASS？
8. 如果我入职前重点补一项能力，您最建议是调度、KV Cache、量化还是算子？

不要优先问官网已有的信息，也不要一开始只问转正、加班和福利。

---

## 14. 自测评分表

每题按0～3分记录：

- 0分：完全不会；
- 1分：只会名词定义；
- 2分：能解释原理，但落不到源码或边界；
- 3分：结论清楚，能落源码、举例、说验证和trade-off。

| 模块 | 题号 | 满分 | 建议通过线 |
|---|---|---:|---:|
| 项目真实性与架构 | Q1～Q16 | 48 | 36 |
| KV Cache/PagedAttention | Q17～Q30 | 42 | 34 |
| 调度与在线服务 | Q31～Q40 | 30 | 23 |
| Attention | Q41～Q48 | 24 | 18 |
| CUDA与算子 | Q49～Q60 | 36 | 24 |
| 量化 | Q61～Q69 | 27 | 18 |
| 多卡 | Q70～Q76 | 21 | 12 |
| Benchmark | Q77～Q82 | 18 | 14 |
| 手写与快问快答 | Q83～Q97 | 45 | 30 |

### 当前最高优先级

1. 能在5分钟内画出并讲通完整调用链；
2. 能手算KV显存、分页地址和跨块边界；
3. 能解释三种策略、bounded 公平边界和单 CUDA worker；
4. 能写PagedAttention伪代码和朴素CUDA kernel；
5. 能解释FlashAttention、GEMM tiling和性能验证；
6. 再扩展AWQ和TP通信。

---

## 15. 主要资料

### 当前项目源码

- `nanovllm/engine/llm_engine.py`：请求入口和主循环；
- `nanovllm/engine/async_llm_engine.py`：单 worker、命令通道和每请求异步流；
- `nanovllm/engine/scheduler.py`：Prefill/Decode调度、Chunked Prefill、抢占；
- `nanovllm/engine/sequence.py`：请求状态；
- `nanovllm/engine/block_manager.py`：分页分配、引用计数、Prefix Cache；
- `nanovllm/engine/model_runner.py`：输入准备、KV Cache、CUDA Graph、TP worker；
- `nanovllm/layers/attention.py`：KV写入和FlashAttention；
- `nanovllm/layers/linear.py`：Column/Row/QKV Parallel；
- `nanovllm/models/qwen3.py`：Qwen3 Attention和MLP结构；
- `nanovllm/serve/sse.py`：在线生成、取消、健康和指标端点；
- `benchmarks/online_scheduler/`：动态负载、Runner、指标和原始产物；
- `bench.py`：上游 offline 吞吐基线；
- `docs/baseline-rtx4060.md`：已验证 RTX 4060 离线基线；
- `docs/online_scheduler/`：在线调度架构、API、实验与复现材料。

### 论文和官方文档

- [PagedAttention / vLLM论文](https://arxiv.org/abs/2309.06180)
- [FlashAttention论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2论文](https://arxiv.org/abs/2307.08691)
- [vLLM Chunked Prefill与调优](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [vLLM Benchmark指标定义](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [vLLM分布式推理](https://docs.vllm.ai/en/v0.5.2/serving/distributed_serving.html)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [AWQ论文](https://arxiv.org/abs/2306.00978)
- [GPTQ论文](https://arxiv.org/abs/2210.17323)
- [SmoothQuant论文](https://arxiv.org/abs/2211.10438)

### 面试题型参考（只用于观察题型，不作为技术事实来源）

- [百度 AI Infra 一面复盘](https://www.nowcoder.com/discuss/875003802187792384)
- 用户提供的百度大模型训推框架优化一面记录；
- 用户提供的清程极智推理优化日常实习一面记录。

---

## 16. 每次模拟面试后的复盘模板

```markdown
# 模拟面试复盘 YYYY-MM-DD

## 总分

- 项目真实性：__/30
- KV Cache/PagedAttention：__/25
- 调度与性能：__/20
- CUDA/算子：__/15
- 表达与手写：__/10

## 三个回答最好的问题

1.
2.
3.

## 三个被追问击穿的问题

1. 问题：
   - 我的原回答：
   - 缺失点：
   - 修正回答：

2.
3.

## 源码补读

- 文件/方法：
- 要验证的状态变化：

## 手写题

- 是否一次写对：
- 边界错误：
- 正确性测试：

## 下一次模拟前的验收标准

- [ ] 90秒自我介绍不超时
- [ ] 5分钟讲完调用链
- [ ] 独立写出PagedAttention伪代码
- [ ] 独立写出Vector Add和Naive GEMM
- [ ] 所有个人项目表述都有代码/测试/实验依据
```
