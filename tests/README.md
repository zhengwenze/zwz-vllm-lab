# Tests

- `unit/`：调度器、引擎状态机、benchmark workload、指标和重复实验聚合测试。
- `contract/`：SSE 服务与 HTTP 接口契约测试。

所有测试均由仓库根目录执行：`pytest tests`。需要真实 GPU 的性能运行不放入正确性测试目录，而由 `scripts/` 启动并把证据写入 `artifacts/`。
