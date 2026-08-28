# Generated artifacts

`artifacts/` 保存本机生成的 benchmark 与环境验证证据，不存放源码。

## 目录约定

- `online_scheduler/<run_id>/`：在线调度 benchmark 的默认输出位置。
- `online_scheduler/experiments/<name>/`：已完成的多轮实验快照。
- `validation/<date>/`：GPU、依赖和启动链路的验证日志。

历史实验应归档到 `experiments/`，不要继续创建 `artifacts_<date>/` 或 `online_scheduler_<suffix>/` 形式的平行目录。除本说明外，生成产物均由 Git 忽略。
