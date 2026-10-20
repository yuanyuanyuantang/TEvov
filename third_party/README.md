# BFCL 固定来源与补丁

官方来源：[ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)，固定提交
`ea13468e4423454d0c213704fb87cf7cb3990433`。保留其 Apache-2.0 LICENSE。
`bfcl.lock.json` 固定 167 个分发文件的 SHA256；不包含文件权限、Git 历史、模型或缓存。
另有一个旧 REST sanity 数据含不透明凭据样例，与 multi-turn-base 无关，记入
`excluded_files`：先验证其原始 SHA256 再丢弃，不在包中分发。

网络环境运行：

```bash
python scripts/bootstrap_bfcl.py
```

已有该提交的本地 Git 仓库可离线运行：

```bash
python scripts/bootstrap_bfcl.py --source-dir /path/to/gorilla
```

脚本读取固定提交的 Git 对象，忽略源目录的未提交修改，再应用已审阅的
`bfcl-local.patch`。已有目标只校验，不覆盖。官方 BFCL 的完整 pip 包包含许多
本实验不使用的闭源模型客户端，因此此处仅安装 `requirements-bfcl.txt`，运行器
通过固定源目录导入官方执行器、解析与评分代码。

## 本地补丁的范围

与历史正式实验的本地版本一致，共 5 个文件有内容变化，文件权限变化排除：

| 文件 | 变更与理由 |
| --- | --- |
| `eval_checker/ast_eval/ast_checker.py` | 延迟导入所有模型注册表，避免本地评测因无关客户端依赖失败。 |
| `eval_checker/eval_runner.py` | 同上，仅在真正选择完整官方 handler 时导入。 |
| `eval_checker/eval_runner_helper.py` | 同上，评分辅助函数不提前加载所有模型客户端。 |
| `eval_checker/multi_turn_eval/multi_turn_utils.py` | 明确执行实例命名空间，提供按模型/任务清理实例的函数；保留原有命名规则和执行语义。 |
| `model_handler/api_inference/qwen.py` | 仅在使用 Qwen-Agent handler 时导入 qwen_agent，标准 OpenAI 兼容评测不触发 dashscope 依赖。 |

多轮 `multi_turn_checker.py`、API 环境实现和判分准则保持该官方提交的内容。
执行器实例清理可以防止任务/trial 状态互相污染。调用前后的代码仍需在每个
trial 中使用独立命名空间；源固定不自动保证调用者正确使用。

官方执行器执行模型生成的 Python 调用表达式，这是 BFCL 的既有机制。此研究
流水线仅用于合成 benchmark 环境，不能作为连接真实文件系统/服务的部署代码。

## 复查和修改

`bfcl-local.patch` 是可审阅的文本差异，脚本对补丁及全部生成文件校验。需要改动
官方源码时，应在独立目录开发、审阅并重新产生锁；不要直接修改 materialized
`benchmark/bfcl/gorilla/`。重新运行 bootstrap 会发现字节不一致并拒绝继续。

补丁上下文的空行可能含上游空白字符，这是复现原始差异和锁定字节的一部分。
不要对 `bfcl-local.patch` 或 materialized 上游源码运行自动格式化/去尾空白；
原始补丁的空白不是本项目 Python 产品源码的格式缺陷。若确需改变补丁，须
重新审阅并更新补丁及生成文件hash，不能在保持旧锁的同时改写字节。
