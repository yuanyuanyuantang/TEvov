# TEvov

从 BFCL 工具执行轨迹中提炼经验，并研究如何用当前任务的输入、工具返回和状态取得参数，避免直接复制历史实例值。执行与提炼使用 Qwen3-8B；评测时记忆冻结只读。

## 当前状态（2026-10-20）

[核心方法对齐](outputs/core_alignment_20261020/结果与下一步.md)已完成代码接入与CPU召回核查：保留123条普通经验，其中15条追加训练证据检查后的参数指导、108条不变；不增加记忆库或重采400条轨迹。新增显式任务工具过滤，旧配置默认行为不变。251项测试及9个子测试通过。6题两内容臂的训练小试已准备，因GPU1/2被其他任务占用尚未执行；这些检查不能证明方法有效。

已有[正式1200次结果](outputs/bfcl_pass4_confirm_20261019/结果汇总.md)：普通记忆＋首轮检索的Avg@4/Pass@4为38.50%/55.33%，123条普通候选＋逐轮检索为39.67%/54.00%，差值区间均跨零。该比较未注入独立编译的参数模板，不代表新追加指导版本已有成绩，也没有本批ReMe对照。

新候选库见[bank](outputs/core_alignment_20261020/compilation_reviewed/bank/)。配置`bfcl_source_aligned_control_20261020.json`与`bfcl_source_aligned_variant_20261020.json`是已检查的正式评测入口，均采用当前轮检索与任务工具过滤，仅正文不同；本轮不自动启动该150题×4次网格。小试计划另见`outputs/core_alignment_20261020/development_plans.json`，使用既有TRAIN题，不能冒充独立验证。

## 既往开发记录（2026-10-18）

本轮[自主修正与开发验证已收束](outputs/autonomous_research_20261018/09_本轮结果与收束.md)：123条候选库＋当前轮检索，在固定6题开发网格上整题1/6→2/6、首次合格启动0/6→2/6、前置错误8→5；token约+10.65%。12条官方重判、120次请求链核验通过；GPU1服务已关闭。这里只是训练小样本初步改善，未证明泛化、ReMe优势或ACL贡献；旧库与400轨迹保留，未追加正式大规模评测。

最新[十二次经验价值检查](outputs/experience_upper_bound_20261018/结果与收束.md)已完成并停止：手工训练指导使首次启动0/6→6/6、前置错误8→0，整题成功2/6→1/6，未过预设整体收益门。117次请求提示哈希与12条官方重判核验通过；这不是自动方法或验证集成绩。GPU1服务已关闭，原库和方法未改，不扩样。

本轮重新授权的有限重构已建库：[重构结果](outputs/memory_rebuild_20261018/修改与建库结果.md)。新库仍136条，19条保留全部步骤并补充条件取值指导、117条不变。新接口仅让模型选经验/关系，证据与渲染归程序；本次复用旧28响应中的选择，未调用新prompt。代理训练审查已完成，215项测试+5子测试通过，不代表方法已有收益。

最新两臂评价1200次已完成：普通经验Avg@4/Pass@4为38.33%/55.33%，补充取值指导为39.83%/56.00%，差值区间均跨零，未证明稳定收益。结果见[配对结果](outputs/bfcl_pass4_rebuild_20261018.results.json)。GPU服务已按要求停止。

动态更新方案的CPU阶段已完成：[阶段A结果与停止说明](outputs/local_memory_update_20261018_checked/阶段A结果与停止说明.md)。616个有限范围事件可判覆盖36.04%，未发现足以停用现有片段的反例，相对简单重编译没有观察增量；按预设门槛停止，未接入在线写库、未启动新模型实验。228项测试及5个子测试通过，现有银行和轨迹未改。

后续[失败诊断与修复](outputs/failure_bottleneck_20261018/诊断与修复结论.md)完成：修复评分转换丢失空用户轮的问题，144条受影响保存输出离线重判后分数均未变；全量233项测试及5个子测试通过。生产评分辅助源码已有新版本，旧manifest保持原样；旧运行需在隔离副本使用对应快照复现。本轮无新模型执行、建库或GPU启动。

[发动机链单项核查](outputs/engine_chain_audit_20261018/结果与收束.md)已结束：锁门/脚刹指导已有；“完全踩下”的补充草案未过三道训练题自然召回门，未发布、未改库或启GPU。本轮只新增1次本地CPU embedding（三输入），未调用LLM。

[首轮子目标召回有限尝试](outputs/subgoal_retrieval_20261018/结果与收束.md)已停止：只新增未启用的切分/MMR纯函数原型。三道训练题仍未同时召回燃油、锁门、脚刹指导，未过预设离线门；1次本地CPU embedding（21片段）＋1次health GET，0 LLM/actor/建库/GPU启动。236项测试及5子测试通过，1631项输入未变，默认运行器未改。

[工具感知召回修复](outputs/tool_aware_retrieval_20261018/修改与停止说明.md)完成：增加可用工具过滤与按标签重叠加权的冗余惩罚，旧默认不变。固定三题离线门1题通过、2题未过，按协议停止，未接runner、未启GPU或新actor；本轮0模型/embedding/网络请求，239项测试及5子测试通过。

[参数指导最终去留分析](outputs/parameter_guidance_effect_20261018/最终去留结论.md)已完成：已有1200条执行复核，当前自然曝光下未显示一致首次取值增量；ID类38对37、加满油4对4、城市47对48（含不可判边界，不能混算总准确率）。旧两条绑定偏离涉及保险对象分支，已保守纠正分析标签，原成绩不改。按约定停止当前版本升级；本轮0模型/embedding/环境执行，未改方法或库。

**当前尚未证明参数指导能稳定提高整体成绩，也没有本批公平 ReMe 对照。** 工程可运行、草稿通过检查和方法有效是不同判断。

完整的版本、成绩、保留入口与主张边界见[项目交付说明](outputs/finite_repair_20261018/cpu_closure/项目交付说明.md)。[最新收尾结果](outputs/finite_repair_20261018/cpu_closure/收尾结果.md)记录了 212 项测试及 5 个子测试通过、28 份响应的零模型回放。

## 从哪里看起

| 路径 | 用途 |
| --- | --- |
| `tevo/` | 模型客户端、实验运行器、记忆存储与证据检查 |
| `benchmark/bfcl/` | BFCL 任务、工具、答案及固定版本执行代码 |
| `scripts/` | 采集、建库、运行、重判与检查入口 |
| `configs/experiments/` | 实验配置；旧配置供追溯，不等于推荐重跑 |
| `tests/` | 独立样例上的代码检查，不是训练题数量 |
| `third_party/` | 第三方源码锁与许可证 |
| `outputs/` | 训练轨迹、已冻结银行、原始响应、结果和诊断材料 |

最简单的记忆参考是 **136 条普通经验**，用于理解和维护现有流程，不是按分数选出的最终方法：

- 银行：[ordinary_bank](outputs/pass4_20261015_preparation/ordinary_bank/)。
- 配置：[bfcl_pass4_ordinary.json](configs/experiments/bfcl_pass4_ordinary.json)。
- 已有结果：[普通经验结果](outputs/bfcl_pass4_ordinary_20261015/)。
- 本轮训练数据：[400 条无记忆轨迹](outputs/train400_20261014/trials/)，50 题×8 次。

早期 268 条编译库、254 条纠错库和两条正文改写版分别保留为历史实验版本。`finite_repair_20261018/generation/proposed_bank/` 与 `readable_drafts_not_for_actor.json` 是未发布草稿，不能当成新方法库。

## 安装与离线检查

Python 3.12。模型权重不随项目提供；服务配置见 `.env.example`，私有 `api` 文件不用于当前流程。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/run_experiment_suite.py --config configs/experiments/bfcl_pass4_ordinary.json --dry-run
```

测试与 `--dry-run` 不生成模型请求；dry-run 通过只说明配置和本地输入可读取，不证明服务可用。BFCL 固定源码缺失时，可运行 `python scripts/bootstrap_bfcl.py` 获取锁定版本；已有源码会被校验。

## 运行入口与边界

- 采集与建库：`scripts/build_memory_from_scratch.py`。现有 400 条轨迹继续保留，本轮不重采或重建。
- 冻结库评测：`scripts/run_experiment_suite.py`。只有显式 `--run` 才运行。
- 离线重判：`scripts/regrade_suite.py`。
- 有限改写：`scripts/rewrite_procedural_memory.py`，输出仍须人工训练内容复核，不自动准入 actor 评价。
- 局部更新CPU原型：`scripts/check_local_memory_update.py --output outputs/<全新目录>`，只读检查和修订建议，不修改正式库、不自动启动后续阶段。

新运行需先准备对应模型与 embedding 服务，核对配置，并指定全新 `--output-dir`。历史输出不要覆盖；当前源码后续有修复，不保证满足旧运行 `--resume` 的指纹要求，应使用历史快照复现，不能修改 manifest 绕过检查。

Avg@4 是每题四次分数的平均，Pass@4 是四次中至少完整成功一次的题目比例。四次使用固定 seed、独立会话，运行中不更新记忆；固定 seed 不保证服务端输出逐位确定。历史结果的顺序运行与并发条件有差别，不能拼成一次独立确认性比较。

当前不启用 ReMe 对照、工具画像、双库或在线写库。收束说明没有授权后台自动提炼或扩大实验；原始材料和各版本继续保留。
