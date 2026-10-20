# 自主研究本轮已收束（2026-10-18）

123条候选库＋当前轮检索在固定训练开发网格达到预设联合门：整题1/6→2/6、首次启动0/6→2/6、前置错误8→5，token约+10.65%。12条官方重判与120请求链核验通过。GPU1已释放；未改旧400轨迹/旧库，未启额外正式评价。候选冻结于`outputs/autonomous_research_20261018/cycle3/ordinary_bank/`，详见该根目录`09_本轮结果与收束.md`。这是开发信号，不能写成已证明泛化、优于ReMe或ACL新颖性成立。

以下为历史记录：

# 最新状态（2026-10-18）：十二次手工经验诊断完成，本轮收束、GPU1释放

最新[十二次经验价值检查](outputs/experience_upper_bound_20261018/结果与收束.md)已完成并停止：手工训练指导使首次启动0/6→6/6、前置错误8→0，整题成功2/6→1/6，未过预设整体收益门。117次请求提示哈希与12条官方重判核验通过；这不是自动方法或验证集成绩。GPU1服务已关闭，原库和方法未改，不扩样。

结果目录outputs/experience_upper_bound_20261018/，6训练题51/59/63/65/66/67，原top5不变＋483字定向指导，seed0/temp0.9，12次正常完成。117LLM/728477报告token＋117未知usage tokenize，准备1embedding（6输入）usage未知；首次启动配对6胜，整题1胜2负3平。仅1/6有门锁与满脚刹返回在首次启动前均可见，不能写严格等待链全部学会。source/task/fingerprint/117prompt/12regrade全过；没有改生产源码、银行或自动提炼器。一次服务CXXABI启动修复发生在0actor时，保留attempt1并复用准备不重嵌，GPU0/2未动。既定门未过，不自动追加方法或运行；局部帮助与整题收益分别报告。下文历史保留。

# 最新状态（2026-10-18）：参数指导已有结果最终分析闭合，当前版本停止升级

[参数指导最终去留分析](outputs/parameter_guidance_effect_20261018/最终去留结论.md)已完成：已有1200条执行复核，当前自然曝光下未显示一致首次取值增量；ID类38对37、加满油4对4、城市47对48（含不可判边界，不能混算总准确率）。旧两条绑定偏离涉及保险对象分支，已保守纠正分析标签，原成绩不改。按约定停止当前版本升级；本轮0模型/embedding/环境执行，未改方法或库。

完整结果outputs/parameter_guidance_effect_20261018/：历史1200网格与上下文/银行/原regrade核验；旧720局部事件复算，再保守修正保险对象分支40事件；新增城市128对象对事件。最终report零请求重算一致，1235输入SHA未变。19改写ID以source_record_id回链原锚（不要说两臂字面id全部相同）。源码/配置/冻结库/原成绩未改，仅新增分析与说明。没有确认值得再跑一轮的编译公式/字段错误，也不声称所有语义正确或方向被证伪。不默认继续召回/动态/重建库/模型试验。下文为历史。

# 最新状态（2026-10-18）：工具感知召回固定修复未过门，本轮结束

[工具感知召回修复](outputs/tool_aware_retrieval_20261018/修改与停止说明.md)完成：增加可用工具过滤与按标签重叠加权的冗余惩罚，旧默认不变。固定三题离线门1题通过、2题未过，按协议停止，未接runner、未启GPU或新actor；本轮0模型/embedding/网络请求，239项测试及5子测试通过。

本轮只修改tevo/experiment/subgoal_retrieval.py与对应测试，显式available_tools启用新策略；未更新正式默认召回。base65缺门锁/脚刹，base70过门，base75缺门锁；同标签粒度不一致与同域无关经验仍有局限。不继续调参/标签修补、建库或启动12次actor。旧1631输入中2项声明内源码/测试变动，1629项未变；旧源码快照和失败结果保留，不能覆盖旧manifest代码指纹。新协议/结果/零请求复核/源码前后差异在outputs/tool_aware_retrieval_20261018/。下文为历史状态。

# 最新状态（2026-10-18）：首轮子目标召回未过离线门，本轮结束

[首轮子目标召回有限尝试](outputs/subgoal_retrieval_20261018/结果与收束.md)已停止：只新增未启用的切分/MMR纯函数原型。三道训练题仍未同时召回燃油、锁门、脚刹指导，未过预设离线门；1次本地CPU embedding（21片段）＋1次health GET，0 LLM/actor/建库/GPU启动。236项测试及5子测试通过，1631项输入未变，默认运行器未改。

原型tevo/experiment/subgoal_retrieval.py；固定切分/每query top20/0.7-0.3 MMR/最多5条，无工具特例、不改库。三题base65/70/75：分别缺门锁/脚刹/门锁，base75出现无关订票经验。保存21向量和选择轨迹，零网络复放通过；不调参追命中、不集成suite、不启动12次训练actor。报告outputs/subgoal_retrieval_20261018/结果与收束.md，全部代码未提交。下文均为历史记录。

# 当前状态（2026-10-18）：评价完成，GPU已释放；动态更新CPU阶段未过门，已停止

最新单项核查已结束：outputs/engine_chain_audit_20261018/结果与收束.md。136条中锁门12条、脚刹7条已有（交集1），没有明确完全踩下指导；训练6次0.5→错误→1.0→running支持小补充，但三道固定训练首轮查询top5均不召回候选脚刹记录。按方案不建新库、不启GPU、不跑12次actor；仅留未发布草案。191个前置错误中178次门锁/脚刹指导未入窗、9次入窗仍错、4次未踩到底，不作因果分解。1624输入未变、783请求hash与41证据引用通过；本轮0 LLM/环境重放，1本地CPU embedding POST＋1health GET，usage未知。未改生产代码或重复跑全量测试，动态分支仍关闭。

最新后续：outputs/failure_bottleneck_20261018/诊断与修复结论.md。既有两臂1200输出离线诊断全部匹配后，发现并修复benchmark/bfcl/bfcl_utils.py丢空用户轮的问题；144条转换受影响输出额外重判，0条分数变化，Avg@4/Pass@4不变；全量233 passed+5 subtests。原型、银行、原始结果不改，GPU未重启。当前与原manifest源码差异为已声明的bfcl_utils.py，旧版/新版快照和差异均留档，不覆盖manifest或绕过指纹。新诊断是开发分析，不是新actor实验；动态分支仍不进入学习或大跑。

最新1200次两臂评价全部完成并官方重判一致。普通经验Avg@4/Pass@4=38.33%/55.33%，补充取值指导=39.83%/56.00%，配对区间跨零，尚无稳定整体增益。结果：outputs/bfcl_pass4_rebuild_20261018.results.json。GPU1/2的本项目服务已按用户要求停止，未重启。

用户授权动态方案后，只实现并执行阶段A：scripts/check_local_memory_update.py及13项新测试。400训练+600既有开发执行，616个限定事件222可判、394弃权，覆盖36.04%低于50%门槛；真实refuted=0、改库=0，相同初始6种关系在简单重编译后不变。原型没有接入runner，不能称在线更新已实现。按方案不进入24次学习/1800次评价，不自动扩范围。全量228 passed+5 subtests；本轮0模型/embedding/网络/环境检查，1014输入哈希未变。

最终报告outputs/local_memory_update_20261018_checked/阶段A结果与停止说明.md，机读report.json、verification.json；首版及一次状态保护修正记录保留在outputs/local_memory_update_20261018/。README/HANDOFF修改前快照在checked目录。本次未提交、不覆盖历史结果或记忆。下文所有运行中状态均为历史。

# 历史状态（2026-10-18）：新一轮有限重构已建库，唯一两臂评价已启动

本轮由用户重新明确授权，不覆盖上轮关闭结论。程序保留136普通经验全部原步骤，19条新增条件取值指导，117条完全不变；键/向量/顺序不变。新接口模型仅选record_id/option_ids；实际先用28旧响应的选择迁移，0新提炼/embedding。30草稿经当前代理训练审查接受19、拒绝11，8种关系；不声称真人复核或自动语义证明。215项CPU测试+5子测试通过，4261历史文件未变。

新冻结库outputs/memory_rebuild_20261018/released_bank/；过程库不可混用。报告同目录修改与建库结果.md，审查training_review_completed.json。自然入窗70/150题、127槽，未为曝光改选规则。唯一1200次原普通/新库两臂评价协议outputs/bfcl_pass4_rebuild_20261018.启动前协议.md，状态查同前缀.state.json。GPU1/2，3+3workers，seed0/temp0.9；复用检索快照，预期0embedding。完成自动重判与配对汇总，错误/漂移停止，无自动重试/扩展。

以下保留旧轮次，不应将旧零发布、旧停止或旧运行中当作本轮最新状态。

# 最新状态（2026-10-18）：零模型收尾完成，本轮改写路线关闭，无新发布库

仅修正共享渲染器把模型动作句强加为 If 条件的格式错误；原句以 Context 保留、参数指导单独呈现，prompt 与语义/引用门禁不变。212项CPU测试+5子测试通过，禁止网络下拦截计数0。28份已有响应/63提议只读回放：58条原拒绝原因不变，5条结构通过；2条可读草稿仅增加同类下单订单ID字段说明，3条仍有对象范围/步骤丢失问题。按实质增量门不发布、不新评测；不能把停止解释成方法已经证伪。

收尾产物：`outputs/finite_repair_20261018/cpu_closure/收尾结果.md`、`final_review.json`、`verification.json`、`completion.json`。`readable_drafts_not_for_actor.json`仅供审计。历史生成仍绑定旧源码，当前渲染源码是新版本；4141旧文件及95个上阶段产物SHA未变。新增模型/embedding/actor/网络均为0。本轮停止，不默认继续改prompt或扩大实验。

以下保留历史状态。

# 最新状态（2026-10-18）：有限修复完成CPU验收，一次真实提炼未过发布门，已停止

本轮修改仅在`scripts/rewrite_procedural_memory.py`和`tests/test_conditional_rewrite.py`，旧版/新版源码与diff存于`outputs/finite_repair_20261018/`。修改涵盖唯一匹配的多对象来源、bool/enum复制排除、完整证据边连接、新的原句/条件选择接口和待审草稿门。全量209项测试+5子测试通过；400条轨迹产生39种候选、494条观察支持、涉及43条原经验和28个任务。CPU候选不等于已发布规则或已证明来源。

按方案仅一次真实Qwen3-8B提炼（温度0/seed0/thinking关闭）：28请求、363877报告tokens、0embedding、0actor。63个提议中58个结构/引用回退，5个结构通过；训练材料审查5个全部拒绝，涉及条件与动作混合、If After/If Retrieve等渲染问题、原步骤弱化或主题/对象范围不当。原始响应完整保留，未再改prompt、猜引用、手改正文或重提炼。

当前没有新发布库，没有新1200次实验运行。`outputs/finite_repair_20261018/generation/proposed_bank/`仅为失败过程草稿，不得当成正式方法库；actor_admission=false。旧136普通库和全部旧结果未改，4141个不可变旧文件哈希复核通过。`training_review_completed.json`是最终审查记录，原生成模板/manifest不覆盖。按停止条件本分支已停止，不默认继续优化或重跑。

详见`outputs/finite_repair_20261018/修改与验收报告.md`及`completion.json`。必须区分CPU代码检查通过、生成达到发布标准、方法有效三个判断；这轮只完成第一项，渲染与语义条件仍有未解决限制。

上次两臂1200次已于2026-10-17 22:54完成并重判一致：control Avg@4=39.50%/Pass@4=55.33%，variant=41.33%/54.00%，未证明稳定收益；结果在`outputs/bfcl_pass4_conditional_20261017.results.json`。下方“运行中”均为历史状态。

# 最新状态（2026-10-17 12:17）：一次CPU回放完成，唯一1200次两臂评价已启动

用户授权的一次接口修复已完成，不改prompt、不重新提炼、不采集、不embedding。只读回放原27响应/73候选，恢复2条股票域返回绑定（symbol与current_time），134条普通经验逐字段不变；没有发布补油差值规则。193项CPU测试与5子测试通过，旧2895项输入/历史结果哈希未变，旧0发布结果原样保留。

新库`outputs/conditional_replay_20261017/`，136原身份一一对应，检索键/向量/相对顺序保持。自然曝光30/150题、30槽；其余120题注入正文相同；不强塞召回，不根据评测分数选规则。实际文本长度差锁在`outputs/conditional_replay_preparation_20261017/acceptance.json`。方法只获轨迹支持，不声称条件充分性或唯一来源已经验证。旧普通库4条人工训练排除身份继续披露。

12:17启动两套同期单臂评价：control原136普通经验，variant仅两条插入已核返回绑定。150题×4次×两臂=1200 trials，seed0/temperature0.9/Qwen3-8B，GPU1/2、每臂3workers总6，独立会话、只读库、固定同一检索快照，运行0embedding。启动后实测两个服务合计6个LLM请求执行、队列0，GPU0不动。

监督器PID/阶段见`outputs/bfcl_pass4_conditional_20261017.process.json/.state.json`；配置`configs/experiments/bfcl_pass4_conditional_control.json`及`bfcl_pass4_conditional_variant.json`；产物各自同名outputs目录。启动收据2930项哈希及协议见同前缀launch.json/启动前协议.md；完成后自动两套官方离线重判与Avg@4/Pass@4配对汇总，任一失败停止另一套件、不自动重试。

只做本次评价与验收，结束即收束，不追加prompt、种子、benchmark或ReMe；这仍是同验证集探索性，额外离线证据/计算及实际长度不能隔离，也不是已证明优于ReMe或达到ACL贡献。详细实现/失败分类/示例/成本见`outputs/conditional_replay_preparation_20261017/回放与启动结果.md`。本轮CPU阶段0请求，随后预检4次GET、启动观察2次metrics GET；actor成本单独记账。

以下保留上一状态为历史，不能把其中“未启动”当作当前状态。

# 最新开发状态（2026-10-17）：条件化改写已实现，停在提炼准入门槛

新增`scripts/rewrite_procedural_memory.py`及7项测试，全量190项+5子测试通过。复用400条训练轨迹，固定136条普通经验骨架，只改相关正文；不改执行器/检索/注入。35类候选（34返回绑定、1容量差值）供8B选择条件和操作。

本轮27次8B提炼、363053报告tokens、0embedding、0actor，73条候选没有通过本轮接口检查，发布改写0条、136条原文均保留。核心问题是模型未使用表达式占位，不能保证正文与所选关系绑定；这不等于已经证明该方法无效。没有为绕过检查人工补写或再次改prompt。

独立验收通过，旧2830个输入文件及2400次历史结果未改。按零改写停止条件，未启动1200次评价；本轮两个136条银行都是原普通银行的精确副本，不是新的有效方法库。详见`outputs/conditional_rewrite_preparation_20261017/实现与提炼结果.md`、`acceptance.json`及`completion.json`，原始输入/响应在`outputs/conditional_rewrite_20261017/`。源码快照已留档。

上一轮corrected600次已在2026-10-17 05:18完成：Avg@4=38.50%、Pass@4=53.33%，600/600官方重判一致；四臂汇总在`outputs/bfcl_pass4_corrected_20261017.四臂结果.json/.md`。当前没有新增actor评测运行，也不默认追分、改prompt、扩种子/模型/benchmark。

以下为旧状态原文，保留历史，不能把其中“运行中”当作现在状态。

# 当前项目状态（2026-10-17）：有限纠错已冻结，唯一600次复验运行中

## 本轮做了什么

共享参数来源检查增加目标返回准入，避免把明确报错或无法判断的调用作为正向规则证据；正文补充当前用户显式提供所需形式、当前目标对象匹配和等待响应的条件。它们不是通用适用性分类器，也不证明规则的语义正确或因果有效。

最新纠错库：`outputs/formal_bank_corrected_20261017/`，254条=136条普通经验+118条参数规则。普通经验所有字段不变；132条旧参数规则经逐支持过滤后删除14条无可用正向支持的记录。2270条支持中保留1870条，302条明确报错、98条unknown不再计入正向支持。完整旧元数据保留在私有source_record_metadata与原库；输入证据、检索键、保留向量及其相对索引顺序不变。保留原先4条人工训练复核排除的身份，不称整个流程完全自动。

没有重采400条，没有再次调用模型提炼，没有重新embedding建库。全量CPU回归183 passed、5 subtests；独立核验旧轨迹、银行、1800次旧结果共2223个文件哈希未变。脚本为`scripts/correct_parameter_bank.py`；原builder修改前快照保留在`outputs/parameter_admission_fix_20261017/build_memory_from_scratch.before.py`。历史清理/生成的重放应使用当时源码，不能用新版checker冒充历史版本。

## 正在运行及停止边界

`configs/experiments/bfcl_pass4_corrected.json`，150题×重复4次=600次；seed=0、temperature=0.9、总并发6、Qwen3-8B、GPU1/2（8010/8011），CPU embedding 8014。记忆冻结只读，4次独立会话不更新库。评价的150条query会重新embedding（5个计划批次）；与建库0embedding分开计量。固定seed不保证逐位可重复。

监督器PID与阶段见`outputs/bfcl_pass4_corrected_20261017.process.json/.state.json`，运行日志同前缀`.run.log`。运行后自动官方离线重判及四臂汇总，输出`.四臂结果.json/.md`和`.exit.json`；失败/来源变化会停止，无自动重试。锁定输入见同前缀`.launch.json`和`.启动前协议.md`。不要在运行中改核心代码、配置、银行、轨迹或旧结果。

原nomem/ordinary/compiled1800次不重跑，Avg@4分别37.33%/38.67%/41.50%，Pass@4分别52.67%/58.00%/57.33%。纠错版是看到这批结果后的探索性复验；支持筛选、正文和覆盖共同变化，不能只归因于某一项。只新增这一轮600次，完成验收即收束，不自动扩种子、改prompt、换模型或增加ReMe。涨分不是工程通过条件。

## 历史记录（2026-10-16；以下保留原文，不代表当前运行状态）

# 当前项目状态（2026-10-16）

本轮采集、建库清理冻结及两臂评测已完成：ordinary 38.67%/58.00%、compiled 41.50%/57.33%（Avg@4/Pass@4）。1200次全完成、0运行错误、官方离线重判1200/1200一致，结束于2026-10-16 01:45（北京时间）。原结果与只读诊断见outputs/bfcl_pass4_20261015同前缀文件。

当前按操作者新要求补跑nomem无记忆四次基线，新增150题×4次=600次；原两臂不重跑。配置`configs/experiments/bfcl_pass4_nomem.json`，同模型、seed=0、temperature=0.9、总并发6及端点映射，无记忆库/检索/注入/embedding。核心代码未改，旧两臂来源指纹仍匹配。此基线在看到两臂结果后追加，属于探索性对照。

后台监督进程与实时阶段见`outputs/bfcl_pass4_nomem_20261016.process.json`、`.state.json`；运行日志为同前缀.run.log。完成后自动regrade及三臂汇总，输出同前缀.三臂结果.json/.md和.exit.json。协议和锁定输入见同前缀.启动前协议.md及.launch.json；保持核心代码、配置、数据、记忆与原两臂结果不变。GPU仅使用1/2。

## 当前使用的记忆库

`outputs/formal_bank_clean_20261015/`：268条，普通经验136条、参数规则132条；普通经验覆盖50道训练题。`freeze.json`固定库、索引、清单与验收报告哈希。`independent_verification.json`确认保留记录和向量与生成库逐条相等、400条训练轨迹未变、存储模块可直接载入268条。详见同目录`清理与冻结说明.md`。

生成输入为`outputs/formal_bank_compact_20261015/`的274条（普通142、参数132）。本轮只删除6条，不改写、不合并、不重新embedding；4条为已有人工训练审查排除项，另2条由加固后的字面守卫排除。完整生成输入保留用于核查，不是另一套默认评测库。

普通经验仍是模型建议，程序只检查证据身份及字面值等；参数规则是训练轨迹内的受限拟合，不等于证明唯一真实来源或跨实例正确。本次含人工排除，必须披露，不称全自动流程。冻结不代表效果达标。

## 已有训练数据

`outputs/train400_20261014/trials/`：50题×8次，400条全新Qwen3-8B无记忆执行。170整题成功、230整题失败，全部保留；400/400离线重判一致。399条原版本加1条仅账本修复版本的补采，原故障及版本差异收据均保留。采集最终状态为`outputs/train400_20261014/summary_completed.json`。

普通提炼每题选择最低/最高分代表（并列按base_seed/episode_id），50次LLM请求；参数检查读取全部8条。新prompt生成阶段用时617.66秒，报告171433 tokens，账本含50 LLM、50 tokenize、9 embedding、1 provider；60条usage未知，不计零。本次CPU清理没有新增任何请求。

## 代码入口与检查

- `scripts/build_memory_from_scratch.py`：采集与生成；普通经验只输出when_to_use/content/evidence_refs。
- `scripts/clean_frozen_memory.py`：显式输入生成库、训练轨迹与审查记录，生成仅删除的子集并冻结；CLI禁止网络，拒绝覆盖或写入输入树。
- 参数编译范围仍限直接用户输入、此前已到达的标量返回；数组和任意算式不自动合成，多候选弃权。
- 字面守卫递归检查嵌套参数、匹配用户目标数字，保留合同枚举处理；不是通用语义或顺序验证器。
- 清理阶段174项CPU测试通过；新增固定seed重复执行及指标回归后，当前177项测试、5项subtests通过。数据文件数量与测试数量无关。

## 已装配的两臂评测

`configs/experiments/bfcl_pass4_ordinary.json`：136条普通经验；`bfcl_pass4_compiled.json`：268条完整记忆。普通经验臂不是ReMe；其库是当前冻结库的类型子集，位于`outputs/pass4_20261015_preparation/ordinary_bank/`，记录与向量未改写、未重新embedding。

两臂均为同150题×重复4次=每臂600次、合计1200次。`seeds=[0]`、`sampling_seed_mode=fixed`、`repeats_per_seed=4`，实际请求seed=0，不做按题派生；temperature=0.9。旧task_derived默认模式保留。四次使用独立会话和不同重复编号，记忆只读；跨臂端点映射已逐项CPU核对。完整四次网格才报告avg_at_4/pass_at_4。

两份dry-run通过，模拟真实actor请求测试确认seed/温度。每套件总并发已由4提高到6，初始约每GPU 3路，两臂按顺序执行、不同时启动叠加并发。8010在GPU 1、8011在GPU 2，CPU embedding仍为8014；并发预检4次只读本地GET，正式启动前另3次只读GET确认模型身份/上下文及CPU embedding状态。协议见`outputs/pass4_20261015_preparation/评测协议与启动说明.md`（启动前历史文件，保持不变），并发变更为同目录`concurrency_update.json`。

此前监督脚本`outputs/bfcl_pass4_20261015.supervisor.py`已完成两个600次套件、分别官方离线重判及独立指标重算，结果在同前缀.results.json及.结果汇总.md，.exit.json为exit0。逐阶段日志为同前缀.ordinary.log、.compiled.log和.regrade_*.log，保留原始含义。

## 历史与后续边界

旧133条成品库、索引和freeze已删除。`outputs/formal_bank_20261015/`仅保留旧prompt原始响应与运行账目；不恢复旧库、不复用为新prompt输出。其他用户目录未修改。

当前只追加操作者要求的600次无记忆基线，结束后收束为三臂对照，不默认增加其他运行。没有启动ReMe、重采或追加提炼。`configs/experiments/no_memory.json`仍只是无记忆示例；正式nomem使用bfcl_pass4_nomem.json。不默认增加模型、种子、benchmark、工具画像或在线更新；不调用DeepSeek。GPU仅1/2可用，GPU0不动。
