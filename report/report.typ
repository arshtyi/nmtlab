#let summary = json("../results/summary.json")
#let cases = json("../results/cases.json")
#let attention = json("../results/figures/attention.json")
#let meta = summary.metadata
#let cfg = meta.config
#let greedy = summary.overall.find(row => row.decoding == "greedy")
#let beam = summary.overall.find(row => row.decoding == "beam")
#let ink = rgb("173740")
#let muted = rgb("5c6970")
#let accent = rgb("176b70")
#let pale = rgb("eef4f4")
#let fmt(value, digits: 2) = if value == none { sym.dash.em } else { $#str(calc.round(value, digits: digits))$ }
#let mode(value) = if value == "greedy" { [Greedy] } else { [Beam $#beam.num_beams$] }
#let group(value) = if value == "<=15" { [$n <= 15$ 词] } else if value == "16-30" { [$15 < n <= 30$ 词] } else {
    [$n > 30$ 词]
}
#let label(value) = (
    highest_average_chrf: "平均句级 chrF 最高",
    lowest_average_chrf: "平均句级 chrF 最低",
    largest_positive_delta: "Beam 增益最大",
    largest_negative_delta: "Beam 降幅最大",
).at(value)
#let note(body) = block(width: 100%, fill: pale, inset: 10pt, radius: 3pt, body)
#let idline(title, value) = block(width: 100%, above: 7pt, below: 7pt)[
    #grid(
        columns: (1fr,),
        row-gutter: 3pt,
        text(weight: "bold", title),
        text(font: "Lato", size: 8.5pt, value),
    )
]
#let panel(title, body) = block(width: 100%, stroke: 0.5pt + rgb("c6d5d7"), inset: 11pt, radius: 3pt)[
    #text(font: ("Lato", "Noto Sans CJK SC"), weight: "bold", fill: accent)[#title]
    #v(4pt)
    #body
]
#set document(title: "英译中神经机器翻译与可视化分析系统", author: "彭靖轩")
#set page(
    paper: "a4",
    margin: (top: 20mm, bottom: 18mm, left: 20mm, right: 20mm),
    header: context [
        #set text(font: ("Lato", "Noto Sans CJK SC"), size: 8pt, fill: muted)
        NMT LAB #h(1fr) 神经机器翻译课程实践
        #v(3pt)
        #line(length: 100%, stroke: 0.4pt + rgb("cbd6d8"))
    ],
    footer: context [
        #set text(font: ("Lato", "Noto Sans CJK SC"), size: 8pt, fill: muted)
        固定模型 · 固定样本 · 可追溯结果 #h(1fr) #counter(page).display("1")
    ],
)
#set text(font: ("Lato", "Noto Serif CJK SC"), size: 10.5pt, fill: ink, lang: "zh")
#set par(justify: true, leading: 0.62em, spacing: 0.8em)
#set heading(numbering: "1.1")
#show heading: set text(font: ("Lato", "Noto Sans CJK SC"), fill: ink)
#show heading.where(level: 1): set text(size: 18pt)
#show heading.where(level: 2): set text(size: 12pt)
#set table(inset: (x: 6pt, y: 6pt), stroke: 0.4pt + rgb("d1dcde"), fill: (x, y) => if y == 0 { pale } else { none })
#show table.cell.where(y: 0): set text(weight: "bold", font: ("Lato", "Noto Sans CJK SC"))

#text(font: ("Lato", "Noto Sans CJK SC"), weight: "bold", size: 24pt)[英译中神经机器翻译与可视化分析系统]
#v(5pt)
#text(font: ("Lato", "Noto Sans CJK SC"), size: 15pt, fill: muted)[解码比较、可复现实验与交叉注意力观察]
#parbreak()
#text(font: ("Lato", "Noto Sans CJK SC"), size: 10pt, fill: muted)[彭靖轩 · 202400130242 · 24智能]
#v(15pt)

= 研究背景与任务定位

== 摘要

本项目构建了可在本地运行的英译汉系统，以预训练 MADLAD-400 3B 的 T5 编码器—解码器 Transformer 为推理核心，提供文本翻译、Greedy 与 Beam Search 对比、交叉注意力观察及固定测试集评估。项目未进行模型训练或微调；工作重点是把课程中的序列建模、条件概率与搜索方法落实到可检查的推理流程，并明确界面展示与正式实验之间的边界。

正式实验采用 FLORES-200 devtest 的 $200$ 个固定英汉句对，统一模型版本、计算设备、线程数、精度和生成参数。质量使用 SacreBLEU 的语料级 BLEU 与 chrF；每句每种解码重复计时 $3$ 次，先取句内中位数，再汇总整体与分长度组时延。

#note[
    *研究问题：* 同一模型与输入下，保留更多候选的 Beam Search 会怎样改变译文、参考匹配分数及本地推理时延？交叉注意力可以展示哪些内部权重，又不能证明什么？
]

== 背景与课程联系

机器翻译要求模型在保留语义的同时改变词序、表达习惯与文字系统。英汉之间的分词、实体写法和省略方式存在差异，因此“看起来通顺”和“忠实表达原文”不能由一个参考匹配指标完全替代。系统需要把译文、参考、耗时以及模型内部可观察量并列呈现，支持逐项检查。

课程中的 RNN 与 LSTM 给出了沿时间递推状态的序列建模视角；现代 Transformer 则以注意力和位置表示组织上下文。本项目通过固定预训练模型，隔离训练因素，观察推理阶段的搜索策略及其代价。

= 从循环模型到非循环注意力

== RNN、LSTM 与条件生成

RNN 通过 $h_t = f(h_(t-1), x_t)$ 顺序更新隐藏状态，较早输入的信息需要沿递推链传播。LSTM @lstm 在这一循环结构中加入输入门、遗忘门和输出门，其记忆单元可写为
$ c_t = f_t dot.o c_(t-1) + i_t dot.o tilde(c)_t, quad h_t = o_t dot.o tanh(c_t). $
其中 $dot.o$ 表示逐元素乘法。门控机制有助于保留长距离信息、缓解梯度传播困难，但仍存在按时间步递推的依赖。早期 Seq2Seq @seq2seq 可用循环编码器与解码器建模源句和目标句，注意力进一步让解码器按需读取源端状态。

本项目模型采用 *非循环 Transformer* @transformer，并非对 LSTM 的改造。编码器用自注意力聚合源句信息；解码器用因果自注意力读取目标前缀，再以交叉注意力读取编码器输出。位置表示提供序列次序。训练阶段的目标位置可并行计算，但自回归推理仍须逐个生成目标 token。

对于源序列 $x$ 与目标序列 $y$，共同的条件生成目标为
$ P(y | x) = product_(t=1)^T P(y_t | y_(<t), x). $
该分解连接了循环 Seq2Seq 与 Transformer 的任务形式；两者组织上下文的网络机制不同。

== Greedy 与 Beam Search

Greedy 在每一步选择当前概率最大的 token：
$ hat(y)_t = arg max_v P(v | hat(y)_(<t), x). $
其优势是候选维护开销小，但局部最优选择无法保证整句条件概率最大。Beam Search 在每步扩展候选前缀，保留有限个高分假设；正式实验束宽为 $#beam.num_beams$，并使用长度惩罚。完成候选的排序可概括为
$ s(y) = (sum_(t=1)^T log P(y_t | y_(<t), x)) / T^alpha, quad alpha = 1. $
这里 $T$ 表示库用于长度归一化的生成长度，EOS 处理及停止细节由已保存的生成配置和依赖版本共同确定。束搜索仍是近似搜索；更高的模型分数不等于更忠实或更高的参考匹配分数。

两种模式都关闭采样，最多生成 $256$ 个新 token。束搜索的 `early_stopping=false` 使用启发式停止判据，并不要求每句生成到上限，也不是穷举全体可能序列。Beam 维护多个候选，增加计算和内存开销。

== 指标观察的含义

BLEU 主要衡量候选与参考的 $n$-gram 匹配，并含长度惩罚；本实验使用 SacreBLEU @sacrebleu 的中文分词配置。chrF @chrf 以字符 $n$-gram 的精确率和召回率构造 F 分数，适合从另一角度观察中文表面匹配。

= 系统架构与完成工作

== 复用与项目贡献的边界

预训练权重 @madlad-model 与分词模型来自 MADLAD-400 @madlad；PyTorch 与 Transformers 执行网络推理及标准生成算法，SacreBLEU 计算指标，Gradio 和 Matplotlib 提供交互与绘图基础。

项目贡献在于明确并实现实验协议、连接可追溯的推理与展示链路、处理输入和 token 索引边界，以及根据真实结果进行有限范围的误差分析。

= 实现正确性与运行边界

== 交叉注意力的计算与索引

对解码器第 $l$ 层、第 $h$ 个头，查询来自目标侧状态，键和值来自编码器输出。本项目 T5 @t5 模型的交叉注意力在有效源位置上计算为
$ A^((l,h)) = op("softmax")_("source")(Q^((l,h)) (K^((l,h)))^top), quad Z^((l,h)) = A^((l,h)) V^((l,h)). $
T5 不显式除以 $sqrt(d_k)$，此处也没有学习的相对位置偏置；源端填充位置由掩码排除。项目直接读取模型重放输出的权重。矩阵 $A$ 的行是预测目标位置，列是源 token 位置；保留全部源位置时，每行权重和约为 $1$。保存张量顺序为“层 $times$ 头 $times$ 目标长度 $times$ 源长度”，界面层与头采用一基编号，头编号 $0$ 表示所有头的算术平均。

设生成结果为 $[y_0, y_1, dots, y_T]$，其中 $y_0$ 为 decoder start。重放时输入 `output_ids[:-1]`，即 $[y_0, dots, y_(T-1)]$；第 $t-1$ 行表示在前缀条件下预测 $y_t$ 时的交叉注意力，故标签使用 `output_ids[1:]`。若把行标签错写为 decoder 输入，会产生一位偏移。重放实际选中的完整生成序列，不混用束搜索中已淘汰候选的权重。

保留 SentencePiece 原始子词边界和位置，不把多个子词合并成看似整词的对齐。语言控制符、EOS 与 PAD 可在显示时隐藏；未知 token 仍保留。语言控制符用于指定翻译方向，其权重不能解释为词汇对齐。bfloat16 权重含舍入误差；绘图时转换为 float32，隐藏行列后不重新归一化，所以可见行的权重总和可以小于 $1$。

== 输入与输出的保护范围

输入先去除首尾空白；空输入被拒绝。MADLAD-400 的 T5 模型在源句前添加 `<2zh>` 指定中文方向，不额外强制生成目标语言 token；$512$ token 输入上限包含源端控制符与 EOS。

输出达到 $256$ 个新 token 时保守标记 `capped`，即使最后一个位置被强制写入 EOS，也提示可能不完整。

== 计时、快照与结果保护

计时从分词前开始，到生成与文本解码结束，包含设备传输；不包含加载、预热、注意力重放、绘图或浏览器渲染。Attention 页的图和译文绑定同一会话快照，修改输入不会把旧图伪装成新输入的解释。

= 固定实验协议与复现信息

== 数据、模型与环境固定

FLORES-200 @nllb @flores 的英文 `eng_Latn` 与简体中文 `zho_Hans` devtest 各含 $1012$ 行。用 Python `random.Random(42)` 从一基行号 $1 <= i <= 1012$ 无放回抽取 $200$ 个 ID，再排序写入清单；加载时重新校验抽样规则、双语对齐、行数和字节哈希。

准备阶段先核对整个压缩包 SHA-256，再读取指定的两个 devtest 普通文件并校验各自哈希，避免不受控解包。模型固定到 commit；推理通过 `local_files_only=True` 加载 tokenizer、模型与生成配置，缓存缺失时明确失败。

#idline("模型", cfg.model.id)
#idline("模型 revision", cfg.model.revision)
#idline("数据归档 SHA-256", cfg.data.sha256)
#idline("样本清单 SHA-256", meta.manifest_sha256)
#idline("依赖锁文件 SHA-256", meta.uv_lock_sha256)

== 成对测量与汇总方法

三个自行编写的 `EXAMPLES` 用于界面演示，首句用于预热；它们独立于正式 $200$ 条样本，不按得分挑选。正式测量前每种模式预热两次；对清单中每句，两模式各运行三次。“样本零基位置 $+$ 重复轮次”为偶数时 Greedy 先运行，否则 Beam 先运行。三次译文、token 与触顶标记必须一致，否则中止。

每句耗时先取三次中位数；表中的 mean、median、$P_(95)$ 再跨句汇总，$P_(95)$ 在排序索引 $(n-1) times 0.95$ 处线性插值。长度组按原始英文空白分隔词数 $n$ 划分为 $n <= 15$、$15 < n <= 30$、$n > 30$，不含语言前缀。

= 正式评估结果

== 整体质量与时延

#table(
    columns: (1.1fr, 0.65fr, 0.8fr, 0.8fr, 1fr, 1fr, 1fr, 0.7fr),
    [解码], [句数], [BLEU], [chrF], [Mean], [Median], [$P_(95)$], [触顶],
    ..summary
        .overall
        .map(row => (
            mode(row.decoding),
            fmt(row.n, digits: 0),
            fmt(row.bleu),
            fmt(row.chrf),
            fmt(row.mean_ms, digits: 1),
            fmt(row.median_ms, digits: 1),
            fmt(row.p95_ms, digits: 1),
            fmt(row.capped, digits: 0),
        ))
        .flatten(),
)

在本次固定样本上，Beam 相对 Greedy 的 BLEU 差为 #fmt(beam.bleu - greedy.bleu)，chrF 差为 #fmt(beam.chrf - greedy.chrf)；平均耗时比为 #fmt(beam.mean_ms / greedy.mean_ms) 倍。输出达到生成上限的句数分别为 $#greedy.capped$ 和 $#beam.capped$。

== 按源句长度分组

#table(
    columns: (1.05fr, 1fr, 0.55fr, 0.75fr, 0.75fr, 0.95fr, 0.95fr),
    [英文长度], [解码], [句数], [BLEU], [chrF], [Mean ms], [$P_(95)$ ms],
    ..summary
        .by_length
        .map(row => (
            group(row.length_group),
            mode(row.decoding),
            fmt(row.n, digits: 0),
            fmt(row.bleu),
            fmt(row.chrf),
            fmt(row.mean_ms, digits: 1),
            fmt(row.p95_ms, digits: 1),
        ))
        .flatten(),
)

三组源句长度中，Beam 的 BLEU 与 chrF 都高于 Greedy。该结果描述本次固定样本上的分组差异，不能证明源句长度本身造成质量变化；各组题材、句法、实体和参考措辞仍然不同。

== 指标配置与完整签名

BLEU 使用中文 tokenizer、大小写敏感、指数平滑且关闭 effective order。chrF 使用字符阶数 $6$、词阶数 $0$、$beta = 2$，大小写敏感、不计空白字符、不启用 epsilon 平滑。句级 chrF 仅用于案例筛选，不能与上表语料级分数混为同一统计量。

#text(font: "Lato", size: 8.5pt)[BLEU: #summary.signatures.bleu]
#parbreak()
#text(font: "Lato", size: 8.5pt)[chrF: #summary.signatures.chrf]

= 预定规则下的案例检查

案例在评估前规定选择规则：两模式平均句级 chrF 最高、最低各一例；再选 $"Beam" - "Greedy"$ 为正和为负时绝对变化最大的各一例，同分按样本 ID 排序。若不存在某个符号的变化则不创建该类别。类别描述参考匹配，不预先判定译文语义正确或错误。

#for (index, case) in cases.enumerate() {
    if index == 2 {
        pagebreak()
        text(font: ("Lato", "Noto Sans CJK SC"), size: 18pt, weight: "bold")[案例续：搜索变化与参考匹配]
        parbreak()
        [以下两例由带符号的句级 chrF 差值极值确定。正向变化不自动意味着同等幅度的语义改善；负向变化也应定位到具体信息与表达。]
        parbreak()
    }
    block(breakable: false)[
        #v(9pt)
        #text(
            font: ("Lato", "Noto Sans CJK SC"),
            size: 12pt,
            weight: "bold",
            fill: accent,
        )[#label(case.category) · ID #case.id]
        #parbreak()
        *原文：* #case.source
        #parbreak()
        *参考：* #case.reference
        #parbreak()
        *Greedy：* #case.greedy.text
        #parbreak()
        *Beam $#beam.num_beams$：* #case.beam.text
        #parbreak()
        #text(
            size: 9pt,
            fill: muted,
        )[句级 chrF：#fmt(case.greedy.chrf) $arrow.r$ #fmt(case.beam.chrf)；差值 #fmt(case.delta_chrf)。]
        #parbreak()
        #if case.id == 147 [
            *检查：* 两种译文相同，保留了执政党、SWAPO、议会选举和“多数”等主要信息，但没有明确译出参考中的“席位”。该例平均句级 chrF 最高，仍需检查具体表述是否充分。
        ] else if case.id == 795 [
            *检查：* 两种译文都把 action 和 music 直译为“行动”和“音乐”，没有表达“观看演出”和“靠近舞台”。Beam 将其中一处改为“入住”，句级 chrF 略升，但没有修复核心场景含义。
        ] else if case.id == 270 [
            *检查：* Greedy 遗漏了第二句关于驾车者的信息；Beam 补回“卡德巴克是司机”，并保留两个人名及括号信息，因此获得本组最大正向句级 chrF 变化。该例说明搜索可以恢复信息，但不证明每句都改善。
        ] else if case.id == 655 [
            *检查：* 两种译文都把习语 makes people tick 误译为“让人们勾选”。Beam 将“核心”改为“重点”，进一步降低与参考的字符匹配；这一差异不说明语义同比例退化，两种模式均未修复本例的习语错误。
        ]
    ]
}

= 交互界面与真实注意力

#figure(
    image("../results/figures/compare.png", fit: "contain"),
    caption: [本地 Compare 页面。这里为单次、Greedy 先运行的交互展示。],
)
#figure(
    image("../results/figures/attention.svg", fit: "contain"),
    caption: [实际生成序列重放的交叉注意力：第 $#attention.view.layer$ 层、全部头算术平均；隐藏特殊 token，不重新归一化。],
)

*图中源句：* #attention.translation.source
#parbreak()
*实际译文：* #attention.translation.text
#parbreak()
横轴为源端原始子词，纵轴为预测目标子词。源端的目标语言控制符已隐藏；平均多个头会掩盖头间差异，且开头的词边界 token 可能形成明显权重集中。

#pagebreak()

#show bibliography: set text(size: 9pt)
#show bibliography: set par(justify: false)
#bibliography("references.bib", title: [参考文献], style: "gb-7714-2015-numeric")
