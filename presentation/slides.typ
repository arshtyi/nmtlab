#set document(title: "英译中神经机器翻译与可视化分析系统", author: "NMT Lab")
#set page(
    width: 1120pt,
    height: 630pt,
    margin: (x: 46pt, top: 36pt, bottom: 42pt),
    fill: white,
    footer: context [
        #set text(size: 18pt, fill: rgb("64748b"))
        #grid(
            columns: (1fr, auto),
            [NMT Lab · 英汉神经机器翻译], [#context counter(page).display("1 / 1", both: true)],
        )
    ],
)
#set text(font: ("Lato", "Noto Sans CJK SC"), size: 22pt, fill: rgb("253449"), lang: "zh")
#set par(leading: 0.45em, spacing: 0.55em)
#set list(indent: 1em, body-indent: 0.45em, spacing: 9pt)
#let teal = rgb("137d83")
#let muted = rgb("64748b")
#let summary = json("../results/summary.json")
#let cases = json("../results/cases.json")
#let attention = json("../results/figures/attention.json")
#let cfg = summary.metadata.config
#let greedy = summary.overall.find(row => row.decoding == "greedy")
#let beam = summary.overall.find(row => row.decoding == "beam")
#let runtime = summary.metadata.runtime
#let number(value, places: 2) = $#str(calc.round(value, digits: places))$
#let heading(kicker, title) = {
    text(size: 18pt, fill: teal, weight: "bold", kicker)
    v(8pt)
    text(size: 34pt, weight: "bold", title)
    v(16pt)
}
#let label(body) = text(fill: teal, weight: "bold", body)

#heading("01 / 课程问题", "英译中神经机器翻译与可视化分析系统")
#text(size: 27pt, fill: muted)[从循环状态到注意力：固定模型，比较解码策略]
#v(30pt)
#grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 32pt,
    [
        #label[RNN]
        #v(12pt)
        逐步更新隐藏状态，序列信息经递归路径传递。
    ],
    [
        #label[LSTM]
        #v(12pt)
        门控与记忆单元缓解长程依赖，仍沿时间步递归。
    ],
    [
        #label[Transformer]
        #v(12pt)
        以注意力连接上下文；编码器—解码器不依赖循环状态。
    ],
)
#v(34pt)
#text(size: 26pt, weight: "bold")[问题：扩大搜索宽度，能否稳定改善翻译？代价多大？]
#v(16pt)
课程联系：序列建模、长程依赖、条件概率与搜索。采用 Transformer，而非 LSTM 的改型。
#v(20pt)
#text(size: 18pt, fill: muted)[算法依据：#cite(<transformer>, style: "apa").]

#pagebreak()
#heading("02 / 系统实现", "一套可操作、可复现的本地实验系统")
#grid(
    columns: (1.45fr, 1fr),
    gutter: 30pt,
    [
        #image("../results/figures/compare.png", width: 100%, height: 275pt, fit: "contain")
        #v(8pt)
        #text(size: 18pt, fill: muted)[同一输入并列比较 Greedy 与 Beam $= #beam.num_beams$。]
    ],
    [
        #label[复用成熟能力]
        #v(8pt)
        MADLAD-400 3B 预训练权重 #cite(<madlad>, style: "apa")、Transformers 推理、Gradio 界面组件。
        #v(12pt)
        #label[本项目实现]
        #v(8pt)
        - 固定配置与离线推理入口
        - 配对测量、指标汇总与案例选择
        - 生成序列回放与注意力可视化
    ],
)
#v(10pt)
#text(size: 18pt, fill: muted)[#cfg.model.id · 固定 devtest $200$ 句比较 Greedy 与 Beam $= #beam.num_beams$。]

#pagebreak()
#heading("03 / 固定实验", "翻译质量与推理延迟：同模型、同样本")
#text(size: 21pt)[FLORES-200 devtest · 英 $arrow.r$ 简中 · 固定 $200$ 句 · batch $= 1$ · #cfg.model.dtype]
#v(8pt)
#text(
    size: 20pt,
)[设备：#runtime.device / #summary.metadata.machine；计算线程 $#runtime.threads$，互操作线程 $#runtime.interop_threads$。]
#v(12pt)
#table(
    columns: (1.2fr, 1fr, 1fr, 1.5fr, 1.4fr),
    align: (left, right, right, right, right),
    inset: (x: 13pt, y: 9pt),
    stroke: none,
    table.hline(stroke: 1.2pt + teal),
    [*解码*], [*BLEU $arrow.t$*], [*chrF $arrow.t$*], [*均值延迟 / ms*], [*$P_(95)$ / ms*],
    table.hline(stroke: 0.6pt + rgb("cbd5e1")),
    [Greedy],
    [#number(greedy.bleu)],
    [#number(greedy.chrf)],
    [#number(greedy.mean_ms, places: 1)],
    [#number(greedy.p95_ms, places: 1)],
    [Beam $= #beam.num_beams$],
    [#number(beam.bleu)],
    [#number(beam.chrf)],
    [#number(beam.mean_ms, places: 1)],
    [#number(beam.p95_ms, places: 1)],
    table.hline(stroke: 1.2pt + teal),
)
#v(10pt)
#text(
    size: 23pt,
    fill: teal,
    weight: "bold",
)[$"Beam" - "Greedy"$：BLEU #number(beam.bleu - greedy.bleu)，chrF #number(beam.chrf - greedy.chrf)；延迟 #number(beam.mean_ms / greedy.mean_ms) 倍。]
#v(10pt)
#text(
    size: 19pt,
)[每模式预热 $2$ 次；每句测 $3$ 次并交替先后顺序。先取逐句中位数，再计算表中均值与 $P_(95)$。计时含分词、传输、生成、解码，不含加载与注意力回放。]
#v(10pt)
#text(
    size: 18pt,
    fill: muted,
)[无采样；生成上限 $256$ token；length penalty $= 1$；early stopping=false。BLEU：zh；chrF：$beta = 2$。#cite(<sacrebleu>, style: "apa")；#cite(<madlad>, style: "apa").]

#pagebreak()
#heading("04 / 注意力观察", "Cross-attention：观察生成时的信息分配")
#grid(
    columns: (1.5fr, 1fr),
    [
        #image("../results/figures/attention.svg", height: 350pt, fit: "contain")
        #text(
            size: 18pt,
            fill: muted,
        )[独立演示 `EXAMPLES[0]` · Beam $= #attention.translation.beams$ · 第 $#attention.view.layer$ 层 · 所有头算术均值]
    ],
    [
        #label[逐 token 对齐回放]
        #v(10pt)
        输入生成序列去掉末项；第 $t$ 行标注其预测的下一 token，去掉 decoder start。
        #v(14pt)
        #label[保留观察边界]
        #v(10pt)
        原始 SentencePiece 子词；隐藏源端语言控制符与 EOS 后不重新归一化。
        #v(14pt)
        权重体现模型内的关联，不能直接视为词对齐、语义正确性或因果解释。
    ],
)

#pagebreak()
#heading("05 / 案例与结论", "从可重复的数字，回到具体译文")
#let example = cases.find(row => row.category == "largest_negative_delta")
#text(size: 19pt, fill: muted)[案例 #example.id · 预定规则选择：$"Beam" - "Greedy"$ 的句级 chrF 最低负差]
#v(14pt)
#grid(
    columns: (110pt, 1fr),
    column-gutter: 16pt,
    row-gutter: 12pt,
    label[原文], text(size: 20pt, example.source),
    label[参考], text(size: 20pt, example.reference),
    label[Greedy], text(size: 20pt, example.greedy.text),
    label[Beam $= #beam.num_beams$], text(size: 20pt, example.beam.text),
)
#v(18pt)
#text(
    size: 21pt,
    fill: teal,
)[该句 chrF：#number(example.greedy.chrf) $arrow.r$ #number(example.beam.chrf)。搜索宽度增加不保证每句改善。]
#v(10pt)
#text(
    size: 20pt,
)[Beam 把“核心”改为“重点”，句级 chrF 下降；两种译文都未解决 makes people tick 的习语误译。负向极值体现逐句差异，并不代表整体质量变化。]

#pagebreak()
#bibliography("../report/references.bib", style: "apa")
