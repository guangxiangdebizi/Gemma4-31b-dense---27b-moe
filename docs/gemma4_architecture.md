# Gemma 4 架构深度解析：31B Dense vs 26B-A4B MoE

> 本文详细对比 Google Gemma 4 两款中大型模型的架构设计，帮助理解稠密模型与混合专家模型的核心差异。

---

## 目录

1. [模型概览](#1-模型概览)
2. [共享基础架构](#2-共享基础架构)
3. [Dense 31B 架构详解](#3-dense-31b-架构详解)
4. [MoE 26B-A4B 架构详解](#4-moe-26b-a4b-架构详解)
5. [核心差异：FFN vs MoE Layer](#5-核心差异ffn-vs-moe-layer)
6. [Router 路由机制](#6-router-路由机制)
7. [注意力机制](#7-注意力机制)
8. [Per-Layer Embedding (PLE)](#8-per-layer-embedding-ple)
9. [视觉编码器](#9-视觉编码器)
10. [性能与效率对比](#10-性能与效率对比)
11. [微调注意事项](#11-微调注意事项)

---

## 1. 模型概览

| 特性 | Gemma 4 31B (Dense) | Gemma 4 26B-A4B (MoE) |
|:---|:---|:---|
| 架构类型 | 稠密 Transformer | 混合专家 (Mixture of Experts) |
| 总参数量 | 30.7B | 25.2B |
| **激活参数量** | **30.7B (100%)** | **3.8B (15%)** |
| 层数 | 60 | 30 |
| 隐藏维度 | 4096 | 2816 |
| 注意力 Q 头数 | 32 | 16 |
| KV 头数 | 16 | 8 |
| Head 维度 | 128 | 256 |
| 专家数 / 激活数 | — | 128 / Top-8 |
| 单个专家参数 | — | ~6M |
| MoE 中间维度 | — | 704 |
| 上下文长度 | 256K | 256K |
| 词表大小 | 262,144 | 262,144 |
| 视觉编码器 | SigLIP2 (~550M) | SigLIP2 (~550M) |
| 许可协议 | Apache 2.0 | Apache 2.0 |

```mermaid
graph LR
    subgraph "Gemma 4 模型家族"
        E2B["E2B<br/>2B 参数<br/>边缘设备"]
        E4B["E4B<br/>4B 参数<br/>边缘旗舰"]
        MoE["26B-A4B<br/>25.2B 总参/3.8B 激活<br/>效率之王"]
        Dense["31B Dense<br/>30.7B 全量<br/>性能天花板"]
    end
    
    E2B -.->|"参数递增"| E4B
    E4B -.->|"参数递增"| MoE
    MoE -.->|"参数递增"| Dense
    
    style MoE fill:#4CAF50,color:#fff
    style Dense fill:#2196F3,color:#fff
```

---

## 2. 共享基础架构

两个模型共享同一套基础设计，差异仅在 FFN 层的处理方式上。

```mermaid
graph TD
    Input["输入 Tokens"] --> Embed["Token Embedding + PLE"]
    Embed --> Block["Transformer Block × N"]
    Block --> Norm["Final RMSNorm"]
    Norm --> LMHead["LM Head → 词表概率分布"]
    
    subgraph "每个 Transformer Block 内部"
        direction TB
        RN1["RMSNorm"] --> Attn["Multi-Head Attention<br/>(全局注意力 / 滑动窗口交替)"]
        Attn --> Res1["残差连接 +"]
        Res1 --> RN2["RMSNorm"]
        RN2 --> FFN_OR_MOE["FFN (Dense)<br/>或<br/>MoE Layer (MoE)"]
        FFN_OR_MOE --> Res2["残差连接 +"]
    end
    
    style FFN_OR_MOE fill:#FF9800,color:#fff
```

### 共享技术栈

| 技术 | 说明 |
|:---|:---|
| **RoPE** | 旋转位置编码，支持长序列外推 |
| **RMSNorm** | 比 LayerNorm 更高效的归一化，省去均值计算 |
| **GeGLU** | FFN 激活函数，= GELU(xW₁) ⊙ (xW₂)，比 ReLU 效果更好 |
| **滑动窗口注意力** | 窗口大小 1024，与全局注意力交替使用，降低长序列计算量 |
| **GQA** | 分组查询注意力，多个 Q 头共享一组 KV 头，节省 KV Cache |

---

## 3. Dense 31B 架构详解

Dense（稠密）模型的核心特点：**每个 token 经过每一层时，所有参数都参与计算，没有跳过、没有选择。**

```mermaid
graph TD
    subgraph "31B Dense — 单层结构 (×60 层)"
        Input_h["输入隐藏状态 h"]
        Input_h --> Norm1["RMSNorm"]
        Norm1 --> Attn["Multi-Head Attention<br/>32 个 Q 头, 16 个 KV 头<br/>head_dim = 128"]
        Attn --> Add1["h = h + Attn(h)"]
        Add1 --> Norm2["RMSNorm"]
        Norm2 --> Gate["gate_proj: h → 中间维度"]
        Norm2 --> Up["up_proj: h → 中间维度"]
        Gate --> GELU["GELU 激活"]
        GELU --> Mul["⊙ 逐元素相乘"]
        Up --> Mul
        Mul --> Down["down_proj: 中间维度 → h"]
        Down --> Add2["h = h + FFN(h)"]
    end
    
    style Gate fill:#E91E63,color:#fff
    style Up fill:#E91E63,color:#fff
    style Down fill:#E91E63,color:#fff
```

### 数据流动（以单个 token 为例）

```
Token "你好"
  │
  ▼ Embedding: 映射到 4096 维向量
  │
  ▼ Layer 1:  Attention(全局) → FFN(全部 30.7B 参数参与)
  ▼ Layer 2:  Attention(滑动窗口) → FFN(全部参数参与)
  ▼ Layer 3:  Attention(全局) → FFN(全部参数参与)
  │  ...
  ▼ Layer 60: Attention → FFN
  │
  ▼ LM Head: 4096 维 → 262144 维 (词表概率)
  │
  ▼ 输出: 下一个 token 的概率分布
```

**关键数字**：
- 每个 token 要经过 **60 层**，每层都做完整的注意力 + FFN 计算
- FFN 中间维度约为隐藏维度的 4 倍 ≈ 16384
- 单次前向传播的 FLOPs ≈ **2 × 30.7B ≈ 61.4 GFLOPs/token**

---

## 4. MoE 26B-A4B 架构详解

MoE 模型的核心特点：**FFN 层被拆成多个"专家"，每个 token 只激活其中少数几个专家。**

```mermaid
graph TD
    subgraph "26B-A4B MoE — 单层结构 (×30 层)"
        Input_h["输入隐藏状态 h"]
        Input_h --> Norm1["RMSNorm"]
        Norm1 --> Attn["Multi-Head Attention<br/>16 个 Q 头, 8 个 KV 头<br/>head_dim = 256"]
        Attn --> Add1["h = h + Attn(h)"]
        Add1 --> Norm2["RMSNorm"]
        Norm2 --> Router["🔀 Router (路由器)<br/>线性层: 2816 → 128<br/>+ Softmax"]
        
        Router -->|"Top-1"| E1["Expert 1<br/>FFN 6M"]
        Router -->|"Top-2"| E2["Expert 2<br/>FFN 6M"]
        Router -->|"..."| E8["Expert 8<br/>FFN 6M"]
        Router -.->|"未选中"| E9["Expert 9~128<br/>💤 不参与计算"]
        
        E1 --> WeightSum["加权求和<br/>Σ wₖ·Eₖ(h), k∈Top-8"]
        E2 --> WeightSum
        E8 --> WeightSum
        WeightSum --> Add2["h = h + MoE(h)"]
    end
    
    style Router fill:#FF9800,color:#fff
    style E1 fill:#4CAF50,color:#fff
    style E2 fill:#4CAF50,color:#fff
    style E3 fill:#9E9E9E,color:#fff
    style E4 fill:#9E9E9E,color:#fff
    style EN fill:#9E9E9E,color:#fff
```

### 类比理解

```mermaid
graph LR
    subgraph "Dense = 全科医生"
        Patient1["患者"] --> Doctor["一个全科医生<br/>啥都会<br/>每次全力看诊"]
    end
    
    subgraph "MoE = 专科医院"
        Patient2["患者"] --> Triage["分诊台<br/>(Router)"]
        Triage -->|"骨折"| Ortho["骨科专家"]
        Triage -->|"骨折"| Surg["外科专家"]
        Triage -.->|"不需要"| Eye["眼科专家 💤"]
        Triage -.->|"不需要"| Derm["皮肤科专家 💤"]
        Triage -.->|"不需要"| Neuro["神经科专家 💤"]
    end
    
    style Triage fill:#FF9800,color:#fff
    style Ortho fill:#4CAF50,color:#fff
    style Surg fill:#4CAF50,color:#fff
    style Eye fill:#9E9E9E,color:#fff
    style Derm fill:#9E9E9E,color:#fff
    style Neuro fill:#9E9E9E,color:#fff
```

### 具体配置（来自 config.json）

| 参数 | 值 | 含义 |
|:---|:---|:---|
| `num_experts` | **128** | 每层有 128 个专家 |
| `top_k_experts` | **8** | 每个 token 只激活 8 个（6.25%） |
| `hidden_size` | 2816 | 隐藏维度 |
| `moe_intermediate_size` | 704 | 每个专家 FFN 的中间维度 |
| `num_attention_heads` | 16 | Q 头数 |
| `num_key_value_heads` | 8 | KV 头数 |
| `head_dim` | 256 | 每个注意力头的维度 |

### 单个专家的结构

每个专家就是一个微型 GeGLU FFN：

```
输入 h (2816 维)
    │
    ├──→ gate_proj (2816 → 704) ──→ GELU ──┐
    │                                       ⊙ 逐元素相乘
    └──→ up_proj   (2816 → 704) ──────────┘
                                 │
                           down_proj (704 → 2816)
                                 │
                           输出 (2816 维)

单个专家参数 = 3 × 2816 × 704 ≈ 5.95M（约 600 万参数）
```

### 每层的参数分布

```
每层专家总参数 = 128 × 5.95M ≈ 761M
每层激活参数   = 8 × 5.95M   ≈ 48M  ← 实际计算量
激活比例 = 48M / 761M = 6.25%
```

### 30 层的注意力类型分布

```
Layer  1-5:  滑动窗口 × 5  ┐
Layer  6:    全局注意力     ├─ 每 6 层一个周期
Layer  7-11: 滑动窗口 × 5  ┤
Layer 12:    全局注意力     ├─ 5:1 的比例
Layer 13-17: 滑动窗口 × 5  ┤  滑动窗口:全局 = 25:5
Layer 18:    全局注意力     ┤
Layer 19-23: 滑动窗口 × 5  ┤
Layer 24:    全局注意力     ┤
Layer 25-29: 滑动窗口 × 5  ┤
Layer 30:    全局注意力     ┘
```

> 30 层中只有 **5 层全局注意力**，其余 25 层都是滑动窗口（只看最近 1024 token）。这也解释了为什么 MoE 在 128K 长上下文检索任务上比 Dense（60 层更多全局注意力）差距较大。

**关键数字**：
- 只有 **30 层**（Dense 的一半），用更宽的层（128 专家）来补偿深度
- 每层 **128 个专家**，每个 token 只激活 **Top-8**（6.25%）
- 单个专家仅 **~6M 参数**，极致细粒度分工（对比 Mixtral 8x7B 只有 8 个大专家）
- 总参数 25.2B，但每次推理只计算 **3.8B**
- 单次前向传播的 FLOPs ≈ **2 × 3.8B ≈ 7.6 GFLOPs/token**（是 Dense 的 1/8）

---

## 5. 核心差异：FFN vs MoE Layer

这是两个模型**唯一的结构性差异**——注意力层完全一样，区别只在 FFN 层。

### Dense FFN（31B 使用）

```
输入 h (4096 维)
    │
    ├──→ gate_proj (4096 → ~16384) ──→ GELU ──┐
    │                                          ⊙ 逐元素相乘
    └──→ up_proj   (4096 → ~16384) ───────────┘
                                    │
                              down_proj (~16384 → 4096)
                                    │
                              输出 h' (4096 维)
```

- **每个 token 都走同一个 FFN**
- 参数量 = 3 × 4096 × 16384 ≈ **201M / 层**
- 60 层 FFN 总参数 ≈ **12B**

### MoE Layer（26B-A4B 使用）

```
输入 h (5376 维)
    │
    ▼
Router: h × W_router (5376 → N_experts) → Softmax
    │
    ▼ 选出 Top-K 个专家及其权重 w_k
    │
    ├──→ Expert_i: 和 Dense FFN 结构完全一样
    │    gate_proj → GELU → ⊙ up_proj → down_proj
    │    输出: e_i
    │
    ├──→ Expert_j: 同上
    │    输出: e_j
    │
    ▼
输出 h' = w_i · e_i + w_j · e_j  (加权求和)
```

- **每个 token 只走被选中的 K 个专家**
- 单个专家参数量远小于 Dense FFN
- 但专家总数 × 单个专家参数 ≈ 总 FFN 参数量很大（存储了更多知识）

### 对比图

```mermaid
graph TB
    subgraph "Dense FFN"
        D_in["h (4096)"] --> D_FFN["单个大 FFN<br/>201M 参数<br/>每个 token 全量计算"]
        D_FFN --> D_out["h'"]
    end
    
    subgraph "MoE Layer"
        M_in["h (5376)"] --> M_Router["Router"]
        M_Router --> M_E1["Expert 1 ✅<br/>小 FFN"]
        M_Router --> M_E2["Expert 2 ✅<br/>小 FFN"]
        M_Router -.-> M_E3["Expert 3 ❌"]
        M_Router -.-> M_E4["Expert 4 ❌"]
        M_Router -.-> M_EN["Expert N ❌"]
        M_E1 --> M_Sum["加权求和"]
        M_E2 --> M_Sum
        M_Sum --> M_out["h'"]
    end
    
    style D_FFN fill:#2196F3,color:#fff
    style M_Router fill:#FF9800,color:#fff
    style M_E1 fill:#4CAF50,color:#fff
    style M_E2 fill:#4CAF50,color:#fff
    style M_E3 fill:#9E9E9E,color:#fff
    style M_E4 fill:#9E9E9E,color:#fff
    style M_EN fill:#9E9E9E,color:#fff
```

---

## 6. Router 路由机制

Router 是 MoE 架构的"大脑"，决定每个 token 该由哪些专家处理。

### 工作流程

```mermaid
sequenceDiagram
    participant T as Token 隐藏状态 h
    participant R as Router (线性层)
    participant S as Softmax
    participant TopK as Top-K 选择
    participant E as 被选中的专家们
    participant O as 输出
    
    T->>R: h (2816维) × W_router (2816×128)
    R->>S: 128 个 logits → Softmax → 概率分布
    S->>TopK: [0.01, 0.15, 0.003, ..., 0.12, ...] (128个值)
    Note over TopK: 选出得分最高的 8 个专家
    TopK->>E: 激活 Expert 7,23,41,55,67,89,102,118
    E->>O: output = Σ wₖ·Eₖ(h), k∈Top-8
```

### 负载均衡问题

MoE 训练中有一个经典难题：**专家坍塌（Expert Collapse）**

```mermaid
graph LR
    subgraph "理想状态：均匀分配"
        T1["Token 1"] --> EA1["Expert A"]
        T2["Token 2"] --> EB1["Expert B"]
        T3["Token 3"] --> EC1["Expert C"]
        T4["Token 4"] --> ED1["Expert D"]
    end
    
    subgraph "坍塌状态：都挤一个专家"
        T5["Token 1"] --> EA2["Expert A 🔥 过载"]
        T6["Token 2"] --> EA2
        T7["Token 3"] --> EA2
        T8["Token 4"] --> EA2
        EB2["Expert B 💤 闲置"]
        EC2["Expert C 💤 闲置"]
    end
    
    style EA2 fill:#f44336,color:#fff
    style EB2 fill:#9E9E9E,color:#fff
    style EC2 fill:#9E9E9E,color:#fff
```

**解决方案**：训练时加入 **辅助损失函数（Auxiliary Loss）**，惩罚负载不均衡的情况，强制 Router 把 token 分散到不同专家。

### 「有哪些专家模型」——官方没有名单

- Google **没有**为 128 个专家起名字，也**没有**公开「Expert 37 = 代码、Expert 52 = 中文」这类映射。
- 在实现上只有 **Expert 0 ~ Expert 127**：每个都是**同构**的小 FFN（2816→704→2816），权重不同而已。
- 训练后 Router 会学到某种**隐式分工**（有的专家在某些层、某些 token 上更常被选中），但这是**涌现现象**，边界模糊，不能当成严格的「业务模块」。

### 如何自己看「谁在干活」

可以用 Hook 读取每层 `Gemma4TextRouter` 的输出第三项 `top_k_index`，统计不同输入下哪些 **ID** 出现得多。

项目里脚本：`analyze_experts.py`（需将 `text_config._experts_implementation` 设为 `"eager"`，否则在部分 GPU 上 `grouped_mm` 会报错）。

下面是一次示例运行（**仅供参考**：同一段话换措辞、换 tokenizer 长度，排序会变；**不能**据此给专家贴永久标签）：

| 测试场景 | 出现较多的专家 ID（Top 若干，计数） |
|:---|:---|
| 中文 | 52, 37, 42, 49, 124, 97, 92, 6, … |
| 英文 | 56, 102, 42, 37, 64, 120, 53, … |
| 数学式 | 102, 74, 28, 20, 16, 33, 111, … |
| Python 代码 | 37, 111, 74, 40, 54, 75, 102, … |
| 日语 | 37, 0, 111, 113, 56, 85, 106, … |
| JSON + 中文 | 37, 48, 120, 117, 0, 106, 43, … |

可见 **37、102、28、56** 等在多类输入里都会反复出现——更像「通用高频专家」，而不是「只属于某一语种」的硬分区。

---

## 7. 注意力机制

两个模型都使用**全局注意力与滑动窗口注意力交替**的策略，但具体配置不同。

### 注意力类型交替

```mermaid
graph TD
    subgraph "层级注意力模式"
        L1["Layer 1: 全局注意力 🌍<br/>看到所有 token"]
        L2["Layer 2: 滑动窗口 🪟<br/>只看最近 1024 个 token"]
        L3["Layer 3: 全局注意力 🌍"]
        L4["Layer 4: 滑动窗口 🪟"]
        L5["...交替进行..."]
        
        L1 --> L2 --> L3 --> L4 --> L5
    end
```

### 为什么要交替？

| 注意力类型 | 计算复杂度 | 能力 |
|:---|:---|:---|
| 全局注意力 | O(n²)，n 为序列长度 | 能捕捉任意距离的依赖关系 |
| 滑动窗口 | O(n × w)，w=1024 | 只关注局部上下文，计算量小 |

交替使用 = **用局部注意力处理大部分"就近参考"的场景，用全局注意力处理需要"远距离回忆"的场景**。

### GQA（分组查询注意力）

```mermaid
graph LR
    subgraph "31B Dense: 32Q / 16KV"
        Q1["Q₁ Q₂"] --> KV1["KV₁"]
        Q3["Q₃ Q₄"] --> KV2["KV₂"]
        Q5["..."] --> KV3["..."]
        Q31["Q₃₁ Q₃₂"] --> KV16["KV₁₆"]
    end
    
    subgraph "26B MoE: 16Q / 8KV"
        QA["Q₁ Q₂"] --> KVA["KV₁"]
        QB["Q₃ Q₄"] --> KVB["KV₂"]
        QC["..."] --> KVC["..."]
        QD["Q₁₅ Q₁₆"] --> KVD["KV₈"]
    end
```

| | 31B Dense | 26B-A4B MoE |
|:---|:---|:---|
| Q 头数 | 32 | 16 |
| KV 头数 | 16 | 8 |
| head_dim | 128 | 256 |
| Q/KV 比 | 2:1 | 2:1 |
| 全局注意力 KV 头 | 16 | 2（`num_global_key_value_heads`） |
| 层数 | 60 | 30 |
| KV Cache 大小 | 大（60层×16KV头×128dim） | **小（30层×8KV头×256dim）** |

MoE 版本层数只有一半，且全局注意力层只有 5 层（全局 KV 头仅 2 个），KV Cache 显存占用远小于 Dense。

---

## 8. Per-Layer Embedding (PLE)

PLE（Per-Layer Embedding）是 Gemma 4 引入的新技术，提升参数效率。

### 传统方式 vs PLE

```mermaid
graph TD
    subgraph "传统: 所有层共享同一个 Embedding"
        Token["Token ID"] --> Emb["Embedding Table<br/>(262144 × 4096)"]
        Emb --> L1_old["Layer 1"]
        Emb --> L2_old["Layer 2"]
        Emb --> LN_old["Layer N"]
        
        style Emb fill:#9E9E9E,color:#fff
    end
    
    subgraph "PLE: 每层有独立的嵌入变换"
        Token2["Token ID"] --> Emb2["Base Embedding"]
        Emb2 --> T1["Transform₁"]
        Emb2 --> T2["Transform₂"]
        Emb2 --> TN["Transform_N"]
        T1 --> L1_new["Layer 1"]
        T2 --> L2_new["Layer 2"]
        TN --> LN_new["Layer N"]
        
        style T1 fill:#4CAF50,color:#fff
        style T2 fill:#4CAF50,color:#fff
        style TN fill:#4CAF50,color:#fff
    end
```

### 为什么 PLE 有效？

传统方式中，浅层和深层看到的是完全相同的输入表示。但实际上：
- **浅层**需要更多的表面特征（词形、语法）
- **深层**需要更多的语义特征（含义、逻辑）

PLE 让每一层都能对输入做一个轻量级的变换，使得不同深度的层看到"适合自己的"输入表示，提升了参数利用效率。

---

## 9. 视觉编码器

两个模型共享同一个视觉编码器 **SigLIP2**，约 550M 参数。

### 多模态处理流程

```mermaid
graph LR
    subgraph "输入"
        Img["🖼️ 图像<br/>(任意分辨率)"]
        Txt["📝 文本"]
    end
    
    subgraph "视觉处理"
        Img --> Resize["动态分辨率处理<br/>切分为 patch"]
        Resize --> SigLIP["SigLIP2 编码器<br/>(~550M 参数)<br/>ViT 架构"]
        SigLIP --> VProj["线性投影层<br/>映射到 LLM 隐藏维度"]
    end
    
    subgraph "语言模型"
        Txt --> TokEmb["Token Embedding"]
        VProj --> Concat["拼接"]
        TokEmb --> Concat
        Concat --> LLM["Gemma 4 LLM<br/>(Dense 或 MoE)"]
        LLM --> Output["输出文本"]
    end
    
    style SigLIP fill:#9C27B0,color:#fff
    style LLM fill:#2196F3,color:#fff
```

### 关键设计

| 特性 | 说明 |
|:---|:---|
| 可变分辨率 | 不强制缩放到固定尺寸，保留原始细节 |
| Pan & Scan | 智能裁剪策略，关注图像重要区域 |
| 软 Token 上限 | 控制视觉 token 数量，避免占用过多上下文 |
| 视频支持 | 将视频帧序列作为多张图像输入 |

---

## 10. 性能与效率对比

### 计算效率对比

```mermaid
graph LR
    subgraph "每个 Token 的计算量"
        D["Dense 31B<br/>61.4 GFLOPs/token"]
        M["MoE 26B-A4B<br/>7.6 GFLOPs/token"]
    end
    
    D ---|"8× 差距"| M
    
    style D fill:#2196F3,color:#fff
    style M fill:#4CAF50,color:#fff
```

### 显存占用分析

| 组件 | 31B Dense | 26B-A4B MoE |
|:---|:---|:---|
| 模型权重 (BF16) | ~62 GB | ~48 GB |
| KV Cache (32K ctx) | ~15 GB (60层×16KV头) | ~4 GB (30层×6KV头) |
| 激活值 (推理) | ~2 GB | ~1 GB |
| **总计 (推理)** | **~79 GB** | **~53 GB** |

> 💡 MoE 虽然要加载全部 48GB 权重到显存（所有专家都要在），但 KV Cache 只有 Dense 的 1/4，因为层数减半 + KV 头更少。

### Benchmark 性能 vs 计算量

```mermaid
quadrantChart
    title 性能 vs 计算效率
    x-axis "低计算量" --> "高计算量"
    y-axis "低性能" --> "高性能"
    quadrant-1 "高性能高成本"
    quadrant-2 "效率之王"
    quadrant-3 "低端"
    quadrant-4 "性价比差"
    "31B Dense": [0.85, 0.92]
    "26B-A4B MoE": [0.25, 0.88]
    "Gemma 3 27B": [0.70, 0.45]
```

### 关键 Benchmark 对比

| Benchmark | 31B Dense | 26B-A4B MoE | MoE 达到 Dense 的 % |
|:---|:---|:---|:---|
| MMLU Pro | 85.2% | 82.6% | 96.9% |
| AIME 2026 | 89.2% | 88.3% | 99.0% |
| GPQA Diamond | 84.3% | 82.3% | 97.6% |
| LiveCodeBench v6 | 80.0% | 77.1% | 96.4% |
| Codeforces ELO | 2150 | 1718 | 79.9% |
| MRCR 128K | 66.4% | 44.1% | 66.4% |
| HLE | 19.5% | 8.7% | 44.6% |

**结论**：
- 常规任务（MMLU、AIME、GPQA）：MoE 达到 Dense **96-99%** 的性能，用 **1/8 的计算量**
- 极端任务（HLE、长上下文）：Dense 的深度优势（60层）明显体现，MoE 差距较大

---

## 11. 微调注意事项

### Dense vs MoE 微调策略差异

```mermaid
graph TD
    subgraph "Dense 31B 微调"
        D_Base["Base 模型权重"]
        D_Base --> D_LoRA["LoRA 适配器<br/>挂在 Attention + FFN 的线性层上"]
        D_LoRA --> D_Train["标准训练流程<br/>QLoRA 4-bit ≈ 22GB 显存"]
    end
    
    subgraph "MoE 26B-A4B 微调"
        M_Base["Base 模型权重"]
        M_Base --> M_Choice{"微调哪些部分？"}
        M_Choice -->|"方案A"| M_Attn["只微调 Attention 层<br/>（最安全，不影响专家路由）"]
        M_Choice -->|"方案B"| M_All["微调 Attention + 所有专家<br/>（效果最好，但可能破坏路由平衡）"]
        M_Choice -->|"方案C"| M_Active["只微调被激活的专家<br/>（折中方案）"]
    end
    
    style D_LoRA fill:#2196F3,color:#fff
    style M_Attn fill:#4CAF50,color:#fff
    style M_All fill:#FF9800,color:#fff
    style M_Active fill:#9C27B0,color:#fff
```

### LoRA 配置建议

| 配置项 | Dense 31B | MoE 26B-A4B |
|:---|:---|:---|
| target_modules | `q,k,v,o,gate,up,down_proj` | `q,k,v,o_proj`（保守）<br/>或加上专家层（激进） |
| rank (r) | 16~64 | 16~32 |
| lora_alpha | 2 × r | 2 × r |
| QLoRA 显存 | ~22 GB | ~16 GB |
| 训练注意事项 | 标准流程 | 注意 Router 冻结/解冻策略 |

### MoE 微调的特殊考量

1. **Router 一般冻结**：Router 的权重在预训练中已经学好了"分诊"能力，微调时通常不动它
2. **负载均衡**：如果微调数据分布和预训练差异大，可能导致某些专家过载
3. **专家冻结策略**：可以只微调最常被激活的 Top 专家，冻结其余的
4. **推荐工具**：Unsloth 对 MoE 微调有专门优化，自动处理上述问题

---

## 总结

```mermaid
mindmap
    root((Gemma 4 架构对比))
        Dense 31B
            60 层深度
            30.7B 全量计算
            适合极端推理任务
            长上下文更强
            微调简单直接
        MoE 26B-A4B
            30 层 + 多专家
            3.8B 激活参数
            推理速度 8× 快
            96-99% 性能
            参数效率碾压
        共享设计
            Transformer Decoder
            RoPE + RMSNorm
            滑动窗口注意力
            SigLIP2 视觉
            PLE 每层嵌入
            256K 上下文
```

> **一句话总结**：Dense 31B 是"一个超强全能选手"，MoE 26B-A4B 是"一支高效的专家团队"。团队用 1/8 的工作量完成了 96% 的任务质量，但在需要极深思考的场景下，全能选手的 60 层深度优势不可替代。
