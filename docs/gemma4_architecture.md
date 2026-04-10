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
| 隐藏维度 | 4096 | 5376 |
| 注意力头数 | 32 | 42 |
| KV 头数 | 16 | 6 |
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
        Norm1 --> Attn["Multi-Head Attention<br/>42 个 Q 头, 6 个 KV 头<br/>head_dim = 128"]
        Attn --> Add1["h = h + Attn(h)"]
        Add1 --> Norm2["RMSNorm"]
        Norm2 --> Router["🔀 Router (路由器)<br/>小型线性层 + Softmax"]
        
        Router -->|"得分最高"| E1["Expert 1<br/>FFN"]
        Router -->|"得分第二"| E2["Expert 2<br/>FFN"]
        Router -.->|"未选中"| E3["Expert 3<br/>FFN"]
        Router -.->|"未选中"| E4["Expert 4<br/>FFN"]
        Router -.->|"未选中"| EN["Expert N<br/>FFN"]
        
        E1 --> WeightSum["加权求和<br/>w₁·E₁(h) + w₂·E₂(h)"]
        E2 --> WeightSum
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

**关键数字**：
- 只有 **30 层**（Dense 的一半），用更宽的层来补偿深度
- 每层有多个专家 FFN，每个 token 只激活 **Top-K** 个
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
    
    T->>R: h × W_router
    R->>S: logits → 概率分布
    S->>TopK: [0.05, 0.82, 0.01, 0.72, 0.03, ...]
    Note over TopK: 选出得分最高的 K 个
    TopK->>E: 激活 Expert 2 (w=0.82) 和 Expert 4 (w=0.72)
    E->>O: output = 0.82 × E₂(h) + 0.72 × E₄(h)
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

### Router 学到了什么？

训练完成后，不同专家会自然地"专精"不同类型的知识：

| 专家 | 可能擅长的领域（示意） |
|:---|:---|
| Expert 1 | 数学运算、逻辑推理 |
| Expert 2 | 自然语言理解、语义分析 |
| Expert 3 | 代码生成、编程语法 |
| Expert 4 | 多语言翻译、跨语言对齐 |
| Expert 5 | 事实性知识、百科问答 |
| ... | ... |

> ⚠️ 注意：这种"专精"是自发涌现的，不是人为指定的。实际中每个专家的功能边界是模糊的。

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
    
    subgraph "26B MoE: 42Q / 6KV"
        QA["Q₁..Q₇"] --> KVA["KV₁"]
        QB["Q₈..Q₁₄"] --> KVB["KV₂"]
        QC["..."] --> KVC["..."]
        QD["Q₃₆..Q₄₂"] --> KVD["KV₆"]
    end
```

| | 31B Dense | 26B-A4B MoE |
|:---|:---|:---|
| Q 头数 | 32 | 42 |
| KV 头数 | 16 | 6 |
| Q/KV 比 | 2:1 | 7:1 |
| KV Cache 大小 | 较大 | **更小（KV 头更少）** |

MoE 版本用了更激进的 GQA 比例（7:1），进一步压缩了 KV Cache 的显存占用。

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

<!-- PLACEHOLDER: performance -->

---

## 11. 微调注意事项

<!-- PLACEHOLDER: finetune -->
