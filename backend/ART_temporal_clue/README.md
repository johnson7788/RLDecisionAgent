# “**逻辑推理类谋杀案解谜数据集**”，结构和玩法类似 **Clue/Cluedo 游戏**，每个条目由 **题目 (prompt)** + **答案 (solution)** 组成。下面我逐层解释。

---

## 数据结构说明

每个数据条目包含两个主要字段：

* **`num_clues`**
  表示这个谜题提供了多少条“逻辑线索”。
  例如第一个谜题是 72 条线索，第二个谜题只有 19 条。

* **`prompt`**
  谜题文本，描述故事背景、嫌疑人、武器、房间、时间范围、嫌疑人的动机，以及关键的逻辑线索。
  线索使用逻辑条件（if and only if、or、together at least once 等），要求推理出唯一解。

* **`solution`**
  标准答案。包含了谁是凶手、使用什么武器、在哪个房间、动机是什么，以及 bonus 题的解答。

---

## 第一个样本（72 条线索）

### 故事背景

* **嫌疑人**：10 人（Miss Peach, Monsieur Brunette, Mr. Green, ...）
* **武器**：8 种（Candlestick, Wrench, Rope, Horseshoe ...）
* **房间**：13 个（Studio, Gazebo, Kitchen, Library, ...）
* **时间**：8 个可能的时间点（08:00 PM → 03:00 AM，每小时一次）
* **动机**：10 种（Jealousy, Ambition, Power, Hatred ...）

谋杀条件：凶手必须和 Mr. Boddy 在同一个房间，房间内至少有一件武器，并且没有其他人。

### 线索示例

* “The murderer was in the Library at 12:00 AM”
  → 凶手在午夜 12 点在图书馆。
* “Mr. Green was at the Fountain at 12:00 AM”
  → Green 在喷泉。
* “Mr. Boddy was murdered at 11:00 PM”
  → 确认了谋杀时间。
* “The suspect motivated by Jealousy moved the Rope from the Trophy Room to the Library at 11:00 PM”
  → 特定动机的人在 11 点搬运了武器。

这些线索交织在一起，要求玩家/模型推理出唯一答案。

### 答案 (solution)

```json
{
  "A": "Monsieur Brunette",   // 凶手
  "B": "Horseshoe",           // 凶器
  "C": "Library",             // 凶案地点
  "D": "Jealousy",            // 动机
  "E": "Courtyard",           // bonus: 复仇动机的人 03:00 AM 在哪里
  "F": "Drawing Room",        // bonus: 烛台 01:00 AM 在哪里
  "G": "Drawing Room",        // bonus: 野心动机的人 08:00 PM 在哪里
  "H": "Fountain"             // bonus: 左轮手枪 08:00 PM 在哪里
}
```

---

## 第二个样本（19 条线索）

### 故事背景

* **嫌疑人**：5 人（Miss Scarlet, Sgt. Gray, Mrs. White, Mrs. Peacock, Mr. Green）
* **武器**：4 种（Rope, Knife, Lead Pipe, Poison）
* **房间**：6 个（Gazebo, Ballroom, Courtyard, Hall, Library, Lounge）
* **时间**：3 个时间点（11 PM, 12 AM, 01 AM）
* **动机**：5 种（Jealousy, Betrayal, Ambition, Power, Fear）

规模比第一个小很多，更容易解。

### 线索示例

* “Miss Scarlet was in the Hall at 11:00 PM”
* “Mrs. White is motivated by Fear”
* “Mr. Boddy was killed with the Poison”
* “The murderer was in the Library at 12:00 AM or the suspect motivated by Ambition was in the Gazebo at 12:00 AM”

这些线索直接缩小了范围。

### 答案 (solution)

```json
{
  "A": "Sgt. Gray",        // 凶手
  "B": "Library",          // 地点
  "C": "11:00 PM",         // 时间
  "D": "Jealousy",         // 动机
  "E": "Jealousy",         // bonus: Sgt. Gray 的动机
  "F": "Mrs. Peacock",     // bonus: 谁是背叛动机
  "G": "Hall",             // bonus: 嫉妒动机的人 01:00 AM 在哪里
  "H": "Ballroom"          // bonus: 野心动机的人 12:00 AM 在哪里
}
```

---

## 总结

这个数据集的设计特点是：

1. **结构化推理**：嫌疑人、武器、房间、时间、动机都是有限集合，解题过程是一个约束满足问题 (CSP)。
2. **多模态逻辑**：包含空间关系（north/south）、时间条件（if and only if）、动作（move/moved），以及逻辑运算（or，iff）。
3. **难度分级**：有的谜题几十条线索，很复杂；有的只有十几条，适合入门。
4. **标准答案明确**：solution 给出了唯一解，方便训练/验证 AI 模型的逻辑推理能力。
