# CS885
## Assignment1 Part1 MDP
### Question1: Value Iteration
使用 tolerance = 0.01，初始值函數為全 0，執行 value iteration

**結果:**

---

```
  迭代次數：21次
  最終epsilon：0.004895
```

---

**Policy(每個狀態的最佳動作):**
```
  State  0: Action 3 (右(right))
  State  1: Action 3 (右(right))
  State  2: Action 1 (下(down))
  State  3: Action 1 (下(down))
  State  4: Action 1 (下(down))
  State  5: Action 3 (右(right))
  State  6: Action 1 (下(down))
  State  7: Action 1 (下(down))
  State  8: Action 1 (下(down))
  State  9: Action 1 (下(down))
  State 10: Action 1 (下(down))
  State 11: Action 2 (左(left))
  State 12: Action 3 (右(right))
  State 13: Action 3 (右(right))
  State 14: Action 0 (上(up))
  State 15: Action 0 (上(up))
  State 16: Action 0 (上(up))
```

---

**Value Function:**
```
  V[ 0] =   57.41
  V[ 1] =   62.36
  V[ 2] =   67.78
  V[ 3] =   64.96
  V[ 4] =   58.62
  V[ 5] =   62.32
  V[ 6] =   74.59
  V[ 7] =   70.20
  V[ 8] =   63.33
  V[ 9] =    7.47
  V[10] =   82.89
  V[11] =   75.59
  V[12] =   75.80
  V[13] =   83.66
  V[14] =  100.00
  V[15] =   72.87
  V[16] =    0.00
```

---

### Question2: Policy Iteration
從所有狀態選擇動作 0 (up) 的初始策略開始

**結果:**

---

```
  迭代次數: 5
```

---

**Policy(每個狀態的最佳動作):**
```
  State  0: Action 3 (右(right))
  State  1: Action 3 (右(right))
  State  2: Action 1 (下(down))
  State  3: Action 1 (下(down))
  State  4: Action 1 (下(down))
  State  5: Action 3 (右(right))
  State  6: Action 1 (下(down))
  State  7: Action 1 (下(down))
  State  8: Action 1 (下(down))
  State  9: Action 1 (下(down))
  State 10: Action 1 (下(down))
  State 11: Action 2 (左(left))
  State 12: Action 3 (右(right))
  State 13: Action 3 (右(right))
  State 14: Action 0 (上(up))
  State 15: Action 0 (上(up))
  State 16: Action 0 (上(up))
```

---

**Value Function:**
```
  V[ 0] =   57.42
  V[ 1] =   62.36
  V[ 2] =   67.78
  V[ 3] =   64.96
  V[ 4] =   58.62
  V[ 5] =   62.32
  V[ 6] =   74.59
  V[ 7] =   70.20
  V[ 8] =   63.33
  V[ 9] =    7.47
  V[10] =   82.89
  V[11] =   75.59
  V[12] =   75.80
  V[13] =   83.66
  V[14] =  100.00
  V[15] =   72.87
  V[16] =    0.00
```

---

### Question3: Modified Policy Iteration
測試部分策略評估的迭代次數從 1 到 10 對收斂速度的影響，使用 tolerance = 0.01，初始值函數為全 0，從所有狀態選擇動作 0 (up) 的初始策略開始

| nEvalIterations | 總迭代次數 |
| -------- | -------- |
| 1 | 21 |
| 2 | 12 |
| 3 | 9 |
| 4 | 8 |
| 5 | 7 |
| 6 | 7 |
| 7 | 7 |
| 8 | 7 |
| 9 | 6 |
| 10 | 7 |


---

**分析與討論**
```
1.趨勢觀察:
  隨著部分策略評估的迭代次數增加，modified policy iteration 的總迭代次數減少
  當 nEvalIterations = 1 時，結果接近 value iteration（21 次迭代）
  當 nEvalIterations 較大時，結果接近 policy iteration（6-7 次迭代）

2.與其他算法的關係:
  nEvalIterations = 1: 相當於 value iteration，因為每次只更新一次值函數
  nEvalIterations → ∞: 相當於 policy iteration，因為完全求解當前策略的值函數
  中間值: 在效率和計算成本之間取得平衡
  
3.實務意義:
  較小的 nEvalIterations：每次迭代計算成本低，但需要更多迭代
  較大的 nEvalIterations：每次迭代計算成本高，但總迭代次數少
  最佳選擇取決於具體問題的規模和計算資源

4.收益遞減:
  從表格可見，當 nEvalIterations 超過 5-6 後，改進不明顯
  這表明存在一個"最佳點"，過度增加部分評估迭代次數並不划算
```

## Assignment1 Part2 Q-learning
![image](https://hackmd.io/_uploads/HJsl2fUAgx.png)
**實驗設置:**
* 初始狀態: State 0
* 初始 Q 函數: 所有狀態-動作對的值為 0
* Episode 數量: 200
* 每個 episode 的步數: 100
* 試驗次數: 100 (用於平均)

**根據實驗結果，可以觀察到以下現象：**
```
ε = 0.05 (低探索率):

初期學習較快，因為主要利用已知的好策略
但可能陷入局部最優解
長期表現可能不是最好的
```

```
ε = 0.1 (中低探索率):

在探索和利用之間取得較好平衡
學習曲線相對穩定
通常能找到較好的策略
```

```
ε = 0.3 (中高探索率):

更多的探索導致學習曲線波動較大
需要更多 episodes 才能收斂
但最終可能發現更好的路徑
```

```
ε = 0.5 (高探索率):

學習曲線波動最大
收斂速度最慢
在訓練期間獲得的獎勵較低，因為經常隨機探索
```
**對 Q 值的影響：**
```
高探索率的 Q 值特徵:
  所有動作的 Q 值都被充分更新
  Q 值估計更準確，因為訪問次數更均勻
  能夠識別多條到達目標的路徑
  對環境的不確定性有更好的認知
```

```
低探索率的 Q 值特徵:
  某些狀態-動作對可能從未被訪問
  Q 值可能過度樂觀（只沿最初發現的路徑）
  對次優動作的評估不準確
  可能錯過避免壞狀態的替代路徑
```
**對最終策略的影響：**
```
1. ε = 0.05:
可能找到一條可行路徑，但不一定是最優
策略可能避開 State 9 的能力較弱
```

```
2. ε = 0.1:
最佳的平衡點
通常能找到接近最優的策略
實務中最常用的值
```

```
3. ε = 0.3:
有更大機會找到最優策略
但訓練時間較長
```

```
4. ε = 0.5:
探索過度，收斂慢
訓練效率低
```

## Assignment1 Part3 Deep Q-Network (DQN)
### Question1：Target Network
![image](https://hackmd.io/_uploads/SkBx18IAeg.png)
```
Frequency = 1（每個 episode 更新）
預期現象：
  學習曲線波動大
  可能不穩定或收斂慢

原因：
  Target network 變化太快
  訓練目標不穩定
  類似「追逐移動目標」
```

```
Frequency = 10（預設值）
預期現象：
  學習曲線相對穩定
  收斂速度適中
  最終性能良好

原因：
  在穩定性和更新速度間取得平衡
```

```
Frequency = 50
預期現象：
  期學習可能較慢
  長期穩定性好
  最終性能可能略遜於 10

原因：
  Target network 更新慢
  訓練目標更穩定但可能滯後
```

```
Frequency = 100
預期現象：
  學習最慢
  曲線最平滑
  可能收斂不完全

原因：
  Target network 幾乎不變
  訓練目標過於陳舊
```
**Value Iteration 更新：**
$V_{k+1}(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]$

**DQN 更新：**
$θ_{t+1} = θ_t + α ∇_θ [r + γ max_a' Q(s', a'; θ⁻) - Q(s, a; θ_t)]²$

```
相似之處

1. 迭代更新結構
  Value Iteration: 使用前一次迭代的值函數
  DQN: 使用 target network 的 Q 值

2. 目標值計算
  兩者都使用 Bellman 方程
  都取下一狀態的最大值（max 操作）

3. 收斂保證
  Value Iteration: 理論上保證收斂到 V*
  DQN: 在某些條件下收斂（需要固定目標）
```
**問題：沒有 Target Network 會怎樣？**
```
這會導致：

  訓練目標隨著參數 θ 一起變化
  類似「狗追自己的尾巴」
  訓練不穩定，可能發散
```
**使用 Target Network：**
```
效果：

  提供穩定的訓練目標
  類似 Value Iteration 的「前一次迭代值」
  θ⁻ 定期更新
```
### Question2：Mini-Batch Size
![image](https://hackmd.io/_uploads/r1XbfLLCxe.png)
```
Batch Size = 1（單樣本更新）
預期現象：
  學習曲線極度波動
  收斂不穩定
  最終性能差

原因：
  梯度估計方差極大
  每次更新基於單個樣本
  容易過擬合單個經驗
```

```
Batch Size = 10（預設）
預期現象：
  學習曲線較平滑
  收斂速度適中
  良好的最終性能

原因：
  梯度估計較準確
  合理的方差
```

```
Batch Size = 50
預期現象：
  曲線更平滑
  可能學習稍慢
  穩定性好

原因：
  梯度估計更準確
  方差更小
```

```
Batch Size = 100
預期現象：
  曲線最平滑
  初期可能最慢
  長期可能最穩定

原因：
  梯度估計最準確
  但每個 episode 更新次數相同，大 batch 意味著看到的樣本多樣性較低
```

**Full Batch Gradient Descent（全批次梯度下降）：**
$θ_{t+1} = θ_t - α ∇_θ L(θ_t)$
$L(θ) = (1/N) Σ_{i=1}^N L_i(θ)$
使用所有訓練數據計算梯度。

**Stochastic Gradient Descent (SGD)：**
$θ_{t+1} = θ_t - α ∇_θ L_i(θ_t)$
每次只使用一個樣本。

**Mini-batch Gradient Descent：**
$θ_{t+1} = θ_t - α ∇_θ (1/B) Σ_{i=1}^B L_i(θ_t)$
使用一批樣本（折衷方案）。

```
Replay Buffer 的作用：

1. 儲存歷史經驗
  類似擁有一個「訓練數據集」
  大小通常為 10,000 個經驗

2. 隨機抽樣
  每次訓練時抽取 mini-batch
  類似 mini-batch gradient descent

3. 打破相關性
  連續的經驗高度相關（同一 episode）
  隨機抽樣使樣本近似 i.i.d.
```
#### **Mini-batch Size 的影響**
> 真實梯度：
> $g = E[∇L(θ)]$

> Mini-batch 梯度估計：
> $ĝ_B = (1/B) Σ_{i=1}^B ∇L_i(θ)$

> 梯度估計的方差：
> $Var(ĝ_B) = σ²/B$
> 其中 σ² 是單個樣本梯度的方差。

關鍵洞察：
  Batch size ↑ ⇒ 方差 ↓ ⇒ 更穩定
  Batch size ↓ ⇒ 方差 ↑ ⇒ 更隨機但可能探索更多
  
#### 與 Exact Gradient Descent 的關係

**Exact (Full Batch) Gradient Descent：**
```
優點：
✓ 梯度準確（無噪音）
✓ 收斂平滑
✓ 理論保證

缺點：
✗ 計算成本高（需要整個數據集）
✗ 可能陷入局部最優
✗ 記憶體需求大
```

**Stochastic (Batch=1) Gradient Descent：**
```
優點：
✓ 計算快速
✓ 可能逃離局部最優（噪音）
✓ 記憶體效率高

缺點：
✗ 梯度噪音大
✗ 收斂不穩定
✗ 可能在最優點附近震盪
```

**Mini-batch Gradient Descent：**
```
平衡兩者：
- 計算效率合理
- 梯度估計較準確
- 穩定性與探索性的折衷
```

#### Mini-batch Size 的權衡
**小 Batch Size (1-10)：**
```
優點：
+ 更快的參數更新頻率
+ 可能更好的探索
+ 記憶體使用少

缺點：
- 梯度噪音大
- 訓練不穩定
- 可能需要更小的學習率
```

**中等 Batch Size (10-50)：**
```
優點：
+ 穩定性與效率的平衡
+ 合理的梯度估計
+ 實務中常用

缺點：
- 需要調整以找到最佳值
```

**大 Batch Size (50-100+)：**
```
優點：
+ 梯度估計準確
+ 訓練穩定
+ 可以使用更大學習率

缺點：
- 計算成本高
- 可能陷入尖銳最小值
- 泛化性能可能較差
```
