# CS885 Assignment2
## Assignment3 Part1 Partially Observable RL
**環境設定**
* 完整狀態空間：4 維 (位置 x, 速度 ẋ, 角度 θ, 角速度 θ̇)
* 部分觀測空間：2 維 (僅有位置 x 和角度 θ)
* 動作空間：2 個離散動作（向左推或向右推）
* 最大步數：200 步
* 獎勵機制：每存活一步獲得 +1 獎勵

**超參數設定**

| 參數 | 數值 | 說明 |
| -------- | -------- | -------- |
| Episodes     | 2000     | 總訓練回合數     |
| Replay Buffer Size     | 10000     | 經驗回放緩衝區大小     |
| Minibatch Size     | 64 (DQN) / 8 (DRQN)     | 每次訓練的樣本數    |
| Learning Rate     | 5e-4     | Adam 優化器學習率    |
| Gamma (γ)     | 0.99     | 折扣因子    |
| Epsilon Start     | 1.0     | 初始探索率     |
| Epsilon End     | 0.1     | 最終探索率    |
| Epsilon Decay Steps     | 10000    | 探索率衰減步數     |
| Train After Episodes     | 10     | 開始訓練前的收集回合數     |
| Train Epochs     | 25     | 每次更新的訓練輪數     |
| Target Network Update     | 每 10 回合     | 目標網絡更新頻率     |
| Hidden Layer Size    | 512    | 全連接層隱藏單元數     |
| LSTM Hidden Size     | 256    | LSTM 隱藏狀態維度     |

**隨機種子**
使用 5 個不同的隨機種子 [1, 2, 3, 4, 5] 進行多次實驗，以確保結果的可靠性。

### DQN 基準實驗結果
**DQN 網絡架構**
Input (2) → Linear(512) → ReLU → Linear(512) → ReLU → Linear(2) → Q值

**DQN 性能表現**
<img width="640" height="480" alt="DQN_result" src="https://github.com/user-attachments/assets/b1b7980c-00cb-4230-aedc-6d9c4d0bdc9d" />
其中 x 軸為回合數（2000 個回合），y 軸為獲得的獎勵。

**DQN 的主要限制：**
1. 無法從單一觀測推斷速度資訊
2. 僅依賴當前觀測做決策，缺乏時序記憶
3. 在需要歷史資訊的情境下表現不佳

### DRQN 實作與結果
**DRQN 網絡架構**
```python
class DRQN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(OBS_N, HIDDEN)      # 2 → 512
        self.relu = torch.nn.ReLU()
        self.lstm = torch.nn.LSTM(HIDDEN, LSTM_HIDDEN, # 512 → 256
                                  batch_first=True)
        self.fc2 = torch.nn.Linear(LSTM_HIDDEN, ACT_N) # 256 → 2
```
架構說明：
* 輸入層：接收 2 維觀測（位置和角度）
* 特徵提取層：全連接層將觀測映射到 512 維特徵空間
* LSTM 層：處理時序資訊，維護 256 維隱藏狀態
* 輸出層：產生 2 個動作的 Q 值

**實作要點**
**(1) LSTM 模型實作**
LSTM 層能夠記住過去的觀測序列，從中推斷隱藏的速度資訊：
* 隱藏狀態 (h, c) 在 episode 內持續傳遞
* 每個 episode 開始時重置隱藏狀態
* 使用 batch_first=True 方便序列處理

**(2) 經驗回放緩衝區調整**
```python
class ReplayBuffer:
    def __init__(self, N, recurrent=False):
        self.recurrent = recurrent
        self.current_episode = []
```
與 DQN 的差異：
* DQN：儲存單一轉換 (s, a, r, s', d)
* DRQN：儲存完整 episode 序列 [(s₁, a₁, r₁, s'₁, d₁), ..., (sₜ, aₜ, rₜ, s'ₜ, dₜ)]

**(3) 策略函數處理隱藏狀態**
```python
def policy(env, obs):
    global hidden_state
    obs = t.f(obs).view(1, -1)
    
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        with torch.no_grad():
            qvalues, hidden_state = Q(obs, hidden_state)
            action = torch.argmax(qvalues).item()
```
關鍵改進：
* 維護全局 hidden_state 變數
* 在 episode 開始時初始化隱藏狀態
* 選擇動作時更新並傳遞隱藏狀態

**(4) 網絡更新函數優化**
```python
def update_networks(epi, buf, Q, Qt, OPT):
    episodes = buf.sample(MINIBATCH_SIZE, t)
    num_episodes_per_update = min(8, len(episodes))
    
    for episode in sampled_episodes:
        S, A, R, S2, D = episode
        # 批次處理整個序列
        S_tensor = torch.stack([t.f(s) for s in S]).unsqueeze(0)
        
        # 一次前向傳播處理整個序列
        lstm_out, _ = Q.lstm(x, hidden)
        q_values = Q.fc2(lstm_out.squeeze(0))
```
效率優化：
* 每次更新處理 8 個 episode（而非 64 個）
* 批次處理整個序列，避免逐步計算
* 使用梯度裁剪防止梯度爆炸

**DRQN 實驗結果**
<img width="640" height="480" alt="DRQN_result" src="https://github.com/user-attachments/assets/71756489-5b22-44ea-b41d-a63de126bfd5" />
其中 x 軸為回合數（2000 個回合），y 軸為獲得的獎勵。

學習曲線特徵：
* 初期學習（0-250 episodes）：快速學習，獎勵從 20 躍升至 150
* 中期穩定（250-1000 episodes）：持續在 160-180 之間波動
* 後期表現（1000-2000 episodes）：穩定在 170-190 左右
* 最終平均獎勵：約 180-185

標準差分析：
* 藍色陰影區域顯示不同種子間的變異性
* 標準差在訓練後期保持穩定，顯示學習的一致性
* 部分時刻達到接近 200 的最大獎勵

### DQN vs DRQN 性能比較
**定量比較**

| 指標 | DQN | DRQN | 改進幅度 |
| -------- | -------- | -------- | -------- |
| 初期學習速度     | 中等     | 中等     | 差不多     |
| 平均獎勵 (最終)     | 35-40     | 180-185     | 顯著改善     |
| 最佳獎勵     | 40-45     | 190-195     | 顯著改善     |
| 學習穩定性    | 較大波動     | 較為穩定     | 顯著改善     |
| 收斂速度     | 中等     | 中等     | 差不多     |

**DRQN 優勢：**
1. 時序記憶能力
* LSTM 能夠記住過去的觀測序列
* 從位置和角度的變化推斷速度和角速度
* 有效解決部分可觀測問題

2. 隱藏狀態推斷
* 觀測序列：$[x₁, θ₁] → [x₂, θ₂] → [x₃, θ₃]$
* LSTM 推斷：$ẋ ≈ (x₂ - x₁)/Δt, θ̇ ≈ (θ₂ - θ₁)/Δt$

3. 長期依賴建模
* 能夠捕捉跨多個時間步的依賴關係
* 更好地預測未來狀態

**DQN 限制：**
1. 資訊不完整
* 僅基於當前觀測 [x, θ]
* 無法得知 [ẋ, θ̇]，導致決策次優

2. Markov 假設違反
* 在 POMDP 中，單一觀測不滿足 Markov 性質
* Q(s, a) 的估計存在偏差

3. 學習效率低
* 需要更多探索才能間接學習動態資訊
* 性能天花板較低

**為何 DRQN 表現更好**
**理論原因：**
1. POMDP vs MDP
* CartPole 在部分可觀測下是 POMDP
* DRQN 通過 LSTM 將 POMDP 轉換為 belief-MDP
* DQN 將 POMDP 當作 MDP 處理（錯誤假設）

2. Belief State 建模
* Belief state: $b(s) = P(s|o₁, o₂, ..., oₜ)$
* LSTM hidden state ≈ sufficient statistic of belief state

3. 資訊整合
* DRQN: $Q(h_t, a)$ where $h_t = LSTM(o₁:t)$
* DQN: $Q(o_t, a) - 資訊損失嚴重$

**實證觀察：**
* 收斂速度：DRQN 在約 300 episodes 就達到良好性能，DQN 需要約 400 episodes
* 最終性能：DRQN 接近完全可觀測環境的理論上限（200），DQN 存在顯著差距
* 穩定性：DRQN 的學習曲線更平滑，標準差更小

## Assignment3 Part2 Distributional RL
**環境設定**
**Noisy CartPole 環境**
這是一個修改過的 CartPole 環境，加入了隨機噪音使其成為隨機環境：
**環境修改：**
1. 力量噪音：每次施加的力量乘以 (1 + ε)，其中 ε ~ Uniform(-0.05, 0.05)
2. 角度噪音：每個時間步的角度乘以 (1 + ε)，其中 ε ~ Uniform(-0.05, 0.05)
3. 摩擦力：
* 小車與地面摩擦係數：5 × 10⁻⁴
* 桿子的空氣阻力：2 × 10⁻⁶

**環境特性：**
* 狀態空間：4 維 (位置 x, 速度 ẋ, 角度 θ, 角速度 θ̇)
* 動作空間：2 個離散動作（向左推或向右推）
* 最大步數：500 步
* 獎勵機制：每存活一步獲得 +1 獎勵
* 隨機性：由於噪音的存在，相同狀態和動作可能產生不同結果

**超參數設定**

| 參數 | 數值 | 說明 |
| -------- | -------- | -------- |
| Episodes     | 500    | 總訓練回合數     |
| Replay Buffer Size     | 10000     | 經驗回放緩衝區大小     |
| Minibatch Size     | 64     | 每次訓練的樣本數     |
| Learning Rate     | 5e-4     | Adam 優化器學習率     |
| Gamma (γ)     | 0.99     | 折扣因子     |
| Epsilon Start     | 1.0     | 初始探索率     |
| Epsilon End     | 0.1     | 最終探索率     |
| Epsilon Decay Steps     | 10000     | 探索率衰減步數     |
| Train After Episodes     | 10     | 開始訓練前的收集回合數     |
| Train Epochs     | 25     | 每次更新的訓練輪數     |
| Target Network Update     | 每 10 回合     | 目標網絡更新頻率     |
| Hidden Layer Size     | 512     | 全連接層隱藏單元數     |

**C51 特定參數：**
* Atoms (N): 51 - 離散化回報分佈的原子數
* $V_{min}$: 0 - 支撐集的最小值
* $V_{max}$: 500 - 支撐集的最大值
* $Δz: (V_{max} - V_{min}) / (N - 1) ≈ 10 - 原子間距$

**隨機種子**
使用 5 個不同的隨機種子 [1, 2, 3, 4, 5] 進行多次實驗。

### DQN 基準實驗
**核心思想：** 學習動作價值函數 Q(s, a)，表示在狀態 s 採取動作 a 的期望回報。
**網絡架構：**
> Input (4) → Linear(512) → ReLU → Linear(512) → ReLU → Linear(2) → Q值

**更新規則：**
* 目標：$y = r + γ max_{a'} Q_target(s', a')$
* 損失：$L = MSE(Q(s, a), y)$

**限制：**
* 只預測期望值，忽略回報的不確定性
* 在隨機環境中可能次優
* 無法區分風險和不確定性

**DQN 結果**
![image](https://hackmd.io/_uploads/H1xSt6t-We.png)
其中 x 軸表示回合數（至少 200 個回合），y 軸表示每個回合獲得的獎勵。
* 初期（0-100 episodes）：快速學習基本策略
* 中期（100-300 episodes）：性能穩步提升
* 後期（300-500 episodes）：接近收斂
* 最終平均獎勵：預計 300-350 左右

**挑戰：**
1. 環境的隨機性增加學習難度
2. 噪音導致相同策略的回報變異性大
3. DQN 的期望值估計可能不穩定

### C51 演算法實作
**分佈式 RL 的動機：**
* 傳統 DQN 只學習期望：$E[Z(s,a)]$
* C51 學習完整分佈：整個隨機變數 $Z(s,a)$
* 保留更多資訊，特別是在隨機環境中

**數學基礎：**
回報分佈滿足 Bellman 方程：
> $Z(s, a) = r + γ Z(s', a')$

C51 將分佈離散化為 N 個原子：
> $z_i ∈ {V_min, V_min + Δz, ..., V_max}, i = 0, 1, ..., N-1$

網絡輸出每個原子的機率：
> $p_i(s, a) = P(Z(s, a) = z_i)$

**網絡架構**
```python
class C51Network:
    Input: state (4,)
    ↓
    Linear(4 → 512) + ReLU
    ↓
    Linear(512 → 512) + ReLU
    ↓
    Linear(512 → ACT_N × ATOMS) = Linear(512 → 102)
    ↓
    Reshape to (ACT_N, ATOMS) = (2, 51)
    ↓
    Softmax over atoms → 每個動作的機率分佈
```
**輸出解釋：**
* 對於每個動作 a，網絡輸出 51 個機率值
* $p_i(s, a)$ 表示回報為 $z_i$ 的機率
* $Σ p_i(s, a) = 1$ (機率和為 1)

**實作要點**
**(1) Policy 函數實作**
```python
def policy(env, obs):
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        # 獲取分佈
        logits = Z(obs).view(-1, ACT_N, ATOMS)
        probs = F.softmax(logits, dim=2)
        
        # 計算期望 Q 值：Q(s,a) = Σ p_i × z_i
        q_values = (probs * support).sum(dim=2)
        
        # 選擇最大 Q 值的動作
        action = torch.argmax(q_values).item()
```
關鍵點：
* 使用分佈的期望值做決策
* 保留分佈資訊用於訓練
* 貪婪策略基於期望回報

**(2) 分佈投影（Distribution Projection）**
```python
# 步驟 1: 計算 Bellman 更新後的支撐集
Tz_j = r + γ × z_j  # 對每個原子

# 步驟 2: 裁剪到 [V_min, V_max]
Tz_j = clip(Tz_j, V_min, V_max)

# 步驟 3: 投影到原始支撐集
b_j = (Tz_j - V_min) / Δz
l_j = floor(b_j)  # 下界索引
u_j = ceil(b_j)   # 上界索引

# 步驟 4: 分配機率質量
m_l = m_l + p_j × (u - b)  # 給下界的機率
m_u = m_u + p_j × (b - l)  # 給上界的機率
```

**(3) Update Networks 實作**
**1. 前向傳播當前網絡：**
```python
logits = Z(S).view(batch, ACT_N, ATOMS)
logits_a = logits[range(batch), A]  # 選擇採取的動作
log_probs = F.log_softmax(logits_a, dim=1)
```
**2. 計算目標分佈：**
```python
# 從目標網絡獲取下一狀態的分佈
logits_next = Zt(S2).view(batch, ACT_N, ATOMS)
probs_next = F.softmax(logits_next, dim=2)

# 選擇最佳動作（Double DQN 思想）
q_next = (probs_next * support).sum(dim=2)
best_actions = torch.argmax(q_next, dim=1)
probs_next_a = probs_next[range(batch), best_actions]
```
**3. Bellman 更新和投影：**
```python
# 計算 Tz
Tz = R.unsqueeze(1) + GAMMA * support * (1 - D)

# 投影
b = (Tz - VMIN) / delta_z
l, u = b.floor(), b.ceil()

# 分配機率質量
m[l] += probs_next_a * (u - b)
m[u] += probs_next_a * (b - l)
```
**4. 損失計算：**
```python
# 交叉熵損失（KL 散度）
loss = -(m * log_probs).sum(dim=1).mean()
```
**實驗結果**
![image](https://hackmd.io/_uploads/SJgoj6tWWe.png)
其中 x 軸表示回合數（至少 200 個回合），y 軸表示每個回合獲得的獎勵。
* 初期學習差不多但
* 中期開始超越 DQN
* 後期性能更高
* 最終平均獎勵：約 400-450

### C51 vs DQN 關鍵差異

| 層面 | DQN | C51 |
| -------- | -------- | -------- |
| 預測目標     | 標量 Q 值     | 機率分佈     |
| 網絡輸出     | ACT_N 個值     | ACT_N × ATOMS 個值     |
| 損失函數     | MSE     | 交叉熵（KL 散度）     |
| Bellman 更新     | 期望值     | 分佈投影     |
| 決策方式     | argmax Q(s,a)     | argmax E[Z(s,a)]     |
| 資訊保留     | 僅期望     | 完整分佈     |

**定量比較**
**DQN:**
*   Final Average Reward: 329.83 ± 9.49
*   Best Average Reward: 350.93
*   Convergence Episode: 429

**C51:**
*   Final Average Reward: 397.76 ± 71.00
*   Best Average Reward: 448.51
*   Convergence Episode: 426

**Improvement: 20.59%**

**為何 C51 表現更好**
1. 完整資訊建模
> DQN:  只知道 E[return] = 400
   C51:  知道 P(return = 350) = 0.3
                P(return = 400) = 0.4
                P(return = 450) = 0.3
2. 更好的信用分配
* C51 保留了回報的多模態性
* 能區分「穩定高回報」和「不穩定高回報」
* 幫助更好地評估動作價值

3. 投影保留結構
* Bellman 算子在分佈空間的收縮性更好
* 投影步驟保留了分佈的形狀特徵
* 減少了期望值估計的偏差

4. 隱式的探索
* 分佈提供了不確定性的信息
* 類似於樂觀初始化，鼓勵探索不確定的狀態
* 在隨機環境中特別有效



