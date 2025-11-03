# CS885 Assignment2
## Assignment2 Part1
### Maze Problem Experiment
**實驗結果：**
<img width="3572" height="2061" alt="image" src="https://github.com/user-attachments/assets/ec413b88-c416-40a0-a4cf-ebb7d6773eb1" />
1. 初期探索階段 (Episodes 0-50):
* 平均累積獎勵: 約 -150 到 -50
* 獎勵波動較大，標準差較高
* Agent 還在探索環境，經常遇到懲罰狀態
* 模型估計不準確，導致策略次優


2. 快速學習階段 (Episodes 50-100):
* 平均累積獎勵快速上升
* 從約 -50 提升到 +20
* Agent 開始學會避開懲罰狀態 (state 9)
* 轉移函數和獎勵函數的估計變得更準確


3. 穩定改進階段 (Episodes 100-150):
* 平均累積獎勵持續上升至 +40
* 標準差逐漸減小
* 策略逐漸收斂到較優解


4. 收斂階段 (Episodes 150-200):
* 平均累積獎勵穩定在 +50 左右
* Agent 學會了相對穩定的策略
* 仍保持 5% 的探索，因此未完全收斂

---

**Model-based RL 特性分析**
優點：
1. 樣本效率高：
* 每次經驗都用來更新模型（T 和 R）
* 可以從單一經驗中學習多個狀態轉移
* 在約 100 個 episodes 後就達到較好的性能


2. 可進行規劃：
* 使用 Value Iteration 在學到的模型上進行規劃
* 不需要直接與環境互動就能改進策略
* 能夠更快地傳播價值資訊


3. 收斂速度快：
* 當模型準確度提高後，策略改進迅速
* 比 model-free 方法（如 Q-learning）更快達到穩定性能



缺點：
1. 初期性能較差：
* 模型估計不準確時，策略可能很差
* 需要足夠的經驗來建立準確的模型


2. 對模型誤差敏感：
* 如果環境與估計模型差異大，性能會下降
* 初期的錯誤估計可能導致次優探索


3. 計算成本較高：
* 每個 episode 後都需要執行 Value Iteration
* 對於大型狀態空間，計算成本顯著

---


| 特性 | Model-based RL | Q-learning |
| -------- | -------- | -------- |
| 學習方式     | 學習模型 (T, R)     | 直接學習 Q-function     |
| 樣本效率     | 較高    | 較低     |
| 初期性能     | 較差（模型不準確）     | 中等    |
| 收斂速度     | 快（當模型準確）    | 較慢     |
| 穩健性    | 較低（依賴模型）     | 較高     |
| 計算成本     | 高（需要規劃）     | 低（簡單更新）     |
| 適用場景     | 小型、確定性環境     | 大型、隨機性環境     |


---

### Bandit Problem Experiment
**實驗結果：**
<img width="3571" height="2061" alt="image" src="https://github.com/user-attachments/assets/17d92231-2776-44bb-93b7-659eade88eb1" />
1. Thompson Sampling (藍色曲線)

**表現：**
* 初期 20 次迭代內快速識別最優臂
* 約 30 次迭代後，平均獎勵接近最優值 0.7
* 收斂最快且最穩定
* 最後 50 次迭代平均獎勵: 0.6850

**原因分析：**
* Beta 分布是 Bernoulli 分布的共軛先驗，更新效率高
* 貝葉斯更新自然地將後驗集中在真實機率附近
* 對於機率差異明顯的情況（0.3 vs 0.5 vs 0.7），能快速區分
* Thompson Sampling 的概率匹配（Probability Matching）特性使其探索更有效

**理論支持：**
* Thompson Sampling 對 Bernoulli bandit 有漸進最優的遺憾界 $O(log T)$
* 在實際應用中通常表現最好

2. UCB (橙色曲線)

**表現：**
* 初期需要每個臂至少拉一次（前 3 次迭代）
* 約 40-50 次迭代後收斂
* 收斂速度比 Thompson Sampling 稍慢
* 最後 50 次迭代平均獎勵: 0.6790

**原因分析：**
* UCB 使用置信上界來平衡探索與利用
* $√(2ln(t)/n)$項確保對未充分探索的臂保持樂觀
* 對於臂 0 (p=0.3)，UCB 很快就確定其為次優
* 對於臂 1 和臂 2，需要更多樣本才能確定臂 2 更優
* UCB 的決策是確定性的，不如 Thompson Sampling 的隨機性靈活

**理論支持：**
* UCB1 有理論遺憾界 $O(log T)$
* 在最壞情況下的表現有保證

3. Epsilon-Greedy (綠色曲線)

**表現：**
* 收斂最慢，約 80-100 次迭代才接近最優
* 學習曲線波動較大
* 即使後期，仍有較多探索（雖然 ε 很小）
* 最後 50 次迭代平均獎勵: 0.6520

**原因分析：**
* ε = 1/t 意味著初期探索很多（t=1 時 ε=1，完全隨機）
* 探索是完全隨機的，不考慮臂的不確定性
* 會浪費樣本在明顯次優的臂上（如臂 0）
* 即使後期 ε 很小，仍會偶爾選擇次優臂
* 沒有利用置信區間或後驗分布的資訊

**理論支持：**
* 取決於 ε 的衰減速率
* ε = 1/t 時，遺憾界為 $O(T^(2/3))$，比 $O(log T)$ 差


---

**累積遺憾分析**
$累積遺憾 (Cumulative Regret) = Σ(最優獎勵 - 實際獎勵)$
> Thompson Sampling:  8.96
> UCB:                18.31
> Epsilon-Greedy:     21.20

* Thompson Sampling 的遺憾最小，表示選擇次優臂的次數最少
* Epsilon-Greedy 的遺憾最大，表示浪費了較多樣本在探索上


---

**收斂速度分析**
達到平均獎勵 0.65 所需的迭代次數：
> Thompson Sampling:  45 次迭代
> UCB:                181 次迭代
> Epsilon-Greedy:     200 次迭代

* Thompson Sampling 收斂速度最快
* Epsilon-Greedy 收斂速度最慢


---

**結果是否令人意外？**
答：結果不令人意外，完全符合理論預期。

1. Thompson Sampling 表現最好：
* 對於 Bernoulli bandit，Thompson Sampling 在理論和實踐中都被證明最優
* Beta-Bernoulli 共軛使得後驗更新最有效
* 概率匹配策略自然地平衡了探索和利用
* 符合理論：有漸進最優的遺憾界 $O(log T)$
2. UCB 表現次之：
* UCB 也有 $O(log T)$ 的理論遺憾界
* 但在實際應用中，UCB 的常數因子通常比 Thompson Sampling 大
* UCB 對所有臂一視同仁地探索，而 Thompson Sampling 更智能
* 符合理論：性能接近 Thompson Sampling
3. Epsilon-Greedy 表現最差：
* 簡單的隨機探索策略，沒有利用不確定性資訊
* ε = 1/t 的衰減速率導致 O(T^(2/3)) 的遺憾
* 初期 ε=1 意味著完全隨機，浪費很多樣本
* 符合理論：遺憾界比 $O(log T)$ 差

## Assignment2 Part2
### Cartpole Environment
圖表說明
* X軸: Episode 數量（REINFORCE 和 REINFORCE with Baseline: 0-800，PPO: 0-150）
* Y軸: 累積獎勵（平均最近 25 個 episodes）
![image](https://hackmd.io/_uploads/HyXXNE8JWe.png)
![image](https://hackmd.io/_uploads/ryJVENIyZe.png)
![image](https://hackmd.io/_uploads/B1u4NNLJbx.png)

1. REINFORCE
* 學習曲線波動較大，variance 高
* 收斂速度較慢，需要較多 episodes 才能穩定
* 最終性能可以達到接近最優，但過程不穩定


2. REINFORCE with Baseline
* 學習曲線比純 REINFORCE 更穩定
* 收斂速度明顯加快
* Variance 降低，訓練更加可靠


3. PPO
* 學習曲線最穩定且最快收斂
* 只需要 150 個 episodes 就能達到良好性能
* 曲線平滑，幾乎沒有大幅波動

| 演算法 | 優點 | 缺點 | 原理 |
| -------- | -------- | -------- | -------- |
| REINFORCE     | • 簡單直觀<br>• 無偏估計<br>• 理論保證收斂     |• 高 variance<br>• 樣本效率低<br>• 學習不穩定     | 使用 Monte Carlo returns 作為梯度估計，完全依賴實際觀察到的回報     |
| REINFORCE with Baseline     | • 降低 variance<br>• 加快收斂<br>• 仍保持無偏性     | • 需要額外訓練 value function<br>• 增加計算複雜度     | 使用 advantage function ($A = R - V(s)$) 代替 return，減少梯度估計的變異性     |
| PPO    | • 降低 variance<br>• 加快收斂<br>• 仍保持無偏性     | • 實作較複雜<br>• 有額外的超參數     | 使用 clipped objective 限制 policy 更新幅度，並透過多次迭代重複使用數據     |

**為什麼 PPO 表現最好？**
* Clipping 機制: 限制 importance sampling ratio 在 [1-ε, 1+ε] 範圍內，避免 policy 有太大的改變
* 多次迭代: POLICY_TRAIN_ITERS=10 讓同一批數據被重複使用，提高樣本效率
* 保守更新: 取 min(clipped, unclipped) 確保更新是保守的，不會因為一次壞的 batch 就破壞已學到的策略


---

**Bonus 問題 : POLICY_TRAIN_ITERS 從 1 改為 10**
在 REINFORCE_Baseline.py 中修改：
```
POLICY_TRAIN_ITERS = 10
```
![image](https://hackmd.io/_uploads/Hy3GqV8k-e.png)

1. 初期（前 100-200 episodes）:
* 學習速度可能稍微加快
* 每個 episode 的數據被更充分利用

2. 中後期（200+ episodes）:
* 可能出現訓練不穩定
* 性能可能反覆波動
* 有時候甚至可能比 POLICY_TRAIN_ITERS=1 還差

**為什麼會不穩定？**

1. On-Policy vs Off-Policy 問題:
>    REINFORCE with Baseline 是 on-policy 演算法
>    → 使用的數據必須來自當前的 policy
>    → 多次迭代後，policy 已經改變
>    → 但仍在使用舊 policy 收集的數據
>    → 導致 policy-data mismatch

2. 缺乏保護機制:
* REINFORCE 沒有 PPO 的 clipping 機制
* 多次迭代可能導致過大的 policy 更新
* 可能會 "忘記" 之前學到的好策略


3. PPO 為什麼可以多次迭代？:
* Clipped objective: 限制更新幅度，防止 policy 偏離太遠
* Importance sampling: 明確考慮新舊 policy 的差異
* Early stopping: 實務上可監控 KL divergence，太大就停止

**數學直覺:**
> REINFORCE 的更新: ∇θ J(θ) = E[∇θ log π(a|s) * R]
> 
> 多次迭代等於:
> - 第1次: 使用 policy π_old 的數據
> - 第2次: policy 已變成 π_1，但仍用 π_old 的數據
> - ...
> - 第10次: policy 已變成 π_9，但仍用 π_old 的數據
> 
> → 數據與 policy 的 mismatch 越來越大


---

### Mountain Car Environment
**圖表說明**
X軸: Episode 數量（同上）
Y軸: 累積獎勵（平均最近 25 個 episodes）
![image](https://hackmd.io/_uploads/SkEE24Ik-x.png)
![image](https://hackmd.io/_uploads/ByjVnEU1Ze.png)
![image](https://hackmd.io/_uploads/r1fr2VIkZl.png)

* 所有三個演算法在 Mountain Car 上的表現都會顯著變差
* 學習曲線可能長期停留在 -200 左右（接近最差情況）
* 很少或幾乎沒有成功到達山頂的 episodes
* PPO 可能稍好一些，但整體仍然困難

---

**為什麼 Mountain Car 這麼難？**
**1. Sparse Reward 問題**

| 環境 | Reward 結構 | 學習信號 |
| -------- | -------- | -------- |
| CartPole     | 每個時間步 +1<br>倒下時 episode 結束     | • Dense reward<br>• 持續的 feedback<br>• 容易看出哪些動作好     |
| Mountain Car     | 每個時間步 -1<br>到達目標才是 0<br>200 步後強制結束    | • Sparse reward<br>• 幾乎沒有正向信號<br>• 難以判斷動作的好壞     |

具體影響：
> CartPole: 
> Episode 1: 獎勵 = 20  → "還不錯，繼續學習類似的動作"
> Episode 2: 獎勵 = 35  → "更好了！"
> Episode 3: 獎勵 = 15  → "這次較差，調整策略"
> 
> Mountain Car:
> Episode 1: 獎勵 = -200  → "最差了"
> Episode 2: 獎勵 = -200  → "還是最差"
> Episode 3: 獎勵 = -200  → "完全沒有進步的信號"
> ...
> Episode 500: 獎勵 = -200  → "依然沒有任何有用的信息"

**2. 探索困難 (Exploration Problem)**

Mountain Car 需要：
* 反直覺的動作序列：必須先往後退來獲得動能
* 長期規劃：單一動作的效果很小，需要連續多個動作配合
* 隨機探索幾乎無效：隨機 policy 幾乎不可能偶然到達山頂

**3. Credit Assignment 困難**

在 Mountain Car 中：
* 哪些動作真正有幫助？很難判斷
* 早期的"往後退"動作看起來是壞的（高度降低）
* 但這些動作對最終成功是必要的
* REINFORCE 難以建立這種長期的因果關係

**4. 環境參數對比**

| 特性 | CartPole | Mountain Car |
| -------- | -------- | -------- |
| State space     | 4 維（位置、速度、角度、角速度）     | 2 維（位置、速度）     |
| Action space     | 2 個動作（左、右）     | 3 個動作（左、不動、右）     |
| Episode 長度     | 可變（倒下即結束）     | 固定 200 步     |
| 成功條件     | 維持平衡     | 到達特定位置     |
| Reward 密度     | 非常密集（每步 +1）     | 極度稀疏（幾乎都是 -1）     |
| 探索需求     | 低（隨機動作也能學）     | 高（需要特定動作序列）     |

---

### Modified Mountain Car Environment
**圖表說明**
X軸: Episode 數量
Y軸: 累積獎勵（平均最近 25 個 episodes）
![image](https://hackmd.io/_uploads/ryXGC48yWl.png)
![image](https://hackmd.io/_uploads/Hk_MA4Ly-e.png)
![image](https://hackmd.io/_uploads/B12MA4LkZe.png)

**Reward Function 差異**

| 版本 | Reward 定義 | 特性 |
| -------- | -------- | -------- |
| 原版 Mountain Car     | 每步 -1<br>到達目標 0     | • Sparse<br>• 無引導性<br>• 只有成功/失敗     |
| Modified Mountain Car    | 每步的 reward = 車子的高度<br>高度範圍: -1.2 到 0.6     | • Dense<br>• 有引導性<br>• 提供持續 feedback     |

---

**為什麼 Modified 版本更容易學習？**
**1. 引導性學習信號 (Guided Learning Signal)**

原版的問題：
> 狀態 A: 高度 = 0.3，Reward = -1
> 狀態 B: 高度 = 0.4，Reward = -1
> 狀態 C: 高度 = 0.2，Reward = -1
> 
> → 演算法無法區分這些狀態的好壞

Modified 版本：
> 狀態 A: 高度 = 0.3，Reward = 0.3  ✓ 好！
> 狀態 B: 高度 = 0.4，Reward = 0.4  ✓ 更好！
> 狀態 C: 高度 = 0.2，Reward = 0.2  ✗ 不太好
> 
> → 清楚的梯度信號，引導往正確方向

**2. 即時 Feedback (Immediate Feedback)**

| 時機 | 原版 | Modified |
| -------- | -------- | -------- |
| 動作後     | 得到 -1（無信息）     | 得到新高度（有信息）     |
| Episode 中間     | 無法判斷進展     | 可以看到高度變化    |
| Episode 結束     | 只知道失敗    | 知道最高到達哪裡    |

**3. 降低 Credit Assignment 難度**

原版的 Credit Assignment：
> "我在 200 步後失敗了，但不知道是哪些步驟出錯"
> → 需要從整個 episode 反推
> → 非常困難

Modified 版本的 Credit Assignment：
> Step 50: 往左 → 高度從 0.2 降到 0.1 → 立即知道這不好
> Step 51: 往右 → 高度從 0.1 升到 0.15 → 立即知道這較好
> → 每步都有清楚的 feedback

**4. 減少 Variance**

**梯度估計的 variance：**
原版：
* 大部分 episodes 都是 -200（相同的壞結果）
* 偶爾一次成功 -100（但極少發生）
* 高 variance，難以學習

Modified：
* 每個 episode 都有不同的總高度
* 持續的進步信號
* 低 variance，容易學習

**5. 數學角度：Reward Shaping 理論**

Modified 版本本質上是一種 Reward Shaping：
原始 reward: $R(s, a, s')$
Shaped reward: $R'(s, a, s') = R(s, a, s') + F(s, s')$
其中 $F(s, s') = Φ(s') - Φ(s)$（potential-based shaping）
在我們的例子中：
* $Φ(s) = height(s)$（狀態的高度）
* 這保證了最優策略不變！








