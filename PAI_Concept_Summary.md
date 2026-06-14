# PAI 觀念重點整理 (Principles of Artificial Intelligence Concept Summary)

本複習指南針對人工智慧與機器學習的核心觀念進行深度梳理，涵蓋統計學習理論、古典機器學習演算法、深度學習架構、生成模型、多臂老虎機（MAB）以及馬可夫決策過程（MDP）與強化學習。

---

## II. Machine Learning (機器學習基礎與理論)

### Classification of learning (學習系統的分類)

1. **Supervised vs Unsupervised Learning (監督式與非監督式學習)**
   * **監督式學習 (Supervised Learning)**：訓練資料包含輸入特徵 $x$ 與對應的標籤 (Label) $y$，即資料集為 $S = \{(x_i, y_i)\}_{i=1}^m$。模型旨在學習一個映射函數 $f: X \to Y$，使其能對未知輸入預測準確的標籤。常見任務包括分類 (Classification) 與迴歸 (Regression)。
   * **非監督式學習 (Unsupervised Learning)**：訓練資料僅包含輸入特徵 $x$，即 $S = \{x_i\}_{i=1}^m$。演算法的目的在於探索資料內部的潛在結構、分佈或特徵表示。常見任務包括集群 (Clustering, 如 K-Means)、降維 (Dimensionality Reduction, 如 PCA) 與密度估計。

2. **Active vs Passive Learning (主動式與被動式學習)**
   * **被動式學習 (Passive Learning)**：模型僅能全盤接受環境或資料集預先提供好的樣本進行訓練，無法自主選擇要觀察哪些資料。
   * **主動式學習 (Active Learning)**：演算法在訓練過程中具有互動能力。它可以從大量的未標記資料中，挑選出「對提升模型性能最有效」的樣本（例如不確定性最高的樣本），並向領域專家 (Oracle) 請求標記。這能以極少的標記成本達到高準確度。

3. **Statistical vs Adversarial Learning (統計學習與對抗式學習)**
   * **統計學習 (Statistical Learning)**：基於傳統統計學假設，認為訓練資料與測試資料均獨立同分佈（i.i.d.）於某個固定但未知的機率分佈 $\mathcal{D}$。目標是極小化在該分佈下的期望風險。
   * **對抗式學習 (Adversarial Learning)**：考慮惡意環境下的學習。假設存在一個對抗者 (Adversary)，會蓄意設計微小的擾動（對抗樣本）來欺騙模型，或者動態改變資料分佈。此框架關注模型的魯棒性 (Robustness) 與防禦能力。

4. **Online vs Batch Learning (線上學習與批次學習)**
   * **批次學習 (Batch / Offline Learning)**：模型在訓練時必須一次性讀入所有的訓練數據。若未來有新數據加入，通常需要將舊數據與新數據合併，重新訓練整個模型。
   * **線上學習 (Online Learning)**：數據以串流 (Stream) 形式逐個或逐小批次抵達。模型每接收到一個新樣本，就進行一次權重更新，隨後即可丟棄該樣本。適用於數據量巨大、內存有限或資料動態隨時間變化的場景。

### Statistical Learning Framework (統計學習框架)

統計學習理論為機器學習提供了數學基礎，其核心組成包含：
* **定義網域 (Domain Set)**：所有可能輸入特徵的集合 $X$。
* **標籤集合 (Label Set)**：所有可能輸出的集合 $Y$（如二元分類中 $Y = \{0, 1\}$）。
* **數據生成分佈 (Data Generation Distribution)**：定義在 $X \times Y$ 上的未知機率分佈 $\mathcal{D}$。
* **假設空間 (Hypothesis Class)**：由所有可能的預測函數 $h: X \to Y$ 組成的集合 $\mathcal{H}$。
* **損失函數 (Loss Function)**：衡量預測值與真實值差異的函數 $\ell(h(x), y)$。
* **真實風險 (True Risk / Generalization Error)**：模型在整個分佈 $\mathcal{D}$ 上的期望損失：
  $$L_{\mathcal{D}}(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}} [\ell(h(x), y)]$$

### Empirical Risk Minimization (ERM, 經驗風險最小化)

由於真實分佈 $\mathcal{D}$ 是一般無法觀測的，我們無法直接計算並極小化真實風險 $L_{\mathcal{D}}(h)$。因此，我們轉而利用手頭上的有限訓練樣本 $S = \{(x_1, y_1), \dots, (x_m, y_m)\}$，計算其平均損失，稱為**經驗風險 (Empirical Risk)**：
$$L_S(h) = \frac{1}{m} \sum_{i=1}^m \ell(h(x_i), y_i)$$
**ERM 策略** 即是在假設空間 $\mathcal{H}$ 中尋找一個使經驗風險最小化的假設：
$$h_S = \arg\min_{h \in \mathcal{H}} L_S(h)$$
* **過擬合 (Overfitting) 防治**：若 $\mathcal{H}$ 複雜度過高，ERM 極易找到一個在訓練集上完美（$L_S(h)=0$）但在測試集上表現極差的函數。為解決此問題，通常會限制 $\mathcal{H}$ 的複雜度（如使用較簡單的模型），或引入結構風險最小化 (SRM)，在損失函數中加入正則化項（Regularization）。

### Probability Approximately Correct (PAC) Learning (機率近似正確學習)

PAC 學習理論定義了什麼是「可學習的」(Learnable)。一個假設空間 $\mathcal{H}$ 被稱為是 PAC 可學習的，若存在一個演算法與一個多項式函數 $m(\epsilon, \delta)$，使得對於任意分佈 $\mathcal{D}$ 和任意真實目標函數，只要樣本量 $m \ge m(\epsilon, \delta)$，演算法輸出之假設 $h_S$ 在滿足以下條件的機率至少為 $1 - \delta$（**Approximately Correct 的信心度**）：
$$L_{\mathcal{D}}(h_S) \le \epsilon \quad \text{(Approximately Correct, 誤差範圍內)}$$
* $\epsilon$：精確度參數 (Accuracy parameter)，代表容許的泛化誤差。
* $\delta$：信心度參數 (Confidence parameter)，代表演算法失敗（抽到極端壞樣本）的機率。
* *註：使用者標題提到的 "PCA Learning" 在經典理論脈絡下通常為 PAC Learning 之誤植，此處聚焦於 PAC 框架。*

### Agnostic PAC Learning (不可知 PAC 學習)

經典 PAC 學習假設了「可實現性 (Realizability)」，即假設真實的目標映射函數必然存在於 $\mathcal{H}$ 中。然而在現實中，這幾乎不可能成立（存在噪聲或模型錯置）。
**不可知 PAC 學習 (Agnostic PAC Learning)** 放寬了此假設：不要求真實風險能達到 $0$。其目標轉化為：尋找一個假設 $h_S$，使其真實風險與 $\mathcal{H}$ 中**最優假設**的真實風險相比，差距不超過 $\epsilon$：
$$L_{\mathcal{D}}(h_S) \le \min_{h' \in \mathcal{H}} L_{\mathcal{D}}(h') + \epsilon$$
同樣要求該結論在 $1-\delta$ 的機率下成立。這使得 PAC 理論能應用於包含噪聲及非線性結構的現實資料中。

---

## III. Classical Learning Algorithms (古典學習演算法)

### Linear Predictors (線性預測器)

線性預測器是一類通過對輸入特徵進行線性組合來做出預測的模型，形式為 $f(x) = \langle w, x \rangle + b$。

#### Linear Programming (線性規劃)
在機器學習中（如線性可分 SVM 或結構化預測），線性規劃 (LP) 被用作優化工具。當數據是嚴格線性可分時，尋找分離超平面的問題可以轉化為一組線性不等式約束，並通過單體法 (Simplex Method) 或內點法 (Interior Point Method) 在多項式時間內求解出最優權重 $w$。

#### Perceptron Algorithm (感知器演算法)
感知器是一種古典的二元線性分類線上演算法。其決策邊界為 $\text{sign}(w^T x)$。
* **更新機制**：逐一檢查樣本，若模型對樣本 $(x_i, y_i)$ 預測錯誤（即 $y_i(w^T x_i) \le 0$），則依據下式修正權重：
  $$w \leftarrow w + y_i x_i$$
* **收斂性 (Novikoff 定理)**：若數據集是線性可分的，且存在一個正邊界 (Margin) $\gamma$，則感知器演算法必能在有限次更新（上限為 $(R/\gamma)^2$，其中 $R$ 為輸入向量的最大模長）內收斂。若數據線性不可分，則演算法會陷入無限循環。

### Linear Regression (線性迴歸)

線性迴歸用於預測連續型目標變數。

#### Least Squares Algorithm (最小平方法)
最小平方法旨在尋找參數 $w$，使得預測值與真實值之間的殘差平方和 (RSS) 達到最小：
$$\min_w L(w) = \|Xw - y\|_2^2$$
* **解析解 (Normal Equation)**：若矩陣 $X^T X$ 可逆，可通過將梯度設為 0 直接求得閉式解：
  $$w = (X^T X)^{-1} X^T y$$
* 當特徵數極大或 $X^T X$ 不可逆（共線性問題）時，需引入脊迴歸 (Ridge, L2) 或 Lasso (L1) 正則化。

### Logistic Regression (羅吉斯迴歸)

羅吉斯迴歸雖名為迴歸，但實質上用於二元分類。它將線性預測器的輸出通過 Sigmoid 函數 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 映射至 $(0,1)$ 區間，解釋為機率值：
$$P(y=1|x) = \sigma(w^T x)$$

#### Maximum Likelihood Estimator (MLE, 最大概似估計)
羅吉斯迴歸無法推導出閉式解，其參數是透過最大化觀測數據的聯合機率（概似函數）來學習的。在二元分類中，最大化概似函數等價於最小化**交叉熵損失函數 (Cross-Entropy Loss / Negative Log-Likelihood)**：
$$L(w) = -\frac{1}{m} \sum_{i=1}^m \left[ y_i \ln(\sigma(w^T x_i)) + (1 - y_i) \ln(1 - \sigma(w^T x_i)) \right]$$
此函數為凸函數 (Convex function)，一般採用梯度下降法或牛頓法求得全域最優解。

### Stochastic Gradient Descent (SGD, 隨機梯度下降)

#### Subgradient (次梯度)
當優化目標函數不可微（例如包含 L1 正則化 $|x|$ 或 ReLU 激活函數 $\max(0,x)$）時，傳統梯度不適用。此時需引入次梯度。對於凸函數 $f$，若向量 $g$ 滿足對於所有 $y$ 都有：
$$f(y) \ge f(x) + g^T(y - x)$$
則稱 $g$ 為 $f$ 在 $x$ 處的一個次梯度。次梯度集合稱為次微分 $\partial f(x)$。次梯度下降法利用此向量引導參數更新。

#### Batch GD vs SGD (批次梯度下降 vs 隨機梯度下降)
* **批次梯度下降 (Batch GD)**：每一次參數更新都需計算整個數據集上所有樣本的梯度平均值。更新方向精確，但單步計算複雜度高，難以應對海量數據。
* **隨機梯度下降 (SGD)**：每一次更新僅隨機挑選**一個**樣本（或一小批次 Mini-batch）來計算梯度並更新。
  * **數學更新式**：$w \leftarrow w - \eta \nabla \ell(h(x_i), y_i)$
  * **優缺點**：計算速度極快，且梯度的隨機噪聲有助於模型跳出局部極小值（Local Minima）或鞍點，但收斂路徑呈震盪狀，後期需逐漸調小學習率 $\eta$。

### Lipschitz Continuity (利普希茨連續性)

若一個函數 $f: \mathbb{R}^n \to \mathbb{R}$ 滿足對任意 $x, y$，皆存在一個常數 $L \ge 0$ 使得：
$$\|f(x) - f(y)\| \le L \|x - y\|$$
則稱 $f$ 為 **$L$-利普希茨連續**。
* **在 ML 中的意義**：它限制了函數變化的劇烈程度。在優化理論中，梯度滿足 Lipschitz 連續性是保證梯度下降法能穩定收斂的重要條件；在深度學習中，限制網絡權重的 Lipschitz 常數（如 WGAN 中的 Weight Clipping 或 Spectral Normalization）能有效防止梯度爆炸，並提升模型的泛化能力與抗干擾能力。

### Federated Learning (聯邦學習)

聯邦學習是一種分佈式機器學習框架，核心理念為**「數據不動，模型動」**，旨在保護用戶隱私。
* **運作流程**：
  1. 中央伺服器分發當前全域模型 (Global Model) 給各個本地客戶端 (Clients)。
  2. 客戶端利用本地擁有的私有數據訓練模型，計算出模型更新量（梯度或權重變更）。
  3. 客戶端僅將模型更新量上傳回伺服器，原始數據不出本地。
  4. 伺服器進行聚合（如 **FedAvg 演算法**：按各客戶端數據量進行加權平均），更新全域模型。
* **挑戰**：通訊瓶頸、非獨立同分佈 (Non-IID) 的數據分佈、系統異構性與安全隱私攻擊（如梯度洩漏）。

---

## IV. Deep Learning (深度學習)

### Multilayer Perceptron (MLP, 多層感知器)

多層感知器是一種前饋神經網路 (Feedforward Neural Network)。它由一個輸入層、一個或多個隱藏層 (Hidden Layers) 以及一個輸出層組成，層與層之間全連接 (Fully Connected)。
* **非線性之必要性**：每一層的運算為 $h = \sigma(W x + b)$。若不加入非線性激活函數 $\sigma$，多個線性層的複合在數學上仍等價於單一線性變換（$W_2(W_1 x + b_1) + b_2 = W_{new} x + b_{new}$），這將使網絡失去逼近複雜非線性函數的能力。

### Rectified Linear Unit (ReLU, 反流線性單元)

ReLU 是現代深度學習中最廣泛應用的激活函數，數學定義為：
$$f(x) = \max(0, x)$$
* **優點**：在 $x>0$ 時梯度恆為 1，有效緩解了深層網路中的**梯度消失 (Vanishing Gradient)** 問題；計算極其簡單（僅需閾值判斷），大幅加快訓練速度；會使部分神經元輸出為 0，為網路帶來**稀疏表達能力 (Sparsity)**。
* **缺點**：**Dying ReLU 問題**。當一個很大的梯度流過神經元，導致權重更新後使得該神經元在所有數據上的輸入都小於 0，則該神經元輸出與梯度將永遠為 0，形同「壞死」。

### Artificial Neural Networks (ANN, 人工神經網路)

人工神經網路是一門模仿生物大腦神經元網路結構的計算模型。它透過大量的節點（神經元）相互連接，藉由調整連接的權重 (Weights) 與偏置 (Biases) 來學習複雜的輸入-輸出映射關係。**通用近似定理 (Universal Approximation Theorem)** 指出，包含至少一個隱藏層及合適激活函數的前饋網路，能夠以任意精度逼近任何閉區間上的連續函數。

### Backpropagation (反向傳播演算法)

反向傳播是訓練神經網路的核心演算法，用於高效計算損失函數對網路中所有參數的偏導數。
* **核心原理**：基於微積分的**連鎖律 (Chain Rule)**。
* **流程**：
  1. **前向傳播 (Forward Pass)**：輸入數據通過各層計算，最終在輸出層得到預測值，並計算出損失值 (Loss)。
  2. **反向傳播 (Backward Pass)**：從輸出層開始，將損失誤差向後傳遞，依次計算損失對每一層權重 $W$ 和偏置 $b$ 的梯度。
  3. **參數更新**：利用優化器（如 SGD、Adam）沿著梯度相反方向更新權重。

### Improving NN (神經網路的最佳化與優化技術)

1. **Weight Initialization (權重初始化)**
   * 若初始權重過大，會導致激活值飽和，引發梯度爆炸；若過小，信號逐層衰減，引發梯度消失。
   * **Xavier / Glorot 初始化**：適用於 Tanh/Sigmoid。根據輸入輸出節點數決定方差，維持前後層信號方差一致。
   * **He / Kaiming 初始化**：適用於 ReLU。方差設為 $\frac{2}{\text{fan\_in}}$，專為應對 ReLU 在負半軸斷流的特性設計。

2. **Unstable Gradient (不穩定的梯度)**
   * **梯度消失 (Vanishing Gradient)**：深層網路中，連鎖律中連續相乘多個小於 1 的導數（如 Sigmoid 的導數最大僅 0.25），導致低層網路的梯度趨近於 0，參數無法更新。
   * **梯度爆炸 (Exploding Gradient)**：連鎖律中多個大於 1 的矩陣特徵值相乘，導致梯度呈指數級增長，數值溢出（NaN）。可用**梯度裁剪 (Gradient Clipping)** 限制梯度的最大模長。

3. **BatchNorm (Batch Normalization, 批次正規化)**
   * 在神經網路的各層之間，對每個 Mini-batch 的激活值進行規範化處理，使其保持均值為 0、方差為 1 的分佈。
   * **數學式**：$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$，隨後通過兩個可學習參數進行縮放與平移：$y = \gamma \hat{x} + \beta$。
   * **功效**：減緩了內部協變量轉移 (Internal Covariate Shift)；允許使用更大的學習率；具備輕微的正則化效果，可加速網路收斂。

4. **Dropout (隨機失活)**
   * 在每次訓練的前向傳播中，以機率 $p$ 隨機將一部分隱藏層神經元的輸出設為 0。
   * **機理**：迫使網路不能依賴於特定的神經元組合，必須學習到更具魯棒性的特徵，有效防止共適應 (Co-adaptation) 現象，是一種強大的**防止過擬合 (Regularization)** 技術。測試時，所有神經元均保持激活，但輸出需乘以 $(1-p)$ 以平衡尺度。

### Convolutional NN (CNN, 卷積神經網路)

專為處理具有網格結構數據（如圖像）而設計的架構。
* **核心特點**：
  * **局部感受野 (Local Receptive Fields)**：每個神經元只與前一層的局部區域連接，專注於捕捉局部特徵（如邊緣、紋理）。
  * **權重共享 (Shared Weights)**：在整個輸入圖像上移動同一個卷積核 (Kernel)，大幅減少了網絡參數。
  * **平移不變性 (Translation Invariance)**：不論特徵出現在圖像的哪個位置，卷積核都能將其捕捉。
* **架構**：交替使用卷積層 (Convolutional Layer) 與池化層 (Pooling Layer, 如 Max Pooling，用以進行空間降維與增強魯棒性)，最後連接全連接層輸出。

### Recurrent NN and Natural Language Processing (RNN 與自然語言處理)

設計用於處理序列數據（如文本、時間序列）的網絡。
* **核心機制**：引入**循環連接**，神經元不僅接受當前輸入 $x_t$，還接受前一時刻的隱藏狀態 $h_{t-1}$，形成內部記憶。
* **長程依賴問題**：傳統 RNN 在處理長序列時，由於時間步長的反向傳播 (BPTT)，梯度需要連續乘以相同的權重矩陣，極易導致梯度消失或爆炸。
* **解決方案 (LSTM & GRU)**：
  * **LSTM (長短期記憶網路)**：引入「細胞狀態 (Cell State)」與三個門控結構——**遺忘門 (Forget Gate)**、**輸入門 (Input Gate)** 和 **輸出門 (Output Gate)**，精細控制資訊的遺忘與保留。
  * **GRU (門控循環單元)**：簡化版 LSTM，將細胞狀態與隱藏狀態合併，僅包含**更新門 (Update Gate)** 與 **重置門 (Reset Gate)**，計算效率更高。

---

## V. Generative Models (生成模型)

### Auto-Encoder (AE, 自編碼器)

自編碼器是一種非監督式學習架構，旨在學習數據的低維緊湊表示。
* **架構**：由**編碼器 (Encoder, $f$)** 與**解碼器 (Decoder, $g$)** 組成。
  * Encoder 將高維輸入 $x$ 壓縮為低維瓶頸空間 (Bottleneck) 的潛在向量 $z = f(x)$。
  * Decoder 將 $z$ 還原為重構數據 $\hat{x} = g(z)$。
* **目標**：極小化重構誤差 $\|x - \hat{x}\|^2$。被迫丟棄次要噪聲，保留核心特徵。

### Denoising Auto-Encoder (DAE, 去噪自編碼器)

為了防止自編碼器只是單純記住輸入（學到恆等映射），DAE 在輸入端引入干擾。
* **機制**：首先將原始乾淨數據 $x$ 蓄意加入噪聲（如高斯噪聲或隨機擦除）變成 $\tilde{x}$。將 $\tilde{x}$ 輸入網路，但訓練的優化目標仍然是要求解碼器重構出**原始乾淨的 $x$**。
* **意義**：迫使網路學習如何消除噪聲，使模型能捕捉數據流形 (Data Manifold) 的內在幾何結構，大幅提升特徵的魯棒性。

### Sparse Auto-Encoder (SAE, 稀疏自編碼器)

SAE 在損失函數中引入稀疏性約束，要求在同一時間內，隱藏層中只有極少數的神經元被激活。
* **實現方式**：在原本的重構損失中，額外加上一個**稀疏性處罰項**。常用的手法是計算隱藏層神經元的平均激活度，並利用 **KL 散度 (Kullback-Leibler Divergence)** 來懲罰該平均激活度與一個接近於 0 的預設稀疏目標值 $
ho$ 之間的差距。這迫使網絡用最具代表性的少數特徵來編碼數據。

### k-Sparse Auto-Encoder (k-稀疏自編碼器)

與 SAE 採用軟性處罰項不同，k-Sparse AE 採用硬性限制。
* **機制**：在前向傳播計算出隱藏層的所有激活值後，演算法會對這些值進行排序，**只保留前 $k$ 個最大（最活躍）的值**，並強行將其餘的所有激活值**直接設為 0**。隨後，僅將這 $k$ 個非零值傳遞給解碼器進行重構。這保證了編碼的絕對稀疏度。

### Variational Auto-Encoder (VAE, 變分自編碼器)

VAE 是一種將自編碼器延伸為**機率生成模型**的經典架構。
* **核心問題**：傳統 AE 的潛在空間 $z$ 分佈是離散且無序的，無法隨機採樣生成有意義的新數據。
* **VAE 機制**：Encoder 不再輸出確定的向量 $z$，而是輸出潛在變數分佈的統計參數：**均值向量 $\mu$** 與 **方差對數 $\log(\sigma^2)$**。
* **優化目標 (ELBO)**：
  $$L_{VAE} = \text{Reconstruction Loss} + D_{KL}(N(\mu, \sigma^2) \parallel N(0, I))$$
  KL 散度項強迫潛在空間逼近標準高斯分佈，使其具備連續性與規整性。
* **重參數化技巧 (Reparameterization Trick)**：為了解決從 $N(\mu, \sigma^2)$ 採樣這一隨機操作無法導出梯度的問題，VAE 將隨機性移出計算圖：先從標準高斯分佈採樣 $\epsilon \sim N(0, I)$，再令：
  $$z = \mu + \sigma \odot \epsilon$$
  這樣梯度便可順利反向傳播回 Encoder。

### Generative Adversarial Network (GAN, 對抗生成網路)

基於賽局理論中「零和博弈 (Zero-sum game)」的生成模型。
* **組成架構**：
  * **生成器 (Generator, $G$)**：輸入隨機噪聲 $z$，試圖生成逼真的虛假數據 $G(z)$，目的是「欺騙 $D$」。
  * **判別器 (Discriminator, $D$)**：輸入真實數據 $x$ 或生成數據 $G(z)$，輸出一個機率值，目的是「分辨真偽」。
* **博弈目標函數**：
  $$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$
* **收斂狀態**：當達到納許均衡 (Nash Equilibrium) 時，$G$ 生成的數據完美符合真實分佈，$D$ 的判別準確度恰好為 $0.5$（完全無法分辨）。

### Denoising Diffusion Probabilistic Model (DDPM, 去噪擴散機率模型)

現代最頂尖的生成模型之一，其架構分為兩個方向：
* **前向擴散過程 (Forward Process)**：在多個時間步長（如 $T=1000$）中，逐步向真實圖像添加微小的參數化高斯噪聲，直到圖像徹底變成純高斯噪聲。此過程是確定的，無須訓練。
* **反向去噪過程 (Reverse Process)**：訓練一個神經網路（通常為 U-Net），輸入帶噪圖像與當前時間步 $t$，去**預測並扣除**在前向過程中該步所添加的噪聲。
* **特點**：數學結構非常穩健，避免了 GAN 的模式崩塌 (Mode Collapse) 問題，生成質量極高，但缺點是採樣時必須模擬整條馬可夫鏈，逐步迭代數百到上千步，生成速度緩慢。

### Denoising Diffusion Implicit Model (DDIM, 去噪擴散隱式模型)

為了克服 DDPM 生成速度過慢的致命缺點，DDIM 被提出。
* **核心創新**：DDIM 重新推導了擴散模型的數學基礎，設計了一種**非馬可夫鏈 (Non-Markovian)** 的前向邊緣分佈。這使得反向去噪過程不再具備隨機性，而是變成一個**確定性的軌跡**。
* **優勢**：由於是確定性軌跡，採樣時可以**跳步 (Sub-sampling)**。原本 DDPM 需要 1000 步的採樣，DDIM 僅需 20 到 50 步便能達到甚至超越同等質量的生成效果，使擴散模型步入實用階段。最重要的是，它**不需要重新訓練模型**，可直接套用 DDPM 訓練好的權重。

---

## VI. Multi-Armed Bandit (MAB, 多臂老虎機)

多臂老虎機模型刻劃了決策中**「探索與學習 (Exploration)」**與**「利用現有知識 (Exploitation)」**之間的權衡。目標是最大化累積獎勵，或等價於極小化累積遺憾 (Cumulative Regret)。

### Greedy Algorithm (貪婪演算法)
* **策略**：完全不進行任何探索。在每一步中，只看當前各個拉桿 (Arms) 的經驗平均獎勵 $\hat{\mu}_i$，並永遠選擇平均值最高的那根拉桿。
* **缺陷**：極易陷入局部最優。如果最優的拉桿在開局前幾次嘗試中運氣不佳得到了較低的隨機獎勵，貪婪演算法將永遠封殺它，再也沒有機會發現它的真實高回報。

### $\varepsilon$-Greedy Algorithm ($\varepsilon$-貪婪演算法)
* **策略**：以一個固定的機率 $\varepsilon \in (0, 1)$ 隨機在所有拉桿中挑選一根拉動（強制探索）；以 $1-\varepsilon$ 的機率選擇當前經驗平均回報最高的拉桿（利用）。
* **優化**：$\varepsilon$ 可以隨著時間步長 $t$ 的增長而逐漸衰減（如 $\varepsilon_t = 1/t$），從而在前期著重探索，後期專注利用。

### Explore-Then-Commit Algorithm (ETC, 先探索後承諾演算法)
* **策略**：將整個時間區間嚴格劃分為兩個階段：
  1. **探索階段**：對 $K$ 個拉桿中的每一個，均勻地各拉動 $m$ 次（總共耗費 $mK$ 步）。
  2. **承諾階段**：結算這 $mK$ 步的數據，找出經驗平均獎勵最高的拉桿，並在剩餘的所有時間步中，毫無保留地永遠拉動這一根拉桿。
* **缺點**：參數 $m$ 的選定極度依賴對獎勵差距的先驗知識，選太大浪費時間在壞拉桿上，選太小可能在第一階段挑錯拉桿。

### Sub-Gaussianity (亞高斯性)

一個隨機變數 $X$（假設期望 $\mathbb{E}[X]=0$）被稱為是 $\sigma$-亞高斯的，若其動差生成函數滿足：
$$\mathbb{E}[e^{\lambda X}] \le e^{\frac{\lambda^2 \sigma^2}{2}}, \quad \forall \lambda \in \mathbb{R}$$
* **在 MAB 中的作用**：亞高斯分佈意味著該隨機變數的尾部分佈衰減速度至少與高斯分佈一樣快（有界隨機變數皆為亞高斯）。它是 Bandit 理論推導的基石，允許我們使用如 **Hoeffding 不等式** 等集中不等式，去為未知的真實獎勵估計出一個極其緊湊的信心區間。

### Upper Confidence Bound Algorithm (UCB, 上信賴界線演算法)

UCB 體現了**「面對不確定性時保持樂觀 (Optimism in the face of uncertainty)」**的經典思想。
* **公式**：在時間步 $t$，為每根拉桿計算一個評估值 $\text{UCB}_i(t)$：
  $$\text{UCB}_i(t) = \hat{\mu}_i(t) + \sqrt{\frac{2 \ln t}{N_i(t)}}$$
  * $\hat{\mu}_i(t)$：拉桿 $i$ 當前的經驗平均獎勵（代表 Exploitation 項）。
  * $\sqrt{\frac{2 \ln t}{N_i(t)}}$：信心區間半寬度，與被拉動次數 $N_i(t)$ 成反比（代表 Exploration 項）。
* **策略**：每次選擇 $\text{UCB}_i(t)$ 最大的拉桿。若一根拉桿很少被拉動，其不確定性大，第二項就會膨脹，從而獲得被探索的機會。UCB 實現了理論上最優的對數累積遺憾比 $O(\ln t)$。

### Thompson Sampling (湯普森採樣)

湯普森採樣是一種優雅的 **貝氏 (Bayesian) 隨機化演算法**。
* **機制**：
  1. 為每根拉桿的未知真實獎勵機率設一個先驗分佈（例如 Beta 分佈或高斯分佈）。
  2. 在每一步開始前，從所有拉桿當前的後驗機率分佈中，**各自隨機抽取一個樣本值 $\theta_i$**。
  3. 比較這些抽樣值，**拉動 $\theta_i$ 最大**的那根拉桿。
  4. 觀測到實際獎勵後，利用貝氏定理更新該拉桿的後驗分佈。
* 由於其隨機性，高潛力（後驗方差大）或高回報的拉桿自然有更高的機率抽到大值，在實務上其表現往往優於 UCB。

### Adversarial Bandit (對抗式老虎機)

在隨機老虎機中，我們假設每個拉桿的獎勵來自一個固定的機率分佈。然而在**對抗式老虎機**中，這個假設被完全推翻：環境被視為一個具備惡意的對抗者，它可以在每一步**任意、甚至蓄意地設定**每個拉桿的獎勵值（完全不依賴分佈），目的是讓玩家的遺憾最大化。在這種情況下，任何確定性的演算法都會被對抗者輕易看穿並針對。

### Exp3 Algorithm (指數權重探索與利用演算法)

Exp3 是解決對抗式老虎機的經典反制武器。
* **策略**：
  1. 維護每個拉桿的權重 $w_i(t)$，並依據權重的比例轉化為拉動各桿的機率 $p_i(t) = \frac{w_i(t)}{\sum_j w_j(t)}$。
  2. 依據該機率分佈進行**隨機抽樣**，決定拉動哪根拉桿。
  3. 獲得獎勵後，使用**重要性採樣 (Importance Sampling)** 技術來建構真實獎勵的無偏估計值：$\hat{r}_i(t) = \frac{r_i(t)}{p_i(t)}$（若選中的機率很低卻拿到了獎勵，該獎勵會被大幅放大，以補償未選中時的遺失資訊）。
  4. 依據指數形式更新權重：$w_i(t+1) = w_i(t) \cdot e^{\gamma \hat{r}_i(t)}$。此法確保了即使在最壞的對抗環境下，累積遺憾依然能被壓制在 $O(\sqrt{t})$ 級別。

---

## VII. Markov Decision Process (MDP, 馬可夫決策過程)

MDP 是強化學習中最核心的數學建模框架，定義為五元組 $(S, A, P, R, \gamma)$。

### Markov Chain (馬可夫鏈)

馬可夫鏈是一種具有**馬可夫性質 (Markov Property)** 的隨機狀態轉移模型。
* **馬可夫性質**：指系統的「未來狀態僅與當前狀態有關，而與過去的歷史無關」。用條件機率數學式表達為：
  $$P(X_{t+1} = x_{t+1} \mid X_t = x_t, X_{t-1} = x_{t-1}, \dots, X_0 = x_0) = P(X_{t+1} = x_{t+1} \mid X_t = x_t)$$
  在 MDP 中，這意味著一旦當前狀態 $s$ 與動作 $a$ 確定，下一狀態的分佈便已完全確立。

### Bellman Optimality Equation (貝爾曼最佳化方程式)

貝爾曼最佳化方程式闡明了最優價值函數之間的遞迴關係：一個狀態的最優價值，等於在該狀態下採取最優動作所能獲得的即時獎勵，再加上未來所有期望狀態的折現最優價值之最大值。
* **狀態價值最佳化方程式**：
  $$V^*(s) = \max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s, a) V^*(s') \right]$$
* **動作價值最佳化方程式**：
  $$Q^*(s,a) = R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s, a) \max_{a' \in A} Q^*(s', a')$$

#### Policy Iteration (策略迭代)
策略迭代包含兩個不斷交替循環的步驟，直至策略收斂：
1. **策略評估 (Policy Evaluation)**：在當前策略 $\pi$ 下，反覆迭代貝爾曼期望方程式，直到求解出該策略對應的精確價值函數 $V^\pi$。
2. **策略改進 (Policy Improvement)**：利用貪婪策略提取法，根據 $V^\pi$ 更新策略：$\pi_{new}(s) = \arg\min_a [R(s,a) + \gamma \sum s' P V^\pi(s')]$。

#### Value Iteration (價值迭代)
價值迭代不包含顯式的策略維護。它直接將貝爾曼最佳化方程式轉化為動態規劃的更新式：
$$V_{k+1}(s) \leftarrow \max_{a \in A} \left[ R(s,a) + \gamma \sum_{s' \in S} P(s' \mid s, a) V_k(s') \right]$$
不斷迭代直到 $V$ 矩陣收斂。最終，只需一步貪婪操作即可直接導出最優策略 $\pi^*$。通常其收斂速度比策略迭代更快。

### Reinforcement Learning (RL, 強化學習)

強化學習與 MDP 的區別在於：在 RL 中，環境的轉移機率 $P$ 和獎勵函數 $R$ 對智能體 (Agent) 而言是**完全未知（Model-Free）**的。智能體必須透過與環境反覆互動（試錯 Trial-and-Error）來學習最優策略。

#### Q-Learning (Q 學習演算法)
Q-Learning 是一種經典的 **時序差分 (Temporal Difference, TD)**、**時機獨立 (Off-Policy)** 的無模型強化學習演算法。
* **核心更新公式**：
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
  * $\alpha$：學習率。
  * $R(s, a) + \gamma \max_{a'} Q(s', a')$：**TD 目標值 (TD Target)**，代表對未來回報的樂觀估計。
  * 括號內整體為 **TD 誤差 (TD Error)**。
* **Off-Policy (離策 / 時機獨立) 之本質**：更新 $Q(s,a)$ 時，目標值中所代入的動作 $\max_{a'} Q(s', a')$ 是**絕對貪婪**的動作；然而，智能體在實際環境中與下一狀態 $s'$ 互動時，所採取的動作通常是基於 $\varepsilon$-Greedy 策略選出的（可能帶有隨機探索）。因為「更新公式使用的策略」與「實際行動所用的策略」不同，故稱為 Off-Policy。
