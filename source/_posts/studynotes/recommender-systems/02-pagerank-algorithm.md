---
title: 02. PageRank Algorithm
toc: true
date: 2020-09-20 21:39:05
tags: 
    - StudyNotes
    - RecommenderSystems
categories: 
    - Study Notes
    - Recommender Systems
---

# PageRank Algorithm

PageRank 알고리즘은 구글이 개발하고 사용중인 구글 검색엔진 알고리즘이다. 사용자가 검색어를 입력하면, 그 검색어에 해당하는 글들을 추천해서 띄워주는 추천 시스템이라고도 볼 수 있다. 본 문서에서는 PageRank 알고리즘에 대해서 정리보해보고자 한다.

# Introduction

---

PageRank 알고리즘을 간단하게 요약하면, 각 웹페이지마다 랭크를 매긴 후, 검색어에 따라, 높은 랭킹의 웹 페이지를 상단에 위치시키기 위한 알고리즘이다. 즉, 페이지마다 랭크를 매기는 기술이다.

구글은 각 웹 페이지의 랭킹을 매기기 위해, 페이지간의 링크를 활용한다. 어떤 특정 웹 페이지로 가는 외부링크(유입경로)가 많다면, 그 웹 페이지의 랭크는 상승하도록 한다. 반면, 유입될 경로가 적은 웹 페이지의 경우, 그 랭크가 낮다.

PageRank 알고리즘을 이해하기 위해서는 먼저 Markov chain을 이해해야 한다. 구글은 페이지 유입 경로인 링크를 통한 페이지 이동을 Markov chain으로 모델링하고, 해당 Markov chain으로 PageRank를 계산한다.

# Preliminaries

---

본 파트에서는 PageRank 알고리즘을 이해하기 위해 필요한 선행지식에 대해 최소한으로 논의해보고자 한다.

PageRank를 이해하는데 다음의 개념을 아는 것이 필요하다.

- Markov chain
    - Transition matrix
    - Transition matrix smoothing
    - Limiting distribution vs stationary distribution
    - (Perron-Frobenius Theorem)

## Markov Chain

---

구글은 페이지간 링크 이동을 Markov chain으로 모델링해 놓았다.

- 페이지 $i$에서 $n$개의 페이지로 가는 링크가 있다고 하자. 그럼, 그 $n$개의 페이지로 갈 확률은 각각 $1/n$이다.
- 페이지 $i$에서 링크를 통해 페이지 $j$로 가는 확률 $p(i \rightarrow j)$은 오직 페이지 $i$에 있을떄의 상태로부터만 영향을 받는다. 사용자가 페이지 $i$로 어떤 경로를 통해 들어왔건, $p(i \rightarrow j)$에는 영향을 끼치지 않는다. 즉, 웹페이지 이동 모델은 Markov assumption을 따른다.

예를들어, $i$라는 페이지가 있다고 가정했을 때, 이 페이지에는 다른 외부페이지 $j, k, l$로 가는 링크가 존재한다고 하자. 그럼, 페이지 $i$에서 각 페이지로 이동할 확률은 각각 $1/3$이 된다.

$$p(i \rightarrow j) = p(i \rightarrow k) = p(i \rightarrow l) = \frac{1}{3}$$

### Transition Matrix of Markov Chain

Markov chain에서는 상태를 이동하는 확률을 저장한 transition matrix를 정의할 수 있다.

웹 페이지가 $i, j, k$ 3개밖에 없다고 가정해보자. 그리고, $i$ 페이지에는 $j$로 가는 링크가 있고, $j$ 페이지에는 $i, k$로 가는 링크가 있고, $k$페이지에는 다른 외부로가는 링크가 없다고 해 보자. 그럼, 세 개의 웹 페이지 $i, j, k$에 대해 transition matrix $A$는 다음처럼 정의할 수 있다.

$$A = \begin{pmatrix}
0 & 1 & 0 \\
\frac{1}{2} & 0 & \frac{1}{2} \\
0 & 0 & 0
\end{pmatrix}$$

첫 번째 행은 페이지 $i$에서 각각 $i, j, k$로 이동할 확률을 의미하며,

두 번째 행은 페이지 $j$에서 각각 $i, j, k$로 이동할 확률,

세 번째 행은 페이지 $k$에서 각각 $i, j, k$로 이동할 확률을 의마한다.

하지만, 알다시피, 구글을 통해 접근할 수 있는 웹 페이지는 한 두개가 아니다. 우리가 상상할 수 없을 정도의 개수만큼 웹 페이지가 존재하는데, 웹 페이지가 100만개 있다고 가정해보자. 그리고, 어느 한 페이지 $i$에 들어가본다고 가정해보자. 이때, 페이지 $i$에는 과연 100만개의 웹 페이지중에서 몇 개나 링크를 가지고 있을지 상상해 볼 수 있을 것이다. 당연히 대부분의 웹 페이지에 대한 링크는 가지고 있지 않고 매우 극소수의 페이지로 가는 링크만이 존재한다. 이것은 페이지 $i$ 뿐 아니라, 100만개의 모든 웹 페이지에게 성립한다.

즉, transition matrix의 관점으로 보면, 대부분의 원소가 0이 된다.

### Limiting Distribution (vs Stationary Distribution)

그럼, 이 transition matrix로 페이지의 랭크를 어떻게 계산할까. 바로 transition matrix의 limiting distribution을 계산하는 것이 PageRank 알고리즘의 핵심이다.

Transition matrix는 각 페이지에서 다른 페이지로 이동할 확률을 정의한 행렬이다. 그러나, 이 행렬은 각 페이지를 비교하는 어떤 수치값을 제공해주지는 못한다. 구글에서 취한 전략은 다음과 같다.

1. Initial state distribution $\pi_0$을 정의한다.

    Initial state distribution이란, 맨 처음 웹 페이지를 켰을 때, 각 페이지를 방문할 확률을 의미한다. 구글에서 접속할 수 있는 웹 페이지가 3개라면, 구글은 initial state distribution $\pi_0$를 다음과 같이 정의한다.

    $$\pi_0 = \begin{pmatrix}
    \frac{1}{3} & \frac{1}{3} & \frac{1}{3}
    \end{pmatrix}$$

    즉, 모든 웹 페이지로 시작할 확률이 같도록 정의한다. 이때, $\pi_0$의 각 원소는 각각 $p(s_0 = 1)$, $p(s_0 = 2)$, $p(s_0 = 3)$을 나타낸다. $s_0$는 첫 번째 state(구글에서 처음 들어간 웹 페이지)를 의미하고, $p(s_0 = i)$는 그 첫 웹 페이지가 $i$일 확률을 의마한다.

2. 사용자가 transition matrix $A$를 따라 랜덤으로 페이지의 링크를 따라간다고 가정한다. 이때, $t$번째 탐색에서 사용자가 각 페이지에 있을 확률분포 $\pi_t$를 계산할 수 있다.

    $$\pi_t = p(s_t) \\
    = \sum_{i} p(s_t | s_{t-1} = i) \\
    = \pi_{t-1} A$$

    위 식이 이해가 잘 안된다면, $t=1$일때를 가정해보자. 첫 웹 페이지에서 첫 번째로 링크를 따라 들어가서 다음 페이지로 간다고 해 보자. 이때, 사용자가 랜덤으로 첫 웹 페이지를 선택하고 랜덤으로 첫 번째 링크를 따라간 뒤, 사용자가 각 웹 페이지에 있을 확률은 각각 다음과 같다.

    $$p(s_1=1) = \sum_i p(s_1=1|s_0=i) \\
    p(s_1 = 2) = \sum_i p(s_1 = 2| s_0 = i) \\ 
    p(s_1 = 3) = \sum_i p(s_1 = 3 | s_0 = i)$$

    얘네를 잘 관찰해보면, $\pi_1 = p(s_1)$은 다음과 같음을 쉽게 알 수 있다.

    $$\pi_1 = p(s_1) = \pi_0 A$$

    마찬가지로, 사용자가 첫 번째 링크를 따라온 상태라고 가정했을 때, 두 번째 링크를 따라가서 각 페이지에 있을 확률은 다음과 같다.

    $$\pi_2 = \pi_1 A = \pi_0 A^2$$

    이를 $t$번까지 확장해보면 다음과 같다.

    $$\pi_t = \pi_{t-1} A = \pi_0 A^t$$

3. 사용자가 랜덤으로 페이지 링크를 따라가는 과정을 무한번 반복한다고 가정해보자. 이때, 사용자가 각 웹 페이지에 있을 확률에 대한 분포 $\pi_{\infty}$는 다음과 같다.

    $$\pi_{\infty} = \pi_{\infty - 1} A$$

    $\infty - 1$은 그냥 $\infty$와 같다. 따라서, 다음이 성립한다.

    $$\pi_{\infty} = \pi_{\infty} A$$

    즉, 무한번 링크를 따라가다보면, 사용자가 각 웹 페이지에 있을 확률분포는 더 이상 링크를 따라가봤자 변하지 않는다. 이때 $\pi_{\infty}$를 **limiting distribution**이라고 부른다. **구글이 각 페이지에 랭크를 매기는 방법**은 바로 이 limiting distribution을 계산하는 것이다. 사용자가 임의로 웹 페이지를 링크를 따라 무한히 이동했을 때, 각 페이지에 있을 확률을 페이지의 랭크로 정의한다. 확률이 높다면, 랭크가 높은 것이고, 낮다면 랭크가 낮은 것이다. 이것은 어느정도 합리적인 구상이라고 볼 수 있는데, limiting distribution에서 높은 확률을 가진 페이지는 결국 방문 빈도가 많을 수 밖에 없고, 이 말은 그 페이지로 연결된 링크가 많다는 이야기이며, 인기가 많은 페이지라는 의미이기 때문이다.

선형대수학을 공부하신 분들이라면, limiting distribution의 식이 eigen decompostion과 모양이 같다는 것을 알 수 있을 것이다. 즉, limiting distribution을 계산하려면, eigen decomposition을 수행하면 되지 않을까 하는 생각을 가질 수 있다. 하지만, 불행히도, eigen decomposition을 통해 계산된 eigen vector는 limiting distribution이 맞을 수도 있고, 아닐 수도 있다. Eigen decomposition을 통해 계산된 eigen vector는 분명히 $A$를 더 곱해봤자 변하지 않는 벡터이다. 이 벡터가 확률분포를 만족한다면, 이 벡터를 **stationary distribution**이라고 부른다. Stationary distribution이 limiting distribution과 같다면, PageRank는 간단히 transition matrix를 eigen decomposition하는 것 만드로도 간단히(?) 계산할 수 있다. 물론, transition matrix의 크기가 장난 아니기 때문에 보통 문제가 아닐 수 있지만, Spark와 같은 분산처리 시스템이 있다면 해결될 문제이다.

하지만, 또 다른 문제가 있는데, eigen decomposition으로 얻을 수 있는 eigen value와 eigen vector 쌍은 최대 웹 페이지 개수만큼 존재할 것이라는 점이다($A$가 $n \times n$ 차원이면, $n$개만큼의 eigen vector가 존재할 수도 있다).

하지만, 다음 섹션에서 논의할 Perron-Frobenius Theorem을 사용하면, eigen decomposition을 이용하여 계산된 stationary distribution이 limiting distribution과 같다는 것을 보장받을 수 있다. 또한, 이 이론은 모든 원소가 0보다 큰 eigen vector는 오직 1개밖에 존재하지 않음을 보장해준다. 이 말은, 확률분포가 될 자격이 있는 eigen vector가 eigen decomposition으로 얻을 수 있는 eigen vector들 중에 오직 1개밖에 없다는 것이다. 즉, eigen decomposition으로 계산된 그 유일한 eigen vector가 결국 limiting distribution과 같다는 것을 보장받을 수 있다는 의미이며, eigen decompositon 만으로 PageRank를 풀 수 있다는 이야기가 된다.

### Perron-Frobenius Theorem

이 이론은 Markov chain의 transition matrix $A$가 다음 조건을 만족했을 때, $A$의 eigen decomposition 결과로 구한 stationary distribution이 limiting distribution과 같다는 것을 보장한다.

- $A$의 각 row는 적법한 probability distribution이어야 한다. 즉, $A$의 모든 row의 원소를 더했을 때, 모든 값은 1이 되어야 한다.
- $A$의 모든 원소는 반드시 0이 아닌 양의 값이어야 한다.

이 조건들은 위키백과에 나와있는 Perron-Frobenius Theorem과 조금 차이가 있는데, PageRank에 맞춰서 Perron-Frobenius Theorem을 해석한 위 조건을 사용하기로 한다.

즉, 위 조건들을 만족하면 $A$의 eigen decomposition만으로도 $A$의 limiting distribution을 계산할 수 있다는 의미가 된다. 

앞에서 계속 논의하고 있던 Transition matrix $A$의 각 row는 합이 1이 나오도록 모델링을 한 상황이다. 하지만, 한 페이지가 엄청난 숫자의 페이지의 링크를 가지고 있는 경우는 없다고 봐야하므로, $A$의 원소는 0이 대부분이다. 따라서, Perron-Frobenius Theorem을 활용하려면 엄청난 숫자의 0을 다 양수로 만들어야 한다. 구글은 이를 위해서 transition matrix $A$에 smoothing이라는 작업을 하게 된다.

위키백과에 있는 원래의 Perron-Frobenius Theorem에서는 위 두가지 조건을 만족할 때, 모든 원소가 양의 실수인 eigen vector는 오직 1개만 존재한다는 것을 보장하고 있다.

### Transition Matrix Smoothing

Transition matrix의 대부분이 0이 된다는 사실은 PageRank를 eigen decomposition으로 계산할 수 없게 만든다. 따라서, 구글은 transition matrix $A$를 그대로 사용하지 않고, $A$의 smoothing 버전을 대신 사용하게 된다.

$$\tilde{A} = 0.85 * A + 0.15 * U$$

이때, $\tilde{A}$는 smoothing이 적용된 transition matrix이고, $A$는 원래의 transition matrix이다. 그리고, $U$는 다음과 같이 정의한다.

$$U = \begin{pmatrix}
1/n & 1/n & \cdots & 1/n \\ 
1/n & 1/n & \cdots & 1/n \\
\cdots & \cdots & \cdots & \cdots
\end{pmatrix}$$

$n$은 웹 페이지의 개수, 즉, $A$의 row 개수(또는 column 개수와도 같다)와 같다. 즉, 모든 페이지에 모든 페이지로 가는 링크 하나씩 박아놓은 transition matrix가 $U$이다.

이 $U$와 $A$를 weighted sum을 한 것이 $\tilde{A}$가 되며, $\tilde{A}$의 모든 원소는 0이 아닌 양수가 된다. 실제로 존재하지 않는 링크의 경우, 0이 아니라 매우 작은 양의 숫자로 만들어주는 것이 smoothing의 핵심이다.

이제, smoothing으로 Perron-Frobenius Theorem의 모든 조건을 만족시켰다. 즉, 구글의 검색 엔진 알고리즘인 PageRank 알고리즘을 transition matrix의 eigen decomposition만으로도 계산할 수 있다.

## PageRank Computation

---

PageRank 알고리즘의 계산 과정을 정리하면 다음과 같다.

1. Transition matrix $A$를 정의한다.
2. Initial state distribution $\pi_0$를 정의한다.
3. Transition matrix $A$를 smoothing하여 $\tilde{A}$를 계산한다.
4. $\tilde{A}$에 대해 eigen decomposition을 수행하고, 모든 원소가 양수인 eigen vector를 찾는다.
5. 해당 eigen vector가 바로 각 웹 페이지의 랭크를 표현하는 PageRank이다.