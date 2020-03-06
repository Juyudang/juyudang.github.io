---
title: Principal Component Analysis
toc: true
date: 2020-03-01 22:28:55
tags:
	- StudyNotes
	- MachineLearning
categories:
	- Study Notes
	- Machine Learning
---



# Principal Component Analysis (PCA)



데이터는 signal과 noise로 구성되어 있다. PCA의 핵심 목표는 데이터로부터 signal만 분리해 내는 것이다. PCA는 데이터 feature space를 회전시켜서 signal에 가까운 축과 noise에 가까운 축을 찾아준다. 그리고 데이터를 signal 축에 사영시킴으로써, 다른 축을 날려버리는데, 이때 데이터 정보 손실이 발생할 수 있다. 그러나, 그걸 감수하는 뛰어난 이득이 있기 때문에 PCA는 매우 유용하다.



PCA의 주 활용 목적은 다음과 같이 크게 두 가지로 분류가 가능하다. (둘 다 dimensionality reduction이다)

- **Feature Transformation**

  Feature 공간의 축을 변환시켜서 signal축과 noise축을 구분되게 한다. 이때, 주의할 점은, PCA는 feature 공간을 non-linear하게 변환하는게 아니라 회전만 시킨다. 정확히 말하면, 축들만 회전시켜서 signal축과 noise축을 찾겠다는 것이다.

- **Data Whitening**

  Data whitening이란, 각 feature의 scale을 맞춰 주는 것을 말한다. PCA로 공간을 회전시킨 후엔, 각 축이 독립적이다. 그리고, 각 축의 variance가 구해지므로, 각 축을 standard deviation으로 나눠주면 whitening이 가능하다.

- **Visualization**

  PCA를 통해 signal에 가까운 축과 noise에 가까운 축을 찾아냈다면, signal에 가까운 축이 있을 것이다. Visualization을 위해 가장 signal다운 축 2개만 선택하는 방법도 있는데, 이럴 경우, 데이터에 대한 정보 상당수를 잃어버리지만, 데이터의 정보 상당량을 유지한 체, visualization을 할 수 있다.



다음은 2차원 데이터를 PCA를 통해 축 2개를 찾은 것을 보여준다. 파란색 방향의 축이 signal이 되고, 빨간색 방향의 축이 noise가 될 수 있다(이것은 상대적인 것으로, 절대적으로 어떤것이 noise라고는 판단할 수 없다).

![image-20191219132645587](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20191219132645587.png)

여기서 만약에, 모든 데이터포인트를 파란색 축으로 사영시킨다고 하자. 그러면, 빨간색 방향의 정보는 사라지고, 데이터를 파란색 축 1차원으로만 표현이 가능하다. 빨간색 방향이 사라졌으므로 어느 정도의 데이터 분포에 대한 정보 손실이 있지만, 파란색 축은 대다수 정보를 보전하고 있는 축이다. 이렇게 데이터의 dimension을 reduction할 수 있는데, 이건 머신러닝 알고리즘의 속도를 크게 향상시킬 수 있다.

이건 매우 중요한 요소이다. 데이터의 모든 정보를 다 머신러닝에 때려넣기보단, 가장 메이저한 정보만 줘도 성능 하락없이 빠른 속도로 비슷한 결과를 달성할 수 있다. 이건 머신러닝에서 매우 중요하게 여기는, latent vector를 찾는다는 방향성과 일치한다.



## PCA Methods

그럼 signal 축은 무엇으로 정의해야 할지에 대해서 고민해야 한다. 정답을 말하면, signal 축은 데이터 분포가 가장 널리 퍼진 방향이다. 그래야, 사영 후에, 퍼진 정도의 정보를 최대한 보존할 수 있다. 즉, 데이터들을 이 축에 사영시키면, 다른 어느 축에 사영시키는 것보다 분산이 커야 한다.

즉,

1. 분산이 가장 큰 축에 해당하는 벡터를 찾고,
2. 데이터포인트를 그 벡터에 정사영시킨다.

참고로, 정보 이론을 잠깐 가져오면, 분산이 클수록 정보량이 많다. 즉, 정보량이 많은 축을 찾는 것이다.



### Data Rotation

PCA는 데이터 feature space안에서 데이터간의 관계를 건들지 않는다. 그저 rotation만 시킴으로써, 축만 변화시킬 뿐이다. 변화시킨 축이 signal이라고 할 수 있는 축들이 있고, noise 라고 할 수 있는 축이 있어서, 축을 골라서 사영시킬 수 있을 뿐이다. 데이터 자체를 non-linear하게 변형시키지는 않는다.



## PCA Derivation

우리가 원하는 축의 **단위 벡터를 $e$라고 하자.** 기본적으로 transpose가 없으면 column vector로 간주한다. 데이터들을 $x$라고 한다.

먼저, 데이터를 0-centerize시킨다.
$$
x_i'= x_i - \mu
$$
그리고 난 후, 데이터 $x'$를 축 $e$에 사영시킨 결과는, $e^Tx'$ 또는 $x'^Te$가 될 것이다. 이 사영시킨 결과로 나온 데이터들의 분포의 분산이 최대화되도록 하는 벡터 $e$를 찾아야 한다. 또한, 여기서, $e$는 단위 벡터로 제한하고자 한다. 즉,
$$
\underset{e}{ \text{max} } \frac{1}{N} \sum_{i=1}^N (e^Tx_i')^2 = 
\underset{e}{ \text{max} } \frac{1}{N} \sum_{i=1}^N e^Tx_i'x_i'^Te
$$
$$
\text{s.t} ~ e^Te = 1
$$

참고로, $e^Tx' = x'^Te = \text{scalar value}$이므로, $(e^T x_i')^2$을 위 식처럼 표현이 가능하다. 여기서, $e$는 $\sum$에 영향을 받지 않는 변수이므로 밖으로 뺄 수 있다.
$$
\underset{e}{\text{max}} ~e^T (\frac{1}{N}\sum_{i=1}^N x_i' x_i'^T) e
$$
$$
\text{s.t} ~ e^T e = 1
$$



그런데, $x'$는 column vector이므로, $()$안의 값은 $x'$의 covariance matrix와 같다는 것을 알 수 있다. 이를 $\Sigma$라고 하자.
$$
\underset{e}{ \text{max} } ~ e^T \Sigma e \\
\text{s.t} ~~~ e^Te = 1
$$
이를 Lagrangian multiplier를 이용해서 식을 변형한다.
$$
\underset{e}{\text{max}} ~ e^T \Sigma e - \lambda (e^Te - 1) = \underset{e}{\text{max}} ~F(e)
$$
이제 미분을 수행해서 그 결과가 0이 나오는 지점을 찾아야 한다.
$$
\frac{dF(e)}{de} = 2\Sigma e - 2\lambda e = 0 \\
\Sigma e = \lambda e
$$
즉, 우리가 찾던 벡터 $e$는 0-centered 된 데이터 $x'$의 covariance matrix의 eigen vector라는 사실을 알 수 있다.



그런데, 그럼, $\Sigma$에겐 eigen vector가 많을 것인데, 어떤 eigen vector에다가 사영할 것인지 결정해야 할 것이다. 다음으로 다시 돌아가보면,
$$
\underset{e}{\text{max}} ~ e^T \Sigma e
$$
위 식은 벡터 $e$ 방향으로의 분산에 대한 식을 변형시킨 결과로 얻었던 것이었다. 그런데, $e$가 $\Sigma$의 eigen vector라는 사실을 알았으므로,
$$
\underset{e}{\text{max}} ~ e^T \Sigma e =  \underset{e}{\text{max}} ~ e^T \lambda e
$$
가 되고, $\lambda$는 eigen value, 즉 스칼라값이므로, 맨 앞으로 올 수 있다.
$$
\underset{e}{\text{max}} ~ \lambda e^Te
$$
근데, $e^Te = 1$이라고 이미 제약을 걸어놓았다. 즉,
$$
\underset{e}{\text{max}} ~ \lambda
$$
다시말해, eigen value가 가장 큰 eigen vector를 고르면, 분산이 가장 큰 축을 찾을 수 있다는 말이 된다.

이 축은 데이터의 첫번째 principal component라고 부르며, 데이터의 가장 많은 정보를 포함하는 축이다.



## Principal Components After 1st Component

그래서, 가장 정보량이 많은 방향을 구했다. 그런데, dimensionality reduction을 하려고 할 때, 1차원으로만 압축해버리면 잃어버리는 정보가 매우 많다. 그래서, 첫번째 principal component(PC)를 제외하고 다른 방향으로 가장 많은 정보를 포함하는 축을 찾으려고 한다. 즉, 두번째 principal component를 찾을 것이다.

그런데 다음의 제한 사항이 있어야 한다.

- PC들은 모두 서로 직교해야 한다.

  직교한다는 의미는 각 축이 서로 캡쳐하지 못하는 순수한 정보만 캡쳐할 수 있다는 이야기이다. 첫번째 PC와 비슷한 방향 축을 고른다고 해서 그 축이 정보를 많이 포함할까? 그 축은 첫번째 PC와 정보가 매우 많이 겹칠 것이다.

그래서, 첫번째 PC에 수직인 sub-space에서 다시 분산이 가장 큰 방향을 구하게 된다.

두 번째 PC를 $e_2$라고 했을 때, $e_1^Te_2 = 0, e_2^Te_2 = 1$이라는 제약 조건을 만족시키면서 다음 variance를 최대화시켜야 한다. ($X$는 zero-centered 시켰다고 가정)
$$
\text{Var}(e_2) = (Xe_2)^T(Xe_2) \\
= e_2^T \Sigma e_2
$$
Lagrangian multiplier에 의해,
$$
\underset{e_2}{\text{argmax}} ~ e_2^T \Sigma e_2 - c_1(e_2^Te_2 - 1) - c_2(e_1^Te_2)
$$
이를 미분한 결과가 0이 나와야 하므로,
$$
2 \Sigma e_2 - 2c_1e_2 - c_2 e_1 = 0
$$
양변에 $e_1^T$를 곱하면,
$$
2 e_1^T \Sigma e_2 - 2 c_1 e_1^T e_2 - c_2 e_1^T e_1 = 0 \\
2 e_1^T \Sigma e_2 - c_2 = 0
$$
그런데 이때, $\Sigma$는 eigen decomposition에 의해 다음과 같다. ($d$는 데이터의 차원이라고 하자)
$$
\Sigma = \begin{pmatrix}
v_1 & v_2 & \cdots v_d
\end{pmatrix}
\begin{pmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\cdots \\
0 & 0 & \cdots & \lambda_d
\end{pmatrix}
\begin{pmatrix}
v_1^T \\
v_2^T \\
\cdots \\
v_d^T
\end{pmatrix}
$$
따라서 위 식을 다음처럼 고칠 수 있다.
$$
2 e_1^T \begin{pmatrix}
v_1 & v_2 & \cdots v_d
\end{pmatrix}
\begin{pmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\cdots \\
0 & 0 & \cdots & \lambda_d
\end{pmatrix}
\begin{pmatrix}
v_1^T \\
v_2^T \\
\cdots \\
v_d^T
\end{pmatrix} e_2 - c_2 = 0 \\

2 \begin{pmatrix}
1 & 0 & \cdots 0
\end{pmatrix}
\begin{pmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\cdots \\
0 & 0 & \cdots & \lambda_d
\end{pmatrix}
\begin{pmatrix}
v_1^T \\
v_2^T \\
\cdots \\
v_d^T
\end{pmatrix} e_2 - c_2 = 0 \\

\begin{pmatrix}
2 \lambda_1 & 0 & \cdots 0
\end{pmatrix}
\begin{pmatrix}
v_1^T \\
v_2^T \\
\cdots \\
v_d^T
\end{pmatrix} e_2 - c_2 = 0 \\

2\lambda_1 v_1^T e_2 - c_2 = 0
$$
근데, $v_1$는 첫 번째 eigen vector로, $e_1$과 같다.
$$
2 \lambda_1 e_1^T e_2 - c_2 = 0 \\
c_2 = 0
$$
다시 원래 최대화 식으로 돌아가보자.
$$
\underset{e_2}{\text{argmax}} ~ e_2^T \Sigma e_2 - c_1(e_2^Te_2 - 1) - c_2(e_1^Te_2)
$$
이건 다음처럼 변경된다.
$$
\underset{e_2}{\text{argmax}} ~ e_2^T \Sigma e_2 - c_1(e_2^Te_2 - 1)
$$
얘네를 미분하는 것? 첫 번째 PC를 구할때 지나왔던 길과 같다. 따라서, 다음처럼 유도될 것이다.
$$
\Sigma e_2 = \lambda_2 e_2
$$
즉, $e_2$ 역시, eigen vector이며, 이 $e_2$방향으로도 $e_1$ 방향을 제외하고 분산이 가장 커야 한다. 다음의 식에 의해 $e_2$방향의 분산은 eigen value $\lambda_2$와 같다.
$$
\text{Var}(e_2) = e_2^T \Sigma e_2 = e_2^T \lambda_2 e_2 = \lambda_2
$$
따라서, $e_1$방향의 분산인 $\lambda_1$보다 작으면서 가장 큰 eigen value, 즉, 두번째로 큰 eigen value가 두 번째 PC의 분산값이 된다. 즉, 두번째로 큰 eigen value에 해당하는 eigen vector가 두 번째 PC축이다.

(사실, 두 번째 PC를 다음처럼 추측도 가능하다. 왜냐하면, covariate matrix는 symmetric matrix이다. 그런데, symmetric matrix의 eigen vector들은 모두 서로 직교한다. 따라서, 첫 번째 PC가 eigen vector임이 밝혀진 상황에서, 두 번째 PC는 첫 번째 PC와 직교하므로, 두 번째 PC 역시 eigen vector라는 사실을 알 수 있다.)



### Covariance between PCs

데이터를 각 PCs(covariate matarix의 eigen vector들)에 사영시켰을 때, 사영된 데이터들간 covariance 또는 correlation은 0이다.

편의를 위해 2차원이라고 생각해보자. 2차원에 분포된 데이터들 $x_i$가 있고, 그들의 principal component를 $e_1, e_2$라고 했을 때, $x_1$을 $e_1$에 사영시킨 것은 $(x_i^T e_1)e_1$이고, $e_2$에 사영시킨 것은 $(x_i^T e_2)e_2$이다. 따라서 이들의 covariance가 0이라는 것은 다음을 만족한다는 의미이다.
$$
((x_i^Te_1)e_1)^T \cdot (x_i^T e_2)e_2 = 0
$$
이때, $()$안에 있는 term들은 모두 scalar값이므로(내적이니까),
$$
(x_i^T e_1)e_1^T \cdot (x_i^T e_2) e_2 = 0 \\
(x_i^T e_1)(x_i^T e_2)e_1^Te_2 = 0 \\
(x_i^T e_1)(x_i^T e_2)0 = 0 \\
0 = 0
$$
Symmetric matrix의 eigen vector끼리는 orthogonal하므로 $e_1^Te_2 = 0$이다.

따라서, 등식이 성립하고, 데이터 포인트에 대해 PC축들은 서로 covariance가 0이다. 그러니까, covariance matrix $\text{Cov}(i,j)$에서, $\text{Cov}(i,j) = 0 ~ \text{if} ~ i \not = j$라는 의미.



## Implementation of PCA (Dimensionality Reduction)

구현에서는 데이터 행렬 $X$의 각 데이터 포인트는 row-vector임을 명시한다. 나머지는 모두 column-vector이다.

PCA의 구현은 다음과 같이 요약이 가능하다.

1. 데이터를 0-centered 한다. $Z = X - \mu_X$

2. $Z$의 covariate matrix를 계산한다. $\Sigma = Z^TZ$ 또는, ```np.cov(X)```로 바로 계산

3. $\Sigma$의 eigen decomposition을 계산한다. $V\Lambda V^T$, ```np.linalg.eig(cov_mat)```로 계산

4. Eigen value를 내림차순으로 정렬한다. 당연히 eigen vector들도 동반 정렬되어야 한다.

   ```python
   evalues, evectors  = np.linalg.eig(cov_mat)
   lst = sorted(zip(evalues, evectors), key=lambda item: item[0], reverse=True)
   # result:
   # [(e1, v1), (e2, v2), ...]
   ```

5. 최상위 $k$개의 eigen vector를 뽑아낸다.

   ```python
   evectors = list(map(lambda item: item[1], lst))[:k]
   E = np.hstack(evectors)
   ```

6. 데이터 포인트를 각 eigen vector에 사영시킨다. $\text{inner prod}(x_i, e_j)$

   ```python
   X_reduced = np.matmul(X, E)
   ```



## Implementation of Data Whitening

데이터 whitening이란, 데이터의 분포를 타원형에서 원형으로 re-scaling해주는 것을 말한다. 각 feature들의 scale을 일치시킨다. PCA를 이용하면 이를 수행할 수 있다. 단, feature는 eigen vector로 변환된 상태로 whitening이 이루어진다.

방법은, PCA로 데이터를 rotation시킨 후(basis가 eigen vector들이 된다), 각 eigen vector축 방향을 그 방향의 standard deviation으로 나눠준다. PCA로 변환된(rotation된) 데이터들을 $Z$라고 했을 때,
$$
\begin{pmatrix}
1 \over \sqrt{\lambda_1} & 0 & \cdots & 0 \\
0 & 1 \over \sqrt{\lambda_2} & \cdots & 0 \\
\cdots \\
0 & 0 & \cdots & 1 \over \sqrt{\lambda_d}
\end{pmatrix} Z
$$
왜냐하면, 각 축 방향의 variance는 eigen vector들이기 때문.

![image-20200221092153746](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200221092153746.png)



구현은 다음과 같다.

1. 데이터를 0-centered 한다. $Z = X - \mu_X$

2. $Z$의 covariate matrix를 계산한다. $\Sigma = Z^TZ$ 또는, ```np.cov(X)```로 바로 계산

3. $\Sigma$의 eigen decomposition을 계산한다. $V\Lambda V^T$, ```np.linalg.eig(cov_mat)```로 계산

4. 데이터 $X$를 eigen vector에 사영시킨다. (정렬은 해도되고 안해도되는데, dimensionality reduction까지 하려면 정렬하고 $k$개만 뽑고 거기에 사영시킨다)

   ```python
   evalues, evectors = np.linalg.eig(cov_mat)
   E = np.hstack(evectors)
   X_transformed = np.matmul(X, E)
   ```

5. 얘네에다가 $\Lambda^{-\frac{1}{2}}$를 곱해준다. (Dimensionality reduction했으면 $k$개만 있는 diagonal matrix이다)

   ```python
   LAMBDA = np.diag(evalues)
   X_whitened = np.matmul(LAMBDA, X_transformed)
   ```

   

   

