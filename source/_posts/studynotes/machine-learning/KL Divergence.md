---
title: KL Divergence
toc: true
date: 2020-03-01 22:28:55
tags:
	- StudyNotes
	- MachineLearning
categories:
	- Study Notes
	- Machine Learning
---



# KL-Divergence



KL-Divergence에는 두 가지가 있다.

- Forward KL-Divergence
- Reverse KL-Divergence



기본적으로 KL-divergence라고 하면 forward 방식을 가리키며, variational autoencoder에는 reverse방식을 사용한다.



## Forward KL-Divergence

다음이 forward KL-divergence의 식이다.
$$
KLD(P||Q) = \sum_x P(x) \cdot log(\frac{P(x)}{Q(x)})
$$
KL-divergence는 두 확률분포 $P,Q$의 유사도를 나타낼 수 있다. 즉, $P,Q$가 서로 비슷한 모양으로 분포된 확률분포라면, KLD값은 낮다.

이 KL-divergence는 entropy와 관련이 있는데, entropy는 정보량의 기댓값으로, 정보량은 두 확률 사이의 차이가 크면 큰 값을 가진다. 즉, 확률값이 많이 다르면 entropy가 높다.

두 확률분포간 거리를 최소화하는게 목적이 아니라면, $P,Q$에 두 확률분포를 넣고 거리를 구하면 된다. 보통 $P$는 target, true 확률분포가 들어가고 $Q$에는 측정하고자 하는 대상이 들어간다.

두 확률분포간 거리를 최소화시키고자 할때는, $P(x)$는 target 확률 분포, 즉, 목표로 하는 확률분포이며, $Q(x)$는 최적화시키고자 하는 확률분포, 즉, 파라미터가 있는 확률분포이다. 그리고 KLD 식을 최소화하는 $Q(x)$를 수정한다. 즉, $P(x)$에 가깝게 $Q(x)$를 수정하게 된다.



### Forward KLD의 특징

Forward KLD는 $P(x)>0$인 모든 지점에 대해서 확률 분포간의 차이를 줄이려고 노력한다. 최적화된 결과, **$P(x)>$*를 만족하는 모든 $x$의 범위를 $Q(x)$가 커버하게 된다.**

다만, 다음처럼, 최소화된 이후의 KLD 값은 상당히 클 수가 있다.

![1569470523821](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/1569470523821.png)

(그림 출처: https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/)

위의 경우, KLD를 최소화한 결과지만, 결과값도 상당히 큰 KLD값을 가진다. 왜냐하면, $P(x)>0$인 전체 범위를 커버하려고 하기 때문에, $Q(x)$를 정확하게 모델링하지 않으면(위의 경우, 두 가우시안의 mixture model로 해야 할 것이다) 위와 같은 문제가 생긴다.





## Reverse KL-Divergence

다음이 Reverse KL-divergence의 식이다.
$$
RKLD(Q||P) = \sum_x Q(x) \cdot log(\frac{Q(x)}{P(x)})
$$


### Reverse KLD의 특징

만약, 두 분포간의 거리를 측정하고자 하면, forward 방식과 별 다를게 없다. 다만, 값의 차이는 있다. KLD는 대칭함수가 아니기 때문이다.

하지만, 최소화하려고 할 경우, 이번엔 파라미터가 있는 $Q(x)$분포와 target 분포 $P(x)$의 자리가 바뀌었다. 이때는, $Q(x)$가 굳이 $P(x)>0$를 만족하는 모든 $x$범위를 커버하려고 하지 않는다. 식에서 보면, $Q(x) \approx 0$으로 맞춰버리면 그 $x$범위는 최소화가 된다. 즉, 필요한 곳만 볼록 솟게 해서 그 범위에서 최소화를 시키고 나머지 봉우리는 $Q(x) \approx 0$으로 해버리므로, **특정 부분만 캡쳐해서 분포간 거리를 최소화한다.**

따라서 다음 그림처럼 된다.

![1569470891541](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/1569470891541.png)

(그림 출처: https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/)



## 어떤 KLD를 사용해야 할까

만약, 모델링한 $Q(x)$가 target 분포 $P(x)$와 매우 가깝다고 자신이 있을 경우, 또는 $P(x)>0$인 모든 $x$를 커버해야 할 경우, forward KLD를 사용하자.

하지만, 모델링한 $Q(x)$가 target 분포 $P(x)$와 가깝다는 자신이 없고, 분포의 major한 부분만 캡쳐해도 좋은 경우, Reverse KLD를 사용하자.



## Reference

https://wiseodd.github.io/techblog/2016/12/21/forward-reverse-kl/