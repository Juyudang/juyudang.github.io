---
title: Lagrangian Multiplication
toc: true
date: 2020-03-02 08:51:55
tags:
	- StudyNotes
	- MachineLearning
categories:
	- Study Notes
	- Machine Learning

---



# Lagrangian Multiplication



Constraint optimization.

어떤 objective function을 파라미터에 대해 최대화하거나 loss function을 최소화하려고 하는데, 파라미터가 가질 수 있는 값에 제약조건이 있는 경우, Lagrangian multiplication을 사용할 수 있다.



## Background

Lagrangian multiplication은 gradient vector의 방향 특성을 이용한, constraint optimization을 푸는 방법론중 하나이다. Lagrangian을 알기 위해서는 gradient vector의 특성을 파악해야 한다.



### Gradient Vectors

원 함수 $F(x, y) = x^2 + y^2 - 4 = 0$을 생각해보자. 이 함수의 differential $d F(x, y)$는 다음과 같다.
$$
d F(x, y) = \frac{\partial F(x, y)}{\partial x} \Delta x + \frac{\partial F(x, y)}{\partial y} \Delta y
$$
왜냐하면, $\frac{\partial F(x, y)}{\partial x} \Delta x$만을 봤을 때, $\frac{\partial F(x, y)}{\partial x}$는 $y$가 고정되어 있고, $x$만 변화시켰을때의 $F(x, y)$의 변화량이다. 즉, $x$방향의 기울기이다. 즉, $\frac{\partial F(x, y)}{\partial x}$는 $\Delta x$ 양 만큼의 미세한 변화를 $x$축에 가했을 때, $F(x, y)$의 변화량이다. 이건 $\frac{\partial F(x, y)}{\partial y}$도 마찬가지로 해석이 가능하다. 따라서, $d F(x, y)$는 $x, y$방향으로 각각 $\Delta x, \Delta y$만큼 변화를 가했을 때의 $F(x, y)$의 변화량이라고 볼 수 있다.

위 식은 다음처럼 표현할 수 있다.
$$
d F(x, y) = \begin{bmatrix}
\frac{\partial F(x, y)}{\partial x} \newline 
\frac{\partial F(x, y)}{\partial y}
\end{bmatrix} \cdot \begin{bmatrix}
\Delta x \newline 
\Delta y
\end{bmatrix} = 
\nabla F(x, y) \cdot \Delta v
$$
이때, $$\cdot$$은 내적이고, $\nabla F(x, y)$는 gradient vector, $\Delta v$는 $x, y$가 변화하는 방향 벡터이다.

Gradient vector $\nabla F(x, y)$는 $F(x, y)$의 surface(tangent plane)에 수직인 특징이 있다.



### Gradient Descent

어떤 함수 $F(x, y)$를 최소화하는 $x, y$를 찾고 싶을 때는, 최소가되는 그 지점에서의 gradient vector가 $\nabla F(x, y)=0$를 만족한다는 성질을 이용하면 된다. 이렇게 계산된다면, closed form으로 minimum점을 계산할 수 있다. 하지만, parameter(여기선 $x, y$)의 수가 많거나 식이 복잡해지면 그게 쉽지가 않고, 적절한 시간 안에 계산불가능할 수 있다.

Gradient descent는 어떤 함수 $F(x, y)$가 있을 때, 이 함수의 최솟값을 iterative한 방식으로 계산하는 방법 중 하나이다. 특히, 최솟값을 찾기 위해 gradient vector를 이용해서 함수 $F(x, y)$의 그래프를 따라 하강하게 된다.

다음의 과정을 통해 gradient descent가 동작한다.

1. 일단, $x_0, y_0$을 설정(초기값)

2. $(x_0, y_0)$에서의 gradient vector $\nabla F(x_0, y_0)$을 계산한다.

3. 여기서 $F(x, y)$의 변화량이 가장 작은(가장 큰 음수) 방향으로 $x, y$를 이동시켜야 하는데, 즉, $\Delta v = \begin{bmatrix} \Delta x \newline \Delta y \end{bmatrix}$를 구해야 한다.

4. Gradient vector에서 나온 식에서, $dF(x, y)$를 최소화하는, 내적을 구해야 하는데, gradient vector는 이미 계산했고, $\Delta v$의 step size가 정해졌을 때, $\Delta v$의 방향을 gradient vector와 180도 반대방향으로 가게 한다면, $dF(x, y)$가 절댓값이 가장 큰 음수가 될 것이다. 즉,
   $$
   \Delta v = -\nabla F(x, y)
   $$
   하지만, step size를 1로 두면 너무 크다. 따라서, 작은 수 $\eta$를 곱해준다.
   $$
   \Delta v = - \eta \nabla F(x, y)
   $$
   



### Contraint Optmization

다음과 같이, 어떤 함수 $F(x, y)$를 최소화 또는 최대화하는데, 파라미터 $x, y$의 범위에 조건이 걸린 경우를 말한다.
$$
\underset{x, y}{ \text{min} } [F(x, y) = x^2 + y^2] ~ \text{ s.t } ~ x + y = 1
$$
즉, $x + y = 1$을 만족하는 $x, y$중에서 $x^2+ y^2$를 최소화하는 $x, y$를 찾아야 한다는 것.

이 경우는 매우 간단하게 contraint 식을 $F(x, y)$에 대입해주면 된다.
$$
F(x, y) = (1 - y)^2 + y^2
$$
따라서, 이를 미분하고 gradient vector가 0이 되는 지점을 찾으면 될 것이다.

하지만, $F(x, y)$가 복잡하고, constaint 식 역시 복잡하며, 심지어 contraint가 여러개일 경우, 이렇게 closed form으로 구하는게 불가능해진다. 이를 좀 더 보편적으로 해결하기 위한 방법이 Lagrangian multiplication을 이용하는 것이다.



## Lagrangian Multiplication

어떤 objective function $F(x, y)$가 있고, constraint 함수인 $g(x, y)$가 있을 때, $g(x, y)$를 만족하면서 $F(x, y)$를 최대화하는 지점 $(x', y')$에서는, $F$의 gradient vector $\nabla F(x', y')$와 $g$의 gradient vector $\nabla g(x', y')$의 방향은 같거나 180도 방향이다.

![image-20200206200220651](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200206200220651.png)

그리고, 그 외 지점에서는 이게 성립되지 않는다. 따라서, 다음을 만족하는 $(x', y')$은 maximum 또는 minimum point라고 볼 수 있다.
$$
\nabla F(x', y') = \lambda \nabla g(x', y')
$$
$\lambda$는 상수이며, 두 gradient vector가 반드시 같은 크기일 필요는 없다는 의미로 해석될 수 있다. 하지만, 방향은 반드시 평행하다.

이 수식을 이용해서 constraint optimization을 해결하는 방식을 lagrangian multiplication이라고 부르며, $\lambda$는 lagrangian constant라고 부른다.

예를들어, 다음을 만족하는 점을 찾는다고 가정한다.
$$
\underset{x, y}{ \text{min} } [F(x, y) = xy] ~ \text{ s.t } ~ x^2 + y^2 - 4 = 0
$$
즉, $g(x, y) = x^2 + y^2 - 4 = 0$이다.

이 수식은 대입법을 이용해서 closed form으로 바로 풀 수 있지만, lagrangian muliplication방법으로 풀어볼 수도 있다.
$$
\nabla F(x, y) = \begin{bmatrix}
\frac{\partial F(x, y)}{\partial x} \newline 
\frac{\partial F(x, y)}{\partial y}
\end{bmatrix}
= \lambda \cdot \begin{bmatrix}
\frac{\partial g(x, y)}{\partial x} \newline 
\frac{\partial g(x, y)}{\partial y}
\end{bmatrix}
$$
이므로,
$$
\begin{bmatrix}
y \newline 
x
\end{bmatrix}
= \begin{bmatrix}
2 \lambda x \newline 
2 \lambda y
\end{bmatrix}
$$
일 것이다. 또한, 3변수 연립방정식을 풀기 위해 $g(x, y) = x^2 + y^2 - 4 = 0$도 같이 이용한다. 이 세 가지 수식을 이용한 연립방정식을 풀면, constraint를 만족하는 극점(극대, 극소)을 얻을 수 있다.



요약하면, 다음을 만족하는 점 $(x, y)$는 optimal point이다. 따라서, 다음 연립방정식을 풀면 된다.
$$
\begin{bmatrix}
\frac{\partial F(x, y)}{\partial x} \newline 
\frac{\partial F(x, y)}{\partial y}
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial g(x, y)}{\partial x} \newline 
\frac{\partial g(x, y)}{\partial y}
\end{bmatrix}
$$

$$
g(x, y) = 0
$$



### Constraint to non-Constraint Problem

Lagrangian multiplication을 푸는 것은 다음 식을 만족하는 $\vec{x}$를 구하는 것이다.
$$
\nabla F(\vec{x}) = \lambda \cdot \nabla g(\vec{x})
$$
이 식을 조금 변형해보면,
$$
\nabla F(\vec{x}) = \nabla \lambda g(\vec{x})
$$
$$
\nabla F(\vec{x}) - \nabla \lambda g(\vec{x}) = 0
$$

$$
\nabla (F(\vec{x}) - \lambda g(\vec{x})) = 0
$$

$Q(\vec{x}, \lambda) = F(\vec{x}) - \lambda g(\vec{x})$라고 정의해보면,
$$
\bigtriangledown Q(\vec{x}, \lambda) = 0
$$
으로 정리할 수 있다. 이것은, $Q(\vec{x},\lambda)$를 non-constaint optimization을 한 식이 된다.

즉, $F(\vec{x})$를 어떤 constraint $g(\vec{x})$에 맞게 optimization을 한다는 것은, $F(\vec{x}) - \lambda g(\vec{x})$를 non-constraint 환경에서 optimization하는 것과 같다.

Neural network regularization도 해당 constraint ($l_1 norm, l_2 norm$) 에 맞게 $loss$함수를 최적화하는 것이라고 해석할 수도 있지 않을까. 다만, 차이점은, lagrangian 에선, $\lambda$도 파라미터이고, $\vec{x}$뿐 아니라 $\lambda$에 대해서도 최적화를 수행한다. Neural network regularization에서는 $\vec{x}$에 대해서만 최적화를 하며, $\lambda$는 하이퍼파라미터로 한다. 제약조건을 완전히 지키지는 않고, 약간의 제제만 가하는 것이라고 볼 수 있겠다.



### Multi-constraint Optimization

만약, $F(\vec{x})$를 최적화하는데, constraint가 $g_1(\vec{x}), g_2(\vec{x}), \cdots, g_k(\vec{x})$ 등 $k$개가 있다고 해 보자. 이때, $F(\vec{x})$의 극점이면서, 위 constraint들을 만족시키는 $$\vec{x}$$를 구하는 것은 다음의 식을 푸는 것과 같다.
$$
\nabla (F(\vec{x}) - \lambda_1 g_1(\vec{x}) - \lambda_2 g_2 (\vec{x}) - \cdots - \lambda_k g_k (\vec{x})) = 0
$$
또는 $Q(\vec{x}, \lambda) = F(\vec{x}) - \lambda_1 g_2(\vec{x}) - \cdots - \lambda_k g_k(\vec{x})$의 극점을 구하는 것과 같다.
$$
\nabla Q(\vec{x}, \lambda) = 0 ~ \text{w.r.t} ~ \vec{x}, \lambda
$$


