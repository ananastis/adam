import numpy as np
import matplotlib.pyplot as plt

# 1. Цільова функція та її аналітичний градієнт
def rosenbrock(pos):
    """
    Функція Розенброка: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    Глобальний мінімум: f(1,1) = 0
    Є класичним тестом для методів оптимізації через
    вузьку параболічну долину.
    """
    x, y = pos[0], pos[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def grad_rosenbrock(pos):
    """
    Аналітичний градієнт функції Розенброка.
    ∂f/∂x = -2(1-x) - 400x(y-x^2)
    ∂f/∂y = 200(y-x^2)
    """
    x, y = pos[0], pos[1]
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])


# 2. Універсальна функція ітераційної оптимізації

def optimize(step_func, init_pos, target, tol=0.01, max_iter=100000):
    """
    Виконує ітераційний процес оптимізації.

    Параметри:
        step_func : callable — функція одного кроку алгоритму
        init_pos  : array-like — початкова точка
        target    : array-like — відома точка оптимуму (для метрики)
        tol       : float — критерій зупинки за евклідовою нормою ||x_k - x*||
        max_iter  : int — максимальна кількість ітерацій

    Повертає:
        trajectory : ndarray — масив точок x_k
        distances  : ndarray — масив ||x_k - x*|| на кожній ітерації
    """
    pos = np.array(init_pos, dtype=float)
    target = np.array(target, dtype=float)

    trajectory = [pos.copy()]
    distances = [np.linalg.norm(pos - target)]

    for i in range(max_iter):
        if distances[-1] < tol:
            break
        pos = step_func(pos, i)
        trajectory.append(pos.copy())
        distances.append(np.linalg.norm(pos - target))

    return np.array(trajectory), np.array(distances)


# 3. Фабрики кроків для кожного алгоритму

def make_gd(alpha):
    """
    Класичний градієнтний спуск (GD).
    x_{k+1} = x_k - alpha * grad_f(x_k)
    """

    def step(pos, i):
        return pos - alpha * grad_rosenbrock(pos)

    return step


def make_momentum(alpha, gamma):
    """
    Метод моменту (канонічна форма, Ruder 2016).
    v_t = gamma * v_{t-1} + alpha * grad_f(x_t)
    x_{t+1} = x_t - v_t

    gamma ≈ 0.9 — коефіцієнт моменту (інерції)
    """
    v = np.zeros(2)

    def step(pos, i):
        nonlocal v
        g = grad_rosenbrock(pos)
        v = gamma * v + alpha * g
        return pos - v

    return step


def make_adam(alpha, beta1, beta2, epsilon):
    """
    Адаптивний оптимізатор Adam (Kingma & Ba, 2014).

    m_t = beta1*m_{t-1} + (1-beta1)*g_t          # перший момент
    v_t = beta2*v_{t-1} + (1-beta2)*g_t^2        # другий момент
    m_hat = m_t / (1 - beta1^t)                  # корекція зміщення
    v_hat = v_t / (1 - beta2^t)                  # корекція зміщення
    x_{t+1} = x_t - alpha * m_hat / (sqrt(v_hat) + epsilon)

    Типові гіперпараметри: beta1=0.9, beta2=0.999, epsilon=1e-8
    """
    m = np.zeros(2)  # перший момент (EMA градієнтів)
    v = np.zeros(2)  # другий момент (EMA квадратів градієнтів)

    def step(pos, i):
        nonlocal m, v
        t = i + 1
        g = grad_rosenbrock(pos)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        return pos - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    return step


# 4. Проведення експерименту
np.random.seed(42)
target_min = [1.0, 1.0]  # глобальний мінімум функції Розенброка
start_pos = [-1.5, 0.5]  # початкова точка (поза долиною)

traj_gd, dist_gd = optimize(make_gd(alpha=0.001), start_pos, target_min)
traj_mom, dist_mom = optimize(make_momentum(alpha=0.001, gamma=0.9), start_pos, target_min)
traj_adam, dist_adam = optimize(make_adam(alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8), start_pos, target_min)

print(f"МГС:      {len(dist_gd) - 1:>7} ітерацій")
print(f"Momentum: {len(dist_mom) - 1:>7} ітерацій")
print(f"Adam:     {len(dist_adam) - 1:>7} ітерацій")

# 5. Візуалізація результатів

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# (а) Траєкторії на контурному графіку
x_grid = np.linspace(-2.0, 2.0, 400)
y_grid = np.linspace(-1.0, 3.0, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = rosenbrock([X, Y])

ax1.contour(X, Y, np.log1p(Z), levels=30, cmap='Blues', alpha=0.6)
ax1.plot(*zip(*traj_gd), color='green', lw=1.2, label='МГС')
ax1.plot(*zip(*traj_mom), color='blue', lw=1.2, label='Momentum')
ax1.plot(*zip(*traj_adam), color='red', lw=1.2, label='Adam')
ax1.scatter(*start_pos, color='black', zorder=5, label='Старт (-1.5, 0.5)')
ax1.scatter(*target_min, color='gold', marker='*', s=200, zorder=5, label='Мінімум (1, 1)')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_title('(а) Траєкторії на контурному графіку')
ax1.legend(fontsize=8)

# (б) Збіжність (логарифмічна шкала)
tol_line = 0.01
ax2.semilogy(dist_gd, color='green', lw=1.5, label='МГС')
ax2.semilogy(dist_mom, color='blue', lw=1.5, label='Momentum')
ax2.semilogy(dist_adam, color='red', lw=1.5, label='Adam')
ax2.axhline(tol_line, color='black', linestyle='--',
            label=f'Критерій зупинки ($\\varepsilon = {tol_line}$)')
ax2.set_xlabel('Номер ітерації ($k$)')
ax2.set_ylabel('Відстань до оптимуму, $\\|x_k - x^*\\|$')
ax2.set_title('(б) Залежність $\\|x_k - x^*\\|$ від номера ітерації')
ax2.legend(fontsize=8)
ax2.grid(True, which='both', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('rosenbrock_convergence.pdf', format='pdf', bbox_inches='tight')
print("Збережено: rosenbrock_convergence.pdf")