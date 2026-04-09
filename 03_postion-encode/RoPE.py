import math


def get_inv_freq(d_model: int) -> list[float]:
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for RoPE.")
    # RoPE 会给每个二维维度对分配一个角频率：
    # omega_i = 1 / 10000^(2i / d_model)
    # 这里 range(0, d_model, 2) 直接枚举的就是 dim = 2i。
    return [1.0 / (10000 ** (dim / d_model)) for dim in range(0, d_model, 2)]


def apply_rope(x: list[float], pos: int) -> list[float]:
    """对给定位置的向量应用 RoPE 旋转位置编码。"""
    d_model = len(x)
    inv_freq = get_inv_freq(d_model)
    out = [0.0 for _ in range(d_model)]

    for pair_index, dim in enumerate(range(0, d_model, 2)):
        # 对第 i 个二维维度对，RoPE 的旋转角为 theta_i = pos * omega_i。
        theta = pos * inv_freq[pair_index]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        x_even = x[dim]
        x_odd = x[dim + 1]

        # 应用二维旋转矩阵：
        # [cos(theta) -sin(theta)] [x_even]
        # [sin(theta)  cos(theta)] [x_odd ]
        out[dim] = x_even * cos_theta - x_odd * sin_theta
        out[dim + 1] = x_even * sin_theta + x_odd * cos_theta

    return out


def dot(x: list[float], y: list[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def rotate_pair(x_even: float, x_odd: float, theta: float) -> list[float]:
    # 这是一个辅助函数，显式展示 apply_rope 内部同样使用的二维旋转。
    return [
        x_even * math.cos(theta) - x_odd * math.sin(theta),
        x_even * math.sin(theta) + x_odd * math.cos(theta),
    ]


def print_basic_rotation_example() -> None:
    x = [1.0, 2.0, 3.0, 4.0]
    pos = 3
    rope_x = apply_rope(x, pos)

    print("=== Basic RoPE rotation ===")
    print(f"input vector : {x}")
    print(f"position     : {pos}")
    print(f"rope(x, pos) : {[round(v, 4) for v in rope_x]}")
    print()


def print_pair_rotation_example() -> None:
    x_even, x_odd = 1.0, 2.0
    pos = 5
    theta = pos * get_inv_freq(8)[0]
    rotated = rotate_pair(x_even, x_odd, theta)

    print("=== One 2D pair rotates in its plane ===")
    print(f"original pair : [{x_even:.4f}, {x_odd:.4f}]")
    print(f"theta         : {theta:.4f}")
    print(f"rotated pair  : [{rotated[0]:.4f}, {rotated[1]:.4f}]")
    print()


def print_relative_position_example() -> None:
    q = [0.2, 0.5, -0.3, 0.8]
    k = [0.7, -0.1, 0.4, 0.6]

    m = 2
    n = 9
    shift = 4

    # RoPE 的 attention 分数只依赖相对距离 n - m。
    # 因此如果 q 和 k 同时平移相同距离，分数应保持不变。
    score_1 = dot(apply_rope(q, m), apply_rope(k, n))
    score_2 = dot(apply_rope(q, m + shift), apply_rope(k, n + shift))

    print("=== Attention score depends on relative position ===")
    print(f"score(pos_q={m}, pos_k={n})             = {score_1:.8f}")
    print(f"score(pos_q={m+shift}, pos_k={n+shift}) = {score_2:.8f}")
    print(f"absolute difference                     = {abs(score_1 - score_2):.8f}")
    print()


def print_same_offset_comparison() -> None:
    q = [0.3, -0.4, 0.9, 0.1, -0.2, 0.5, 0.7, -0.6]
    k = [0.8, 0.2, -0.7, 0.4, 0.1, -0.5, 0.6, 0.3]

    pairs = [(1, 6), (3, 8), (5, 10)]

    print("=== Same relative offset gives the same score pattern ===")
    for m, n in pairs:
        score = dot(apply_rope(q, m), apply_rope(k, n))
        print(f"score(pos_q={m}, pos_k={n}, offset={n-m}) = {score:.8f}")
    print()


def print_rope_vs_manual_rotation() -> None:
    q_pair = [0.8, -0.3]
    k_pair = [0.4, 0.9]
    m = 2
    n = 7
    omega = get_inv_freq(8)[0]

    # 分别按各自的绝对位置去旋转 q 和 k。
    q_rot = rotate_pair(q_pair[0], q_pair[1], m * omega)
    k_rot = rotate_pair(k_pair[0], k_pair[1], n * omega)
    direct_score = dot(q_rot, k_rot)

    # 等价的相对位置写法：
    # <R(m)q, R(n)k> = <q, R(n - m)k>
    relative_rotated_k = rotate_pair(k_pair[0], k_pair[1], (n - m) * omega)
    relative_score = dot(q_pair, relative_rotated_k)

    print("=== q^T R(n-m) k matches rotated dot product ===")
    print(f"direct rotated score   = {direct_score:.8f}")
    print(f"relative-position form = {relative_score:.8f}")
    print(f"absolute difference    = {abs(direct_score - relative_score):.8f}")
    print()


if __name__ == "__main__":
    print_basic_rotation_example()
    print_pair_rotation_example()
    print_relative_position_example()
    print_same_offset_comparison()
    print_rope_vs_manual_rotation()
