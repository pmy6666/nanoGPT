import math


def sinusoidal_position_encoding(seq_len: int, d_model: int) -> list[list[float]]:
    """构造经典 Transformer 的正弦位置编码表。"""
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal position encoding.")

    pe = [[0.0 for _ in range(d_model)] for _ in range(seq_len)]
    for pos in range(seq_len):
        for dim in range(0, d_model, 2):
            """ 原始公式：
            PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
            这里 dim = 2i, 因此
            pos / 10000^(dim / d_model)
            == pos * 10000^(-dim / d_model)
            == pos * exp((-ln 10000) * dim / d_model)
            """
            div_term = math.exp((-math.log(10000.0) * dim) / d_model)
            angle = pos * div_term
            # 每一对相邻的偶数维/奇数维共享同一个角度，
            # 可以看成一个二维正余弦子空间。
            pe[pos][dim] = math.sin(angle)
            pe[pos][dim + 1] = math.cos(angle)
    return pe


def dot(x: list[float], y: list[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def norm(x: list[float]) -> float:
    return math.sqrt(dot(x, x))


def cosine_similarity(x: list[float], y: list[float]) -> float:
    return dot(x, y) / (norm(x) * norm(y))


def print_basic_example() -> None:
    seq_len = 6
    d_model = 8
    pe = sinusoidal_position_encoding(seq_len, d_model)

    print("=== Basic sinusoidal position encoding ===")
    print(f"shape: ({seq_len}, {d_model})")
    for pos in range(seq_len):
        row = ", ".join(f"{x: .4f}" for x in pe[pos])
        print(f"pos={pos}: [{row}]")
    print()


def print_frequency_example() -> None:
    seq_len = 8
    d_model = 8
    pe = sinusoidal_position_encoding(seq_len, d_model)

    print("=== Different dimensions use different frequencies ===")
    print("dim 0/1 change fast, dim 6/7 change slowly")
    for pos in range(seq_len):
        print(
            f"pos={pos:>2d} | "
            f"(d0,d1)=({pe[pos][0]: .4f}, {pe[pos][1]: .4f}) "
            f"(d6,d7)=({pe[pos][6]: .4f}, {pe[pos][7]: .4f})"
        )
    print()


def print_similarity_example() -> None:
    seq_len = 16
    d_model = 32
    pe = sinusoidal_position_encoding(seq_len, d_model)

    base_pos = 5
    print("=== Cosine similarity to a base position ===")
    for other_pos in [4, 5, 6, 7, 10, 15]:
        sim = cosine_similarity(pe[base_pos], pe[other_pos])
        print(f"sim(pos={base_pos}, pos={other_pos}) = {sim: .4f}")
    print()


def rotation_matrix(delta: float) -> list[list[float]]:
    # 把 [sin(theta), cos(theta)] 旋转成
    # [sin(theta + delta), cos(theta + delta)]。
    # 这正是正弦位置编码具备相对位移性质的原因。
    return [
        [math.cos(delta), math.sin(delta)],
        [-math.sin(delta), math.cos(delta)],
    ]


def matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [
        m[0][0] * v[0] + m[0][1] * v[1],
        m[1][0] * v[0] + m[1][1] * v[1],
    ]


def print_relative_shift_example() -> None:
    d_model = 8
    pos = 7
    k = 3

    pe = sinusoidal_position_encoding(seq_len=32, d_model=d_model)

    pair_index = 0
    # pair_index 选择某一个二维子空间的频率。
    # 当 pair_index = i 时，对应角频率为：
    # omega_i = 1 / 10000^(2i / d_model)
    omega = math.exp((-math.log(10000.0) * (2 * pair_index)) / d_model)
    # 从位置 pos 移动到 pos + k，本质上就是相位增加了 k * omega_i。
    delta = k * omega

    v_pos = pe[pos][0:2]
    v_pos_k = pe[pos + k][0:2]

    # 在单个二维子空间里，位置平移等价于一次旋转。
    predicted = matvec(rotation_matrix(delta), v_pos)
    max_error = max(abs(predicted[0] - v_pos_k[0]), abs(predicted[1] - v_pos_k[1]))

    print("=== Relative shift as rotation in one 2D subspace ===")
    print(f"using dim pair (0, 1), pos={pos}, k={k}")
    print(f"vector at pos    : [{v_pos[0]: .4f}, {v_pos[1]: .4f}]")
    print(f"vector at pos + k: [{v_pos_k[0]: .4f}, {v_pos_k[1]: .4f}]")
    print(f"rotation result  : [{predicted[0]: .4f}, {predicted[1]: .4f}]")
    print(f"max error        : {max_error: .8f}")
    print()


if __name__ == "__main__":
    print_basic_example()
    print_frequency_example()
    print_similarity_example()
    print_relative_shift_example()
