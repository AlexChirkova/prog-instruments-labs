from prettytable import PrettyTable
import numpy as np


def solve_random_system(
        n: int,
        min_a: int = 1,
        max_a: int = 10,
        min_b: int = 1,
        max_b: int = 20
        ) -> None:
    """
    Creates an arbitrary lower unitriangular matrix A of order n,
    vector B is arbitrary.
    It solves the system Ax = B.
    :param n: Matrix dimension n x n
    :param min_a: The minimum value of the coefficients in matrix A.
    :param max_a: The maximum value of the coefficients in matrix A.
    :param min_b: The minimum value of the coefficients in vector B.
    :param max_b: The maximum value of the coefficients in vector B.
    :return: None.
    """
    A = np.zeros((n, n))

    np.random.seed(42)
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                A[i, j] = 1
            else:
                A[i, j] = np.random.randint(min_a, max_a)

    print("Нижняя унитреугольная матрица A:")
    print(A)

    B = np.random.randint(min_b, max_b, size=n)
    print(f"\nВектор B: {B}")

    X = np.zeros(n)
    X[0] = B[0]
    for i in range(1, n):
        X[i] = B[i] - np.dot(A[i, :i], X[:i])

    print(f"\nРешение системы AX = B:")
    print(f"X = {X}")

    print(f"\nПроверка: A @ X = {A @ X}")
    print(f"Должно быть равно B = {B}")


def LU_decomposition(A: np.array, B: np.array) -> None:
    """
    Solves a system of equations Ax = B using LU decomposition.
    It leads the system from the type Ax=B to the type LUx=B.
    :param A: The initial matrix of coefficients.
    :param B: The vector of values of the equations of the system.
    :return: None.
    """
    n = len(A)

    print("Матрица A:")
    print(A)
    print(f"\nВектор B: {B}")

    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        L[i, i] = 1.0
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    print("Матрица L:")
    print(L)
    print("Матрица U:")
    print(U)

    # LUx=b
    # Ly=b
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])
    # Ux=y
    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X[i] = (Y[i] - np.dot(U[i, i + 1:], X[i + 1:])) / U[i, i]

    print(f"\nРешение системы AX = B:")
    print(f"X = {X}")

    print(f"\nПроверка: A @ X = {A @ X}")
    print(f"Должно быть равно B = {B}")


def QR_decomposition(A: np.array, B: np.array) -> None:
    """
    Solves a system of equations Ax = B
    using the QR decomposition of the matrix A.
    It leads the system from the type Ax=B to the type QRx=B.
    Ax=b => QRx=B => Rx = (Q^T)B
    Finds the QR decomposition using the Householder method.
    Verifies the resulting decomposition
    by direct substitution into the original system,
    as well as by the np.solve method.
    :param A: The initial matrix of coefficients.
    :param B: The vector of values of the equations of the system.
    :return: None.
    """
    print("Матрица A:")
    print(A)
    print(f"\nВектор B: {B}")

    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)

    for k in range(n):
        x = R[k:, k]

        norm_x = np.linalg.norm(x)
        sign = 1 if x[0] >= 0 else -1

        v = x.copy()
        v[0] = v[0] + sign * norm_x
        v = v / np.linalg.norm(v)

        # Преобразование Хаусхолдера
        H_k = np.eye(m)
        H_k[k:, k:] -= 2 * np.outer(v, v)

        R = H_k @ R
        Q = Q @ H_k.T  # H_k симметрична и ортогональна

    print("Матрица Q (ортогональная):")
    print(Q)
    print(f"\nОпределитель Q: {np.linalg.det(Q):.6f}")

    print("\nМатрица R (верхняя треугольная):")
    print(R)

    print(f"\nПроверка QR-разложения:")
    print(
        "Норма разности ||A - Q*R|| = {:.2e}".format(np.linalg.norm(A - Q @ R))
    )

    print(f"Норма ||Q^T * Q - I|| = {np.linalg.norm(Q.T @ Q - np.eye(4)):.2e}")

    qt_b = Q.T @ B

    X = np.zeros(n)

    for i in range(n - 1, -1, -1):
        X[i] = qt_b[i]
        for j in range(i + 1, n):
            X[i] -= R[i, j] * X[j]
        X[i] /= R[i, i]

    print(f"\nРешение системы AX = B:")
    print(f"X = {X}")

    print(f"\nПроверка: A @ X = {A @ X}")
    print(f"Должно быть равно B = {B}")

    print(f"Решая СЛАУ с помощью функции np.solve, поличим:")
    print(f"X = {np.linalg.solve(A, B)}")


def check_diagonal_dominance(A: np.array) -> bool:
    """
    Verifies the fulfillment of a sufficient convergence condition
    (diagonal predominance) for the method of simple iterations.
    :param A: The matrix of coefficients.
    :return: Is there a diagonal predominance.
    """
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i, i])
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            return False
    return True


def rearrange_for_dominance(
        A: np.array,
        B: np.array
        ) -> tuple[np.array, np.array, bool]:
    """
    Transforms the matrix of coefficients
    and the vector of values so
    that there is a diagonal predominance.
    :param A: The initial matrix of coefficients
    :param B: The vector of values of the equations of the system
    :return: The transformed matrix and vector.
             Is there a diagonal predominance in the new matrix.
    """
    n = len(A)
    A_new = A.copy()
    B_new = B.copy()

    available_rows = list(range(n))

    for col in range(n):
        max_val = 0
        max_row = -1

        for row in available_rows:
            if abs(A[row, col]) > max_val:
                max_val = abs(A[row, col])
                max_row = row

        if max_row == -1:
            return A, B, False

        if max_row != col and max_row in available_rows:
            A_new[[col, max_row]] = A_new[[max_row, col]]
            B_new[[col, max_row]] = B_new[[max_row, col]]
            available_rows.remove(max_row)

    return A_new, B_new, check_diagonal_dominance(A_new)


def method_of_simple_iterations(
        A: np.array,
        B: np.array,
        epsilon: float = 0.001
        ) -> None:
    """
    Solves the system of equations Ax = B
    using simple iterations with a given accuracy.
    It leads the system from the type Ax=B to the type X^(k+1) = B_s*X^k + C
    Verifies that a sufficient convergence condition is met.
    Forms the iterations in the form of a table.
    Checks the received solution.
    :param A: The initial matrix of coefficients.
    :param B: The vector of values of the equations of the system.
    :param epsilon: The precision with which it is necessary
           to find a solution to the system.
    :return: None.
    """

    print("A = ", A)
    print("B = ", B)

    table = PrettyTable()
    table.field_names = ["k", "x_1", "x_2", "x_3", "norma(x_k - x_k-1)"]
    table.add_row([0, 0, 0, 0, ""])

    has_dominance = check_diagonal_dominance(A)

    if not has_dominance:
        print("\nИсходная матрица не имеет диагонального преобладания.")

        A, B, success = rearrange_for_dominance(A, B)
        if success:
            print(
                "Удалось достичь диагонального преобладания перестановкой "
                "строк."
            )
        else:
            print(
                "Не удалось достичь диагонального преобладания перестановкой "
                "строк."
            )
            for i in range(3):
                if abs(A[i, i]) <= sum(abs(A[i, j])
                                       for j in range(3) if j != i):
                    for j in range(3):
                        if i != j and A[j, i] != 0:
                            factor = A[j, i] / A[i, i] if A[i, i] != 0 else 1
                            A[i] += factor * A[j]
                            B[i] += factor * B[j]
                            break

            if check_diagonal_dominance(A):
                print(""
                      "Удалось достичь диагонального преобладания "
                      "преобразованиями."
                      )
            else:
                print(
                    "Не удалось достичь строгого диагонального преобладания."
                )
                print("Сходимость метода не гарантирована.")
    else:
        print("\nМатрица имеет диагональное преобладание.")

    n = len(A)
    B_s = np.zeros((n, n))
    C = np.zeros(n)

    for i in range(n):
        C[i] = B[i] / A[i, i]
        for j in range(n):
            if i != j:
                B_s[i, j] = -A[i, j] / A[i, i]

    print("\nМатрица B (итерационная):")
    print(B_s)
    print("\nВектор C:")
    print(C)

    X = np.zeros(n)

    k = 0
    while True:
        X_new = np.zeros(n)

        try:
            for i in range(n):
                sum_term = 0
                for j in range(n):
                    if i != j:
                        sum_term += B_s[i, j] * X[j]
                X_new[i] = sum_term + C[i]

            diff = X_new - X

            norm_diff = np.max(np.abs(diff))

            if k < 10 or k % 10 == 9 or norm_diff < epsilon:
                table.add_row([k + 1, X_new[0], X_new[1], X_new[2], norm_diff])
            if norm_diff < epsilon:
                print("\nДостигнута заданная точность epsilon")
                break

            X = X_new.copy()
            previous_norm = norm_diff
            k += 1

            if (norm_diff > 1e10
                    or (previous_norm != float('inf')
                        and norm_diff > 10 * previous_norm)):
                print(f"\nПредупреждение: "
                      f"Метод может расходиться. "
                      f"Норма разности: {norm_diff:.6f}"
                      )
                break

        except (FloatingPointError, OverflowError) as e:
            print(f"\nОшибка вычислений на итерации {k + 1}: {e}")
            print("Метод расходится.")
            break

    print(table)
    print("Решая через np.solve: ", np.linalg.solve(A, B))


def least_squares_method(A: np.array, B: np.array) -> None:
    """
    Finds a pseudo-solution of a system of equations Ax = B
    by the least squares method.
    :param A: The initial matrix of coefficients.
    :param B: The vector of values of the equations of the system.
    :return: None
    """
    print("Матрица A:")
    print(A)
    print(f"\nВектор B: {B}")

    # Находим псевдорешение методом наименьших квадратов
    X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

    print("Псевдорешение системы:")
    print(f"x1 = {X[0]:.6f}")
    print(f"x2 = {X[1]:.6f}")
    print(f"x3 = {X[2]:.6f}")
    print(f"x4 = {X[3]:.6f}")

    print("\nНорма невязки:", np.linalg.norm(A @ X - B))


if __name__ == "__main__":
    print("Task 1")
    solve_random_system(7)

    A2 = np.array([
        [3.8, 14.2, 6.3, -15.5],
        [8.3, -6.6, 5.8, 12.2],
        [6.4, -8.5, -4.3, 8.8],
        [17.1, -8.3, 14.4, -7.2]
    ], dtype=float)

    B2 = np.array([2.8, -4.7, 7.7, 13.5], dtype=float)

    print("\nTask 2")
    LU_decomposition(A2, B2)

    print("\nTask 3")
    QR_decomposition(A2, B2)

    A4 = np.array([
        [5.3, 2.1, 2.8],
        [4.1, 6.7, 4.8],
        [2.7, 1.8, 8.1]
    ])
    B4 = np.array([0.8, 5.7, 3.2])

    print("\nTask 4")
    method_of_simple_iterations(A4, B4)

    A5 = np.array([
        [4.4, -2.5, 19.2, -10.8],
        [5.5, -9.3, -14.2, 13.2],
        [7.1, -11.5, 5.3, -6.7],
        [14.2, 23.4, -8.8, 5.3],
        [8.2, -3.2, 14.2, 14.8]
    ])

    B5 = np.array([4.3, 6.8, -1.8, 7.2, -8.4])

    print("\nTask 5")
    least_squares_method(A5, B5)
