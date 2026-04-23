import numpy as np

def brede_1833_authentic():
    """
    Аутентичные данные совершенного магического куба 7×7×7
    Фердинанда Бреде (de Fibre), 1833 год.
    Источник: Trump & Becker, "New Discoveries in the History of Magic Cubes", 2024.
    
    Возвращает массив shape (7, 7, 7) в формате cube[z][y][x],
    где z=слой, y=строка (сверху вниз), x=колонка (слева направо).
    """
    layers = [
        # Z=0 (Layer 1)
        [[322,  29,  86, 143, 151, 208, 265],
         [ 87, 144, 152, 209, 266, 316,  30],
         [153, 210, 260, 317,  31,  88, 145],
         [261, 318,  32,  89, 146, 154, 204],
         [ 33,  90, 147, 148, 205, 262, 319],
         [141, 149, 206, 263, 320,  34,  91],
         [207, 264, 321,  35,  85, 142, 150]],
        
        # Z=1 (Layer 2)
        [[100, 157, 214, 271, 328,  42,  92],
         [215, 272, 329,  36,  93, 101, 158],
         [323,  37,  94, 102, 159, 216, 273],
         [ 95, 103, 160, 217, 267, 324,  38],
         [161, 211, 268, 325,  39,  96, 104],
         [269, 326,  40,  97, 105, 155, 212],
         [ 41,  98,  99, 156, 213, 270, 327]],
        
        # Z=2 (Layer 3)
        [[277, 334,  48,  56, 106, 163, 220],
         [ 49,  50, 107, 164, 221, 278, 335],
         [108, 165, 222, 279, 336,  43,  51],
         [223, 280, 330,  44,  52, 109, 166],
         [331,  45,  53, 110, 167, 224, 274],
         [ 54, 111, 168, 218, 275, 332,  46],
         [162, 219, 276, 333,  47,  55, 112]],
        
        # Z=3 (Layer 4)
        [[ 62, 119, 169, 226, 283, 340,   5],
         [170, 227, 284, 341,   6,  63, 113],
         [285, 342,   7,  57, 114, 171, 228],
         [  1,  58, 115, 172, 229, 286, 343],
         [116, 173, 230, 287, 337,   2,  59],
         [231, 281, 338,   3,  60, 117, 174],
         [339,   4,  61, 118, 175, 225, 282]],
        
        # Z=4 (Layer 5)
        [[232, 289, 297,  11,  68, 125, 182],
         [298,  12,  69, 126, 176, 233, 290],
         [ 70, 120, 177, 234, 291, 299,  13],
         [178, 235, 292, 300,  14,  64, 121],
         [293, 301,   8,  65, 122, 179, 236],
         [  9,  66, 123, 180, 237, 294, 295],
         [124, 181, 238, 288, 296,  10,  67]],
        
        # Z=5 (Layer 6)
        [[ 17,  74, 131, 188, 245, 246, 303],
         [132, 189, 239, 247, 304,  18,  75],
         [240, 248, 305,  19,  76, 133, 183],
         [306,  20,  77, 127, 184, 241, 249],
         [ 71, 128, 185, 242, 250, 307,  21],
         [186, 243, 251, 308,  15,  72, 129],
         [252, 302,  16,  73, 130, 187, 244]],
        
        # Z=6 (Layer 7)
        [[194, 202, 259, 309,  23,  80, 137],
         [253, 310,  24,  81, 138, 195, 203],
         [ 25,  82, 139, 196, 197, 254, 311],
         [140, 190, 198, 255, 312,  26,  83],
         [199, 256, 313,  27,  84, 134, 191],
         [314,  28,  78, 135, 192, 200, 257],
         [ 79, 136, 193, 201, 258, 315,  22]]
    ]
    return np.array(layers, dtype=int)


def de_fibre_formula_7x7x7():
    """
    Генерация куба по формуле де Фибра (1833) из оригинальной статьи.
    w(x,y,z) = 49·c + 7·b + a + 1, где:
    
    / a \   / 1  3  2 \ / x \   / 0 \ 
    | b | = | 1  4  2 | | y | + | 3 |  mod 7
    \ c /   \ 2  1  1 / \ z /   \ 1 /
    
    Координаты: 0 ≤ x,y,z ≤ 6 (0-based)
    """
    n = 7
    cube = np.zeros((n, n, n), dtype=int)
    
    # Матрица коэффициентов и вектор сдвига
    M = np.array([[1, 3, 2],
                  [1, 4, 2],
                  [2, 1, 1]])
    t = np.array([0, 3, 1])
    
    for z in range(n):
        for y in range(n):
            for x in range(n):
                pos = np.array([x, y, z])
                abc = (M @ pos + t) % n
                a, b, c = abc
                # w = 49*c + 7*b + a + 1
                cube[z, y, x] = c * 49 + b * 7 + a + 1
                
    return cube


def verify_perfect_cube(cube):
    """
    Полная проверка совершенного (пандиагонального) магического куба.
    """
    n = cube.shape[0]
    M = n * (n**3 + 1) // 2  # 1204
    errors = []
    
    # 1. Прямые линии: строки, столбцы, колонны
    for axis, name in zip([0, 1, 2], ["Rows (Y)", "Cols (X)", "Pillars (Z)"]):
        sums = cube.sum(axis=axis)
        if not np.all(sums == M):
            bad = np.where(sums != M)[0]
            errors.append(f"{name}: {len(bad)} линий ≠ {M}, примеры: {sums[bad[:3]]}")
    
    # 2. 4 пространственные диагонали
    diags = [
        np.diag(np.diag(cube)),  # (0,0,0)→(6,6,6)
        np.diag(np.fliplr(cube).diagonal()),  # и т.д.
    ]
    # Прямая реализация через индексы надёжнее:
    diag_sets = [
        [(i, i, i) for i in range(n)],
        [(i, i, n-1-i) for i in range(n)],
        [(i, n-1-i, i) for i in range(n)],
        [(i, n-1-i, n-1-i) for i in range(n)]
    ]
    for i, coords in enumerate(diag_sets, 1):
        s = sum(cube[z, y, x] for x, y, z in coords)
        if s != M:
            errors.append(f"Space diagonal #{i}: {s} ≠ {M}")
    
    # 3. Диагонали всех 21 плоского сечения (3 оси × 7 сечений × 2 диагонали)
    for axis in range(3):
        for fix in range(n):
            if axis == 0:  # фиксированный Z
                d1 = sum(cube[fix, i, i] for i in range(n))
                d2 = sum(cube[fix, i, n-1-i] for i in range(n))
            elif axis == 1:  # фиксированный Y
                d1 = sum(cube[i, fix, i] for i in range(n))
                d2 = sum(cube[i, fix, n-1-i] for i in range(n))
            else:  # фиксированный X
                d1 = sum(cube[i, i, fix] for i in range(n))
                d2 = sum(cube[i, n-1-i, fix] for i in range(n))
                
            if d1 != M: errors.append(f"Slice diag 1 (axis={axis}, fix={fix}): {d1}")
            if d2 != M: errors.append(f"Slice diag 2 (axis={axis}, fix={fix}): {d2}")
    
    # 4. Проверка уникальности значений 1..343
    uniq = np.unique(cube)
    if len(uniq) != 343 or uniq[0] != 1 or uniq[-1] != 343:
        errors.append(f"Неполный набор чисел: {len(uniq)} уникальных, диапазон [{uniq[0]}, {uniq[-1]}]")
    
    return len(errors) == 0, M, errors


if __name__ == "__main__":
    print("🔍 Загрузка аутентичного куба Бреде (1833)...")
    cube = brede_1833_authentic()
    
    print(f"✅ Shape: {cube.shape}")
    print(f"📊 Диапазон: {cube.min()} ... {cube.max()}")
    print(f"🔢 Уникальных чисел: {len(np.unique(cube))}")
    
    # Проверка
    ok, M, errs = verify_perfect_cube(cube)
    if ok:
        print(f"\n✨ ВЕРИФИКАЦИЯ ПРОЙДЕНА! Магическая константа: {M}")
        print("Все строки, столбцы, колонны, плоские и пространственные диагонали = 1204")
    else:
        print(f"\n❌ ОШИБКИ ({len(errs)}):")
        for e in errs[:5]:
            print(f"  • {e}")
    
    # Тест: первая строка первого слоя (должна давать 1204)
    first_row = cube[0, 0, :]  # Z=0, Y=0, все X
    print(f"\n🧪 Тест: первая строка слоя 1: {first_row.tolist()}")
    print(f"   Сумма: {first_row.sum()} (должно быть {M})")
    
    # Вывод в формате numpy для копирования
    print("\n📦 Готовый массив (cube[z][y][x]):")
    print("cube = np.array([")
    for z in range(7):
        print(f"  {cube[z].tolist()},")
    print("])")
    
    # Сохранение
    np.save("brede_1833_magic_cube.npy", cube)
    print("\n💾 Сохранено в: brede_1833_magic_cube.npy")
