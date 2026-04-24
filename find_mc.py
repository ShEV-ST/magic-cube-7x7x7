import numpy as np
import random
import time
from itertools import product

class RandomPerfectCubeFinder:
    def __init__(self, n=7):
        self.n = n
        self.magic_const = n * (n**3 + 1) // 2  # 1204
        self.found_cubes = []
        self.canonical_forms = set()
        
    # ─────────────────────────────────────────────
    # 1. РАНДОМИЗИРОВАННАЯ АЛГЕБРАИЧЕСКАЯ ГЕНЕРАЦИЯ
    # ─────────────────────────────────────────────
    def generate_algebraic_random(self):
        """Генерация через случайную невырожденную матрицу над Z_7"""
        n = self.n
        while True:
            # Случайная матрица 3x3 над Z_7
            M = np.random.randint(0, n, (3, 3))
            if np.linalg.det(M) % n == 0:
                continue  # Требуем невырожденность
            
            # Случайный вектор сдвига
            t = np.random.randint(0, n, 3)
            
            cube = np.zeros((n, n, n), dtype=int)
            for z, y, x in product(range(n), repeat=3):
                pos = np.array([x, y, z])
                abc = (M @ pos + t) % n
                a, b, c = abc
                cube[x, y, z] = c * 49 + b * 7 + a + 1
            return cube

    # ─────────────────────────────────────────────
    # 2. СЛУЧАЙНЫЕ ПРЕОБРАЗОВАНИЯ СИММЕТРИИ
    # ─────────────────────────────────────────────
    def random_symmetry_transform(self, cube):
        """Применяет случайную комбинацию преобразований, сохраняющих совершенство"""
        c = cube.copy()
        
        # 1. Случайная перестановка осей
        axes_perm = np.random.permutation(3)
        c = np.transpose(c, axes_perm)
        
        # 2. Случайное обращение осей
        reverses = np.random.randint(0, 2, 3)
        for i, rev in enumerate(reverses):
            if rev:
                c = c[::-1] if i == 0 else (c[:, ::-1] if i == 1 else c[:, :, ::-1])
                
        # 3. Случайное дополнение (v -> 344 - v) с вероятностью 50%
        if random.random() < 0.5:
            c = self.n**3 + 1 - c
            
        # 4. Случайный модульный сдвиг координат (x -> (a*x+b)%7)
        for axis in range(3):
            a = random.choice([1, 2, 3, 4, 5, 6])  # взаимно просто с 7
            b = random.randint(0, 6)
            if axis == 0:
                c = c[np.array([(a*i + b) % 7 for i in range(7)]), :, :]
            elif axis == 1:
                c = c[:, np.array([(a*i + b) % 7 for i in range(7)]), :]
            else:
                c = c[:, :, np.array([(a*i + b) % 7 for i in range(7)])]
                
        return c

    # ─────────────────────────────────────────────
    # 3. ПОЛНАЯ ВАЛИДАЦИЯ
    # ─────────────────────────────────────────────
    def validate(self, cube):
        """Возвращает (is_valid, magic_const, список_ошибок)"""
        n = self.n
        M = self.magic_const
        errors = []
        
        # 1. Прямые линии (147 проверок)
        for axis, name in zip([0, 1, 2], ["X-rows", "Y-cols", "Z-pillars"]):
            sums = cube.sum(axis=axis)
            if not np.all(sums == M):
                errors.append(f"{name}: найдены отклонения от {M}")
                
        # 2. Пространственные диагонали (4)
        diags = [
            cube[range(n), range(n), range(n)],
            cube[range(n), range(n), range(n-1, -1, -1)],
            cube[range(n), range(n-1, -1, -1), range(n)],
            cube[range(n), range(n-1, -1, -1), range(n-1, -1, -1)]
        ]
        for i, d in enumerate(diags, 1):
            if d.sum() != M:
                errors.append(f"Space diag #{i}: {d.sum()} ≠ {M}")
                
        # 3. Диагонали сечений (42 проверки)
        for axis in range(3):
            for fix in range(n):
                if axis == 0:
                    d1 = sum(cube[fix, i, i] for i in range(n))
                    d2 = sum(cube[fix, i, n-1-i] for i in range(n))
                elif axis == 1:
                    d1 = sum(cube[i, fix, i] for i in range(n))
                    d2 = sum(cube[i, fix, n-1-i] for i in range(n))
                else:
                    d1 = sum(cube[i, i, fix] for i in range(n))
                    d2 = sum(cube[i, n-1-i, fix] for i in range(n))
                    
                if d1 != M: errors.append(f"Slice diag 1 (axis={axis}, fix={fix})")
                if d2 != M: errors.append(f"Slice diag 2 (axis={axis}, fix={fix})")
                
        # 4. Уникальность значений
        if len(np.unique(cube)) != n**3:
            errors.append("Нарушена уникальность чисел 1..343")
            
        return len(errors) == 0, M, errors

    # ─────────────────────────────────────────────
    # 4. ДЕДУПЛИКАЦИЯ (нормализация)
    # ─────────────────────────────────────────────
    def get_canonical(self, cube):
        """Возвращает каноническое представление для сравнения"""
        # Берём минимальную строку среди всех 48 симметрий
        candidates = []
        for perm in product([0, 1], repeat=3):  # обращения осей
            c = cube.copy()
            for i, p in enumerate(perm):
                if p: c = c[::-1] if i==0 else (c[:,::-1] if i==1 else c[:,:,::-1])
            for axes_perm in product([0,1,2], repeat=3):
                if len(set(axes_perm)) == 3:
                    candidates.append(np.transpose(c, axes_perm).flatten())
        return tuple(min(candidates))

    # ─────────────────────────────────────────────
    # 5. ОСНОВНОЙ ЦИКЛ ПОИСКА
    # ─────────────────────────────────────────────
    def search(self, target_count=3, max_attempts=200):
        print(f"🎯 Поиск {target_count} уникальных совершенных кубов 7×7×7...")
        print("⏳ Запуск рандомизированной генерации + валидация...\n")
        
        attempts = 0
        while len(self.found_cubes) < target_count and attempts < max_attempts:
            attempts += 1
            
            # 1. Генерация
            if random.random() < 0.6:
                cube = self.generate_algebraic_random()
            else:
                # Загружаем базу и трансформируем
                base = self.generate_algebraic_random()
                cube = self.random_symmetry_transform(base)
                
            # 2. Валидация
            valid, M, errs = self.validate(cube)
            
            if valid:
                canon = self.get_canonical(cube)
                if canon not in self.canonical_forms:
                    self.canonical_forms.add(canon)
                    self.found_cubes.append(cube)
                    print(f"✅ [{len(self.found_cubes)}] Найден новый куб! (Попытка #{attempts})")
                    print(f" Магическая константа: {M}")
                    print(f"🔍 Валидация: ПРОЙДЕНА (0 ошибок)")
                    print("-" * 50)
                else:
                    if attempts % 20 == 0:
                        print(f"⚠️ Попытка #{attempts}: куб валиден, но изоморфен ранее найденным")
            else:
                if attempts % 50 == 0:
                    print(f"❌ Попытка #{attempts}: не прошёл валидацию ({len(errs)} ошибок)")
                    
        if not self.found_cubes:
            print(" Не удалось найти валидные кубы за отведённое время.")
        return self.found_cubes

# ─────────────────────────────────────────────
# ЗАПУСК
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)  # Для воспроизводимости (уберите для полной случайности)
    random.seed(42)
    
    finder = RandomPerfectCubeFinder(n=7)
    cubes = finder.search(target_count=3, max_attempts=300)
    
    if cubes:
        print("\n📦 ГОТОВЫЕ МАССИВЫ ДЛЯ ВИЗУАЛИЗАЦИИ:")
        for i, c in enumerate(cubes, 1):
            print(f"\n🔹 Куб #{i} (shape {c.shape}):")
            print(f"   Диапазон: {c.min()}..{c.max()} | Уникальных: {len(np.unique(c))}")
            # Вывод первого слоя для проверки
            print(f"   Слой Z=0:\n{c[:, :, 0]}")
            # Сохранение
            np.save(f"random_perfect_cube_{i}.npy", c)
        print("\n💾 Все кубы сохранены как random_perfect_cube_*.npy")
