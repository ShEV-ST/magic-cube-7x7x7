import numpy as np
import hashlib
import os
import time
from itertools import product

class InfiniteMagicCubeStreamer:
    def __init__(self, save_dir="found_magic_cubes"):
        self.n = 7
        self.magic_const = 1204
        self.save_dir = save_dir
        self.found_hashes = set()
        
        # Создаем папку если нет
        os.makedirs(self.save_dir, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Загружает хеши уже найденных кубов, чтобы не дублировать между сессиями"""
        print(f"📂 Сканирование папки {self.save_dir} для загрузки истории...")
        count = 0
        for filename in os.listdir(self.save_dir):
            if filename.endswith(".npy"):
                # Имя файла = хеш куба
                self.found_hashes.add(filename.replace(".npy", ""))
                count += 1
        print(f"✅ Загружено {count} ранее найденных уникальных кубов.\n")

    def generate_candidate(self):
        """
        Генерирует случайный магический куб порядка 7.
        Использует линейную конструкцию w = 49c + 7b + a + 1 над полем Z_7.
        """
        # 1. Случайная обратимая матрица 3x3 над Z_7
        while True:
            M = np.random.randint(0, self.n, (3, 3))
            # Проверка на невырожденность (детерминант не делится на 7)
            if np.linalg.det(M) % self.n != 0:
                break
        
        # 2. Случайный вектор сдвига
        t = np.random.randint(0, self.n, 3)
        
        # 3. Векторизованное построение куба
        # Создаем сетку координат (x, y, z)
        coords = np.indices((self.n, self.n, self.n))
        # Преобразуем в векторы (3, 343)
        vecs = np.stack([coords[0].flatten(), coords[1].flatten(), coords[2].flatten()])
        
        # Применяем линейное преобразование: (M * vec + t) % 7
        res = (M @ vecs + t.reshape(3, 1)) % self.n
        
        # Распаковываем a, b, c
        # Важно: порядок разрядов должен соответствовать формуле w = 49*c + 7*b + a + 1
        # В res[0] - первая строка результата умножения, и т.д.
        # Для корректного распределения цифр используем перестановку строк M или просто назначаем:
        # Обычно a - младший (coeff 1), b - средний (coeff 7), c - старший (coeff 49)
        
        # Назначим случайно, какая строка результата за какой разряд отвечает
        perm = np.random.permutation(3)
        a = res[perm[0]].reshape(self.n, self.n, self.n)
        b = res[perm[1]].reshape(self.n, self.n, self.n)
        c = res[perm[2]].reshape(self.n, self.n, self.n)
        
        cube = c * 49 + b * 7 + a + 1
        return cube

    def validate_cube(self, cube):
        """
        Быстрая, но полная проверка магических свойств.
        Возвращает True, если куб совершенный.
        """
        # 1. Суммы по осям (самая быстрая проверка)
        # axis=0 (Z-pillars), axis=1 (Y-cols), axis=2 (X-rows) - зависит от реализации, но суть в суммах срезов
        if not (np.all(cube.sum(axis=0) == self.magic_const) and
                np.all(cube.sum(axis=1) == self.magic_const) and
                np.all(cube.sum(axis=2) == self.magic_const)):
            return False

        n = self.n
        # 2. Главные пространственные диагонали
        d1 = sum(cube[i, i, i] for i in range(n))
        d2 = sum(cube[i, i, n-1-i] for i in range(n))
        d3 = sum(cube[i, n-1-i, i] for i in range(n))
        d4 = sum(cube[i, n-1-i, n-1-i] for i in range(n))
        
        if not (d1 == d2 == d3 == d4 == self.magic_const):
            return False

        # 3. Диагонали всех сечений (плоские диагонали)
        # Проверяем XY, XZ, YZ сечения
        # XY (фикс Z)
        for k in range(n):
            if sum(cube[k, i, i] for i in range(n)) != self.magic_const: return False
            if sum(cube[k, i, n-1-i] for i in range(n)) != self.magic_const: return False
        # XZ (фикс Y) - в numpy куб[z, y, x] или [x, y, z]? 
        # Мой генератор вернул shape (7,7,7) где индексы [z_dim, y_dim, x_dim] условно, 
        # но np.indices создает [0->x, 1->y, 2->z].
        # Давайте использовать явные срезы для надежности.
        
        # Срезы по 3-й оси (Z)
        for k in range(n):
            if np.trace(cube[:, :, k]) != self.magic_const: return False
            if np.trace(np.fliplr(cube[:, :, k])) != self.magic_const: return False
            
        # Срезы по 2-й оси (Y)
        for k in range(n):
            if np.trace(cube[:, k, :]) != self.magic_const: return False
            if np.trace(np.fliplr(cube[:, k, :])) != self.magic_const: return False

        # Срезы по 1-й оси (X)
        for k in range(n):
            if np.trace(cube[k, :, :]) != self.magic_const: return False
            if np.trace(np.fliplr(cube[k, :, :])) != self.magic_const: return False

        return True

    def get_hash(self, cube):
        """Вычисляет MD5 хеш массива для проверки уникальности"""
        return hashlib.md5(cube.tobytes()).hexdigest()

    def save_cube(self, cube, file_hash):
        """Сохраняет куб в файл"""
        filename = f"{self.save_dir}/cube_{file_hash}.npy"
        np.save(filename, cube)
        print(f"💾 Сохранено: {filename}")

    def run_stream(self):
        """Основной бесконечный цикл"""
        print(f"🚀 Запуск бесконечного потока генерации (Порядок {self.n}x{self.n}x{self.n})")
        print("🎯 Ищем совершенные магические кубы (Sum = 1204)...")
        print(" Нажмите Ctrl+C для остановки.\n")
        
        found_in_session = 0
        attempts = 0
        
        try:
            while True:
                attempts += 1
                
                # 1. Генерация
                cube = self.generate_candidate()
                
                # 2. Валидация
                if self.validate_cube(cube):
                    h = self.get_hash(cube)
                    
                    # 3. Проверка уникальности
                    if h not in self.found_hashes:
                        self.found_hashes.add(h)
                        self.save_cube(cube, h)
                        found_in_session += 1
                        print(f"✨ #{found_in_session} (Попытка {attempts}): Найдена новая структура!")
                    else:
                        # Если валиден, но не уникален — это тоже хорошо, значит алгоритм работает
                        if attempts % 10 == 0:
                            print(f"⚙️ Валидный куб, но дубликат (Попытка {attempts})")
                
                # Небольшая задержка, чтобы не нагружать CPU на 100% и дать время на I/O
                time.sleep(0.05) 
                
        except KeyboardInterrupt:
            print("\n\n🛑 Поток остановлен пользователем.")
            print(f" Итого за сессию найдено: {found_in_session} уникальных кубов.")

if __name__ == "__main__":
    # Фиксируем seed для воспроизводимости (или уберите для полной случайности)
    # np.random.seed(None) 
    
    streamer = InfiniteMagicCubeStreamer(save_dir="unique_magic_cubes_db")
    streamer.run_stream()
