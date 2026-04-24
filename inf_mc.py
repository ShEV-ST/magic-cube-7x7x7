import numpy as np
import hashlib
import os
import time
# from itertools import product

class InfiniteMagicCubeStreamer:
    def __init__(self, save_dir="found_magic_cubes"):
        self.n = 7
        self.magic_const = 1204
        self.save_dir = save_dir
        self.found_hashes = set()
        self.invalid_hashes = set()
        self.tried_hashes = set()
        self.duplicate_candidate_count = 0
        self.invalid_candidate_count = 0
        self.found_in_session = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.report_interval = 60
        self.invalid_history_path = os.path.join(self.save_dir, "invalid_hashes.txt")
        self.invalid_save_batch = []
        self.invalid_flush_interval = 1000
        
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
        print(f"✅ Загружено {count} ранее найденных уникальных кубов.")
        self._load_invalid_history()

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

    def _load_invalid_history(self):
        """Загружает хеши невалидных кандидатов между запусками"""
        if not os.path.exists(self.invalid_history_path):
            print(f"✅ Нет истории невалидных кандидатов ({self.invalid_history_path}).\n")
            return
        count = 0
        with open(self.invalid_history_path, "r", encoding="utf-8") as f:
            for line in f:
                h = line.strip()
                if h:
                    self.invalid_hashes.add(h)
                    count += 1
        print(f"✅ Загружено {count} ранее невалидных кандидатов.\n")

    def _flush_invalid_history(self):
        if not self.invalid_save_batch:
            return
        with open(self.invalid_history_path, "a", encoding="utf-8") as f:
            for h in self.invalid_save_batch:
                f.write(h + "\n")
        self.invalid_save_batch.clear()

    def _get_cpu_load_info(self):
        try:
            load1, load5, load15 = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            load_pct = min(100.0, load1 / cpu_count * 100)
            return load1, load5, load15, cpu_count, load_pct
        except (AttributeError, OSError):
            return None

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
        
        attempts = 0
        
        try:
            while True:
                attempts += 1
                
                # 1. Генерация
                cube = self.generate_candidate()
                candidate_hash = self.get_hash(cube)
                if candidate_hash in self.tried_hashes or candidate_hash in self.invalid_hashes or candidate_hash in self.found_hashes:
                    self.duplicate_candidate_count += 1
                    continue
                self.tried_hashes.add(candidate_hash)
                
                # 2. Валидация
                if self.validate_cube(cube):
                    if candidate_hash not in self.found_hashes:
                        self.found_hashes.add(candidate_hash)
                        self.save_cube(cube, candidate_hash)
                        self.found_in_session += 1
                        print(f"✨ #{self.found_in_session} (Попытка {attempts}): Найдена новая структура!")
                    else:
                        # Если валиден, но не уникален — логируем всегда для отладки
                        print(f"⚙️ Валидный куб, но дубликат (Попытка {attempts})")
                else:
                    self.invalid_candidate_count += 1
                    if candidate_hash not in self.invalid_hashes:
                        self.invalid_hashes.add(candidate_hash)
                        self.invalid_save_batch.append(candidate_hash)
                        if len(self.invalid_save_batch) >= self.invalid_flush_interval:
                            self._flush_invalid_history()
                
                # Ускорение: минимальная пауза, только если загрузка сильно выше 100%
                cpu_info = self._get_cpu_load_info()
                if cpu_info is not None:
                    load1, load5, load15, cpu_count, load_pct = cpu_info
                    if load_pct > 120:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)

                # Ежеминутный отчёт о прогрессе
                now = time.time()
                if now - self.last_report_time >= self.report_interval:
                    elapsed = now - self.start_time
                    speed = attempts / elapsed if elapsed > 0 else 0
                    cpu_line = ""
                    if cpu_info is not None:
                        cpu_line = f", load1={load1:.2f}, cpus={cpu_count}, load={load_pct:.0f}%"
                    print(f"📊 Отчёт: проверено {attempts} кандидатов, найдено {self.found_in_session} уникальных, "
                          f"неуспешных {self.invalid_candidate_count}, повторов {self.duplicate_candidate_count}, "
                          f"скорость {speed:.1f}/сек{cpu_line}, время {int(elapsed)} сек")
                    self._flush_invalid_history()
                    self.last_report_time = now
                
        except KeyboardInterrupt:
            self._flush_invalid_history()
            print("\n\n🛑 Поток остановлен пользователем.")
            print(f" Итого за сессию найдено: {self.found_in_session} уникальных кубов.")

if __name__ == "__main__":
    # Фиксируем seed для воспроизводимости (или уберите для полной случайности)
    # np.random.seed(None) 
    
    streamer = InfiniteMagicCubeStreamer(save_dir="unique_magic_cubes_db")
    streamer.run_stream()
