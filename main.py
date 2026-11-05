"""Mini NPU Simulator.

입력 패턴과 필터를 위치별로 곱한 뒤 모두 더하는 MAC(Multiply-Accumulate) 연산으로
패턴이 십자가(Cross)인지 X인지 판별하고, 크기별 연산 시간을 측정한다.

실행: python main.py [data.json 경로]
     (경로를 생략하면 main.py와 같은 폴더의 data.json을 사용한다)
"""

import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

# ----------------------------------------------------------------------
# 설정값
# ----------------------------------------------------------------------
EPSILON = 1e-9           # 두 점수 차이가 이보다 작으면 동점으로 본다.
REPEAT_COUNT = 10        # 성능 측정 반복 횟수
MODE1_SIZE = 3           # 모드 1(사용자 입력) 행렬 크기
BENCHMARK_SIZES = (3, 5, 13, 25)
GENERATOR_MAX_SIZE = 51

LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

# 다양한 표기를 내부 표준 라벨(Cross, X)로 정규화하기 위한 표.
# 비교 전에 앞뒤 공백 제거 + 소문자 변환을 거치므로 'Cross', ' X ' 등도 처리된다.
LABEL_ALIASES = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data.json"
FILTER_KEY_PATTERN = re.compile(r"^size_(\d+)$")
PATTERN_KEY_PATTERN = re.compile(r"^size_(\d+)_(\d+)$")

SECTION_LINE = "#" + "-" * 39


# ----------------------------------------------------------------------
# 데이터 구조: n×n 2차원 배열
# ----------------------------------------------------------------------
class Matrix:
    """n×n 크기의 2차원 숫자 배열. 특정 위치의 값을 저장하고 읽을 수 있다."""

    def __init__(self, size, fill=0.0):
        if size < 1:
            raise ValueError("크기는 1 이상이어야 합니다.")
        self.size = size
        self.rows = [[float(fill)] * size for _ in range(size)]

    @classmethod
    def from_rows(cls, rows, expected_size=None):
        """2차원 리스트를 검증해 Matrix로 만든다. 형식이 틀리면 ValueError."""
        if not isinstance(rows, list) or not rows:
            raise ValueError("2차원 배열(리스트의 리스트)이 아닙니다.")
        size = len(rows) if expected_size is None else expected_size
        if len(rows) != size:
            raise ValueError(f"행 수 불일치: {size}행이 필요하지만 {len(rows)}행입니다.")
        for r, row in enumerate(rows, start=1):
            if not isinstance(row, list):
                raise ValueError(f"{r}행이 배열이 아닙니다.")
            if len(row) != size:
                raise ValueError(
                    f"열 수 불일치: {r}행에 {size}개가 필요하지만 {len(row)}개입니다."
                )
            for c, value in enumerate(row, start=1):
                if not is_number(value):
                    raise ValueError(f"({r}, {c}) 위치의 값 {value!r}은(는) 숫자가 아닙니다.")
        matrix = cls(size)
        matrix.rows = [[float(value) for value in row] for row in rows]
        return matrix

    def get(self, row, col):
        return self.rows[row][col]

    def set(self, row, col, value):
        self.rows[row][col] = float(value)

    def flatten(self):
        """길이 N²의 1차원 리스트로 변환한다. (보너스: 메모리 접근 최적화)"""
        return [value for row in self.rows for value in row]


def is_number(value):
    # bool은 int의 하위 타입이라 True/False가 1/0으로 통과하지 않도록 따로 막는다.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


# ----------------------------------------------------------------------
# MAC 연산 / 판정
# ----------------------------------------------------------------------
def mac(pattern, kernel):
    """두 n×n 행렬의 같은 위치 값을 곱해 모두 더한 점수를 반환한다."""
    if pattern.size != kernel.size:
        raise ValueError(
            f"크기 불일치: 패턴 {pattern.size}×{pattern.size}, "
            f"필터 {kernel.size}×{kernel.size}"
        )
    a = pattern.rows
    b = kernel.rows
    total = 0.0
    for i in range(pattern.size):
        for j in range(pattern.size):
            total += a[i][j] * b[i][j]
    return total


def mac_flat(pattern_flat, kernel_flat):
    """1차원(길이 N²)으로 펼친 두 배열의 MAC 점수를 반환한다. (보너스)"""
    total = 0.0
    for k in range(len(pattern_flat)):
        total += pattern_flat[k] * kernel_flat[k]
    return total


def decide(score_a, score_b, label_a, label_b):
    """점수가 큰 쪽의 라벨을 반환한다. 차이가 EPSILON보다 작으면 None(동점)."""
    if abs(score_a - score_b) < EPSILON:
        return None
    return label_a if score_a > score_b else label_b


def normalize_label(raw):
    """'+', 'cross', 'x' 같은 다양한 표기를 표준 라벨(Cross/X)로 바꾼다. 실패 시 None."""
    if not isinstance(raw, str):
        return None
    return LABEL_ALIASES.get(raw.strip().lower())


def measure_average_ms(func, *args):
    """func(*args)를 REPEAT_COUNT번 실행해 1회 평균 시간(ms)을 반환한다.

    입력/출력/파일 읽기는 제외하고 연산 함수 호출 구간만 측정한다.
    """
    func(*args)  # 워밍업: 첫 호출의 일회성 오버헤드를 측정에서 제외한다.
    total = 0.0
    for _ in range(REPEAT_COUNT):
        start = time.perf_counter()
        func(*args)
        total += time.perf_counter() - start
    return total / REPEAT_COUNT * 1000


# ----------------------------------------------------------------------
# 패턴 생성기 (보너스)
# ----------------------------------------------------------------------
def generate_cross(size):
    """가운데 행과 가운데 열이 1인 N×N 십자가 패턴을 만든다."""
    matrix = Matrix(size)
    center = size // 2
    for k in range(size):
        matrix.set(center, k, 1)
        matrix.set(k, center, 1)
    return matrix


def generate_x(size):
    """두 대각선이 1인 N×N X 패턴을 만든다."""
    matrix = Matrix(size)
    for k in range(size):
        matrix.set(k, k, 1)
        matrix.set(k, size - 1 - k, 1)
    return matrix


# ----------------------------------------------------------------------
# 출력 도우미
# ----------------------------------------------------------------------
def print_section(title):
    print()
    print(SECTION_LINE)
    print(f"# {title}")
    print(SECTION_LINE)


def format_value(value):
    return str(int(value)) if value.is_integer() else str(value)


def print_matrix(matrix, indent="  "):
    for row in matrix.rows:
        print(indent + " ".join(format_value(value) for value in row))


def display_width(text):
    # 한글 등 전각 문자는 터미널에서 2칸을 차지한다.
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text, width, align="left"):
    """터미널 표시 너비 기준으로 공백을 채워 열을 맞춘다."""
    space = " " * max(0, width - display_width(text))
    return text + space if align == "left" else space + text


def print_table(headers, rows, widths):
    """첫 열은 왼쪽, 나머지 열은 오른쪽 정렬로 표를 출력한다."""
    def line(cells):
        return "".join(
            pad(cell, width, "left" if index == 0 else "right")
            for index, (cell, width) in enumerate(zip(cells, widths))
        )

    print(line(headers))
    print("-" * sum(widths))
    for row in rows:
        print(line(row))


def print_performance_table(rows):
    """rows: [(크기 N, 평균 시간 ms), ...]"""
    print_table(
        ["크기(N×N)", "평균 시간(ms)", "연산 횟수(N²)"],
        [[f"{size}×{size}", f"{avg_ms:.4f}", str(size * size)] for size, avg_ms in rows],
        [10, 16, 16],
    )


def score_text(score):
    return "-" if score is None else repr(score)


# ----------------------------------------------------------------------
# 입력 도우미
# ----------------------------------------------------------------------
def read_choice(prompt, valid_choices):
    guide = ", ".join(valid_choices)
    while True:
        value = input(prompt).strip()
        if value in valid_choices:
            return value
        print(f"입력 오류: {guide} 중 하나를 입력하세요.")


def read_int(prompt, min_value, max_value):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print(f"입력 오류: {min_value}~{max_value} 사이의 정수를 입력하세요.")
            continue
        if min_value <= value <= max_value:
            return value
        print(f"입력 오류: {min_value}~{max_value} 사이의 정수를 입력하세요.")


def read_matrix(name, size):
    """size줄을 한 줄씩(공백 구분) 입력받아 Matrix로 반환한다. 오류 줄은 다시 입력받는다."""
    print(f"{name} ({size}줄 입력, 공백 구분)")
    rows = []
    while len(rows) < size:
        tokens = input().split()
        if len(tokens) != size:
            print(
                f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요. "
                f"(입력된 값 {len(tokens)}개 → {len(rows) + 1}번째 줄을 다시 입력)"
            )
            continue
        try:
            values = [float(token) for token in tokens]
        except ValueError:
            print(
                "입력 형식 오류: 숫자로 바꿀 수 없는 값이 있습니다. "
                f"({len(rows) + 1}번째 줄을 다시 입력)"
            )
            continue
        if not all(math.isfinite(value) for value in values):
            print(
                "입력 형식 오류: nan, inf는 사용할 수 없습니다. "
                f"({len(rows) + 1}번째 줄을 다시 입력)"
            )
            continue
        rows.append(values)
    return Matrix.from_rows(rows)


# ----------------------------------------------------------------------
# 모드 1: 사용자 입력 (3×3)
# ----------------------------------------------------------------------
def run_user_input_mode():
    print_section("[1] 필터 입력")
    filter_a = read_matrix("필터 A", MODE1_SIZE)
    print()
    filter_b = read_matrix("필터 B", MODE1_SIZE)
    print("\n✓ 필터 A, B 저장 완료")
    print("필터 A:")
    print_matrix(filter_a)
    print("필터 B:")
    print_matrix(filter_b)

    print_section("[2] 패턴 입력")
    pattern = read_matrix("패턴", MODE1_SIZE)

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    # 판정에 필요한 MAC 2번(A, B)의 1회 평균을 MAC 1회 기준 시간으로 본다.
    avg_ms = (
        measure_average_ms(mac, pattern, filter_a)
        + measure_average_ms(mac, pattern, filter_b)
    ) / 2
    decision = decide(score_a, score_b, "A", "B")

    print_section("[3] MAC 결과")
    print(f"A 점수: {score_a!r}")
    print(f"B 점수: {score_b!r}")
    print(f"연산 시간(평균/{REPEAT_COUNT}회, MAC 1회 기준): {avg_ms:.4f} ms")
    if decision is None:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    else:
        print(f"판정: {decision}")

    print_section(f"[4] 성능 분석 ({MODE1_SIZE}×{MODE1_SIZE}, 평균/{REPEAT_COUNT}회)")
    print_performance_table([(MODE1_SIZE, avg_ms)])


# ----------------------------------------------------------------------
# 모드 2: data.json 분석
# ----------------------------------------------------------------------
def load_json(path):
    """JSON 파일을 읽어 반환한다. 실패하면 안내 후 None."""
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"✗ 데이터 파일을 찾을 수 없습니다: {path}")
    except (OSError, ValueError) as error:
        # json.JSONDecodeError, UnicodeDecodeError도 ValueError의 하위 클래스다.
        print(f"✗ 데이터 파일을 읽을 수 없습니다: {error}")
    return None


def load_filters(raw_filters):
    """filters 항목을 {N: {"Cross": Matrix, "X": Matrix}}로 읽는다.

    잘못된 필터는 건너뛰고, 해당 크기의 패턴이 FAIL 사유로 쓸 수 있도록
    {N: 오류 메시지}로 따로 모아 반환한다.
    """
    filters = {}
    errors = {}
    if not isinstance(raw_filters, dict):
        print("✗ filters 항목이 없거나 객체 형식이 아닙니다.")
        return filters, errors

    for key, raw_pair in raw_filters.items():
        match = FILTER_KEY_PATTERN.match(str(key).strip())
        if not match:
            print(f"✗ {key}: 키 형식 오류 (size_N 형식이어야 합니다)")
            continue
        size = int(match.group(1))
        try:
            filters[size] = parse_filter_pair(raw_pair, size)
        except ValueError as error:
            errors[size] = f"size_{size} 필터 오류: {error}"
            print(f"✗ {key} 필터 로드 실패: {error}")
            continue
        print(f"✓ {key:<7} 필터 로드 완료 (Cross, X)")
    return filters, errors


def parse_filter_pair(raw_pair, size):
    if not isinstance(raw_pair, dict):
        raise ValueError("cross/x 필터를 담은 객체가 아닙니다.")
    pair = {}
    for raw_label, rows in raw_pair.items():
        label = normalize_label(raw_label)
        if label is None:
            raise ValueError(f"알 수 없는 필터 라벨 '{raw_label}'")
        try:
            pair[label] = Matrix.from_rows(rows, expected_size=size)
        except ValueError as error:
            raise ValueError(f"{label} 필터 - {error}") from None
    missing = [label for label in (LABEL_CROSS, LABEL_X) if label not in pair]
    if missing:
        raise ValueError(f"{', '.join(missing)} 필터가 없습니다.")
    return pair


def evaluate_case(key, case, filters, filter_errors):
    """패턴 하나를 판정한다. 데이터 문제는 예외 대신 FAIL 결과로 돌려준다."""
    result = {
        "key": key,
        "cross": None,
        "x": None,
        "decision": None,
        "expected": None,
        "passed": False,
        "reason": "",
    }
    try:
        match = PATTERN_KEY_PATTERN.match(str(key).strip())
        if not match:
            raise ValueError("키 형식 오류: size_{N}_{idx} 형식이 아닙니다.")
        size = int(match.group(1))
        if not isinstance(case, dict):
            raise ValueError("패턴 항목이 객체 형식이 아닙니다.")

        raw_expected = case.get("expected")
        expected = normalize_label(raw_expected)
        if expected is None:
            raise ValueError(f"expected 라벨 {raw_expected!r}을(를) 해석할 수 없습니다.")
        result["expected"] = expected

        if size in filter_errors:
            raise ValueError(filter_errors[size])
        if size not in filters:
            raise ValueError(f"size_{size} 필터가 없습니다.")
        try:
            pattern = Matrix.from_rows(case.get("input"), expected_size=size)
        except ValueError as error:
            raise ValueError(f"패턴 오류 - {error}") from None
    except ValueError as error:
        result["reason"] = str(error)
        return result

    cross_score = mac(pattern, filters[size][LABEL_CROSS])
    x_score = mac(pattern, filters[size][LABEL_X])
    decision = decide(cross_score, x_score, LABEL_CROSS, LABEL_X)
    result["cross"] = cross_score
    result["x"] = x_score
    result["decision"] = decision or LABEL_UNDECIDED
    result["passed"] = decision == expected
    if decision is None:
        result["reason"] = (
            f"동점(UNDECIDED) 규칙 적용 - "
            f"|Cross-X| = {abs(cross_score - x_score):.3g} < {EPSILON}"
        )
    elif decision != expected:
        result["reason"] = f"판정 {decision} ≠ expected {expected}"
    return result


def print_case_result(result):
    print(f"--- {result['key']} ---")
    print(f"Cross 점수: {score_text(result['cross'])}")
    print(f"X 점수: {score_text(result['x'])}")
    status = "PASS" if result["passed"] else f"FAIL ({result['reason']})"
    print(
        f"판정: {result['decision'] or '-'} | "
        f"expected: {result['expected'] or '-'} | {status}"
    )


def benchmark(sizes, filters):
    """크기별 MAC 평균 시간을 2차원/1차원 방식으로 측정한다.

    data.json에 해당 크기 필터가 있으면 그 Cross 필터를, 없으면(3×3 등)
    패턴 생성기로 만든 십자가 필터를 사용한다. 입력 패턴은 생성기의 X 패턴이다.
    """
    results = []
    for size in sizes:
        kernel = filters[size][LABEL_CROSS] if size in filters else generate_cross(size)
        pattern = generate_x(size)
        avg_2d = measure_average_ms(mac, pattern, kernel)
        # 펼치기는 한 번만 미리 해 두고, 연산 구간만 비교한다.
        avg_1d = measure_average_ms(mac_flat, pattern.flatten(), kernel.flatten())
        results.append((size, avg_2d, avg_1d))
    return results


def print_optimization_table(results):
    rows = []
    for size, avg_2d, avg_1d in results:
        change = (avg_1d - avg_2d) / avg_2d * 100 if avg_2d > 0 else 0.0
        rows.append([f"{size}×{size}", f"{avg_2d:.4f}", f"{avg_1d:.4f}", f"{change:+.1f}%"])
    print_table(["크기(N×N)", "2차원(ms)", "1차원(ms)", "변화율"], rows, [10, 12, 12, 10])


def run_json_mode(data_path):
    data = load_json(data_path)
    if data is None:
        return
    if not isinstance(data, dict):
        print("✗ 데이터 파일의 최상위 구조가 객체(dict)가 아닙니다.")
        return

    print_section("[1] 필터 로드")
    filters, filter_errors = load_filters(data.get("filters"))

    print_section("[2] 패턴 분석 (라벨 정규화 적용)")
    raw_patterns = data.get("patterns")
    results = []
    if not isinstance(raw_patterns, dict):
        print("✗ patterns 항목이 없거나 객체 형식이 아닙니다.")
    else:
        for key, case in raw_patterns.items():
            result = evaluate_case(key, case, filters, filter_errors)
            print_case_result(result)
            results.append(result)

    sizes = sorted(set(BENCHMARK_SIZES) | set(filters))
    bench = benchmark(sizes, filters)

    print_section(f"[3] 성능 분석 (평균/{REPEAT_COUNT}회)")
    print_performance_table([(size, avg_2d) for size, avg_2d, _ in bench])

    print_section("[3-1] 보너스: 1차원 배열 최적화 비교 (동일 입력)")
    print_optimization_table(bench)

    print_section("[4] 결과 요약")
    passed = sum(1 for result in results if result["passed"])
    failed = [result for result in results if not result["passed"]]
    print(f"총 테스트: {len(results)}개")
    print(f"통과: {passed}개")
    print(f"실패: {len(failed)}개")
    if failed:
        print("\n실패 케이스:")
        for result in failed:
            print(f"- {result['key']}: {result['reason']}")


# ----------------------------------------------------------------------
# 모드 3: 패턴 생성기 (보너스)
# ----------------------------------------------------------------------
def run_generator_mode():
    print_section("[1] 패턴 생성")
    size = read_int(
        f"생성할 패턴 크기 N (3~{GENERATOR_MAX_SIZE}, 홀수 권장): ",
        3,
        GENERATOR_MAX_SIZE,
    )
    cross = generate_cross(size)
    x_pattern = generate_x(size)
    if size % 2 == 0:
        print("※ 짝수 크기는 정중앙 칸이 없어 가운데 두 줄 중 아래/오른쪽 줄을 기준으로 생성합니다.")
    if size <= 15:
        print(f"\nCross 패턴 ({size}×{size}):")
        print_matrix(cross)
        print(f"\nX 패턴 ({size}×{size}):")
        print_matrix(x_pattern)
    else:
        print(f"\n({size}×{size}은 화면 출력을 생략합니다.)")

    print_section("[2] 생성 패턴 판정 (생성된 Cross/X를 필터로 재사용)")
    for name, pattern, expected in (
        ("Cross 패턴", cross, LABEL_CROSS),
        ("X 패턴", x_pattern, LABEL_X),
    ):
        cross_score = mac(pattern, cross)
        x_score = mac(pattern, x_pattern)
        decision = decide(cross_score, x_score, LABEL_CROSS, LABEL_X) or LABEL_UNDECIDED
        status = "PASS" if decision == expected else "FAIL"
        print(
            f"{name}: Cross 점수 {format_value(cross_score)}, "
            f"X 점수 {format_value(x_score)} → 판정 {decision} | {status}"
        )

    print_section(f"[3] 성능 분석 ({size}×{size}, 평균/{REPEAT_COUNT}회)")
    print_performance_table([(size, measure_average_ms(mac, x_pattern, cross))])


# ----------------------------------------------------------------------
# 메인 메뉴
# ----------------------------------------------------------------------
def main():
    # 출력이 파이프/파일로 연결돼도 한글과 기호(✓, ×)가 깨지지 않도록 UTF-8로 고정한다.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    print("=== Mini NPU Simulator ===")
    try:
        while True:
            print("\n[모드 선택]")
            print(f"1. 사용자 입력 ({MODE1_SIZE}x{MODE1_SIZE})")
            print("2. data.json 분석")
            print("3. 패턴 생성기 (보너스)")
            print("0. 종료")
            choice = read_choice("선택: ", ("1", "2", "3", "0"))
            if choice == "0":
                print("프로그램을 종료합니다.")
                return
            if choice == "1":
                run_user_input_mode()
            elif choice == "2":
                run_json_mode(data_path)
            else:
                run_generator_mode()
    except (KeyboardInterrupt, EOFError):
        print("\n입력이 중단되어 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
