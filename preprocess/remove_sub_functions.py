import os
from typing import List

def remove_sub_functions_from_file(
    input_path: str,
    output_path: str | None = None
) -> int:
    """
    .c 파일에서 /* Function: sub_로 시작하는 주석부터 다음 /*로 시작하는 주석 전까지의 블록을 삭제합니다.

    Args:
        input_path (str): 입력 C 파일 경로
        output_path (str | None): 결과를 저장할 파일 경로. None이면 입력 파일을 덮어씀
    Returns:
        int: 제거된 sub_ 함수 블록의 개수
    """
    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    to_remove: List[tuple[int, int]] = []

    while i < n:
        if lines[i].lstrip().startswith("/* Function: sub_"):
            start = i
            i += 1
            while i < n and not (lines[i].lstrip().startswith("/*") and i != start):
                i += 1
            end = i
            to_remove.append((start, end))
        else:
            i += 1

    for start, end in reversed(to_remove):
        del lines[start:end]

    save_path = output_path if output_path else input_path
    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return len(to_remove)

def collect_c_files(path: str) -> List[str]:
    """
    주어진 경로에서 모든 .c 파일의 경로를 재귀적으로 수집합니다.

    Args:
        path (str): 파일 또는 디렉토리 경로
    Returns:
        List[str]: .c 파일 경로 리스트
    """
    c_files: List[str] = []
    if os.path.isfile(path) and path.endswith(".c"):
        c_files.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".c"):
                    c_files.append(os.path.join(root, file))
    print(f"수집된 .c 파일 개수: {len(c_files)}")
    print(f"수집된 .c 파일 목록: {c_files}")
    return c_files

def count_function_blocks_in_file(input_path: str) -> tuple[int, int]:
    """
    .c 파일에서 /* Function: ... */ 블록의 총 개수와 /* Function: sub_... */ 블록의 개수를 셉니다.

    Args:
        input_path (str): 입력 C 파일 경로
    Returns:
        tuple[int, int]: (전체 함수 블록 개수, sub_ 함수 블록 개수)
    """
    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    total_blocks = 0
    sub_blocks = 0
    while i < n:
        if lines[i].lstrip().startswith("/* Function:"):
            total_blocks += 1
            if lines[i].lstrip().startswith("/* Function: sub_"):
                sub_blocks += 1
            # 다음 블록 시작 전까지 이동
            i += 1
            while i < n and not (lines[i].lstrip().startswith("/* Function:") and not lines[i].lstrip().startswith("/* Function: sub_")):
                if lines[i].lstrip().startswith("/* Function:"):
                    break
                i += 1
        else:
            i += 1
    return total_blocks, sub_blocks

def process_path(
    path: str,
    output_dir: str | None = None
) -> None:
    """
    파일 또는 디렉토리를 입력받아, .c 파일에 대해 sub_ 함수 블록을 삭제합니다.
    디렉토리일 경우 재귀적으로 모든 하위 .c 파일을 처리합니다.
    처리 상황을 출력합니다.

    Args:
        path (str): 파일 또는 디렉토리 경로
        output_dir (str | None): 결과를 저장할 디렉토리. None이면 원본 파일을 덮어씀
    """
    c_files = collect_c_files(path)
    total = len(c_files)
    if total == 0:
        print(f"처리할 .c 파일이 없습니다: {path}")
        return
    print(f"총 {total}개의 .c 파일을 처리합니다.")
    total_removed = 0
    total_func_blocks = 0
    total_sub_blocks = 0
    for idx, file_path in enumerate(c_files, 1):
        output_path = None
        if output_dir:
            rel_dir = os.path.relpath(os.path.dirname(file_path), path)
            out_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, os.path.basename(file_path))
        print(f"[{idx}/{total}] 처리 중: {file_path}")
        func_blocks, sub_blocks = count_function_blocks_in_file(file_path)
        total_func_blocks += func_blocks
        total_sub_blocks += sub_blocks
        removed = remove_sub_functions_from_file(file_path, output_path)
        total_removed += removed
    print("모든 파일 처리가 완료되었습니다.")
    print(f"전체 함수 블록 개수: {total_func_blocks}")
    print(f"sub_ 함수 블록 개수: {total_sub_blocks}")
    print(f"총 제거된 sub_ 함수 블록 개수: {total_removed}")

if __name__ == "__main__":
    # 하드코딩된 디렉토리 경로를 사용하여 처리
    target_dir = r"E:\symbol-infer\dataset\SymGen\strip\O3_division_v2"
    # 필요시 output_dir도 지정 가능
    # output_dir = "output_dir_path"
    process_path(target_dir) 