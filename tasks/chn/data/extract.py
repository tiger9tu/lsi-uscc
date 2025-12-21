#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path
from typing import Optional, Dict


# ===== 正则模式 =====
PATTERNS = {
    "LASSCF": re.compile(
        r"^\s*LASSCF.*?energy\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*$"
    ),
    "CASCI": re.compile(
        r"^\s*CASCI.*?energy\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*$"
    ),
    # 兼容：LASSI[1,2]energy = ...
    "LASSI": re.compile(
        r"^\s*LASSI.*?energy\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*$"
    ),
    # 兼容：energy: 或 energy =
    "LSILCCSD": re.compile(
        r"^\s*LSILCCSD\s+energy\s*[:=]\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*$"
    ),
}


def parse_one_file(path: Path) -> Dict[str, Optional[float]]:
    """从单个文件中提取能量"""
    result = {k: None for k in PATTERNS}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for key, pat in PATTERNS.items():
                m = pat.match(line)
                if m:
                    result[key] = float(m.group(1))

    return result


def main():
    # ===== 当前目录 =====
    workdir = Path(__file__).resolve().parent
    output_csv = workdir / "energies.csv"

    rows = []

    # ===== 扫描 chn-* 文件 =====
    for file in sorted(workdir.iterdir()):
        if file.is_file() and file.name.startswith("chn-"):
            energies = parse_one_file(file)
            rows.append([
                file.name,
                energies["LASSCF"],
                energies["CASCI"],
                energies["LASSI"],
                energies["LSILCCSD"],
            ])

    # ===== 写 CSV =====
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "LASSCF", "CASCI", "LASSI", "LSILCCSD"])
        writer.writerows(rows)

    print(f"✓ 已处理 {len(rows)} 个文件")
    print(f"✓ 输出文件: {output_csv}")


if __name__ == "__main__":
    main()
