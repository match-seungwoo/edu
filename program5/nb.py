"""
nb.py — 의존성 없는 Jupyter 노트북(.ipynb) 빌더.

세션 노트북은 '파이썬 빌더 스크립트'로 정의하고 이 헬퍼로 .ipynb를 만든다.
→ 노트북이 코드로 관리되어 리뷰·재생성·일괄수정이 쉽다.

사용:
    from nb import md, code, save
    cells = [ md("# 제목"), code("print('hi')") ]
    save(cells, "session1/session1.ipynb")
"""
import json


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


def _split(src):
    # ipynb source는 줄 끝 \n 유지하는 문자열 리스트가 표준
    lines = src.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [""]


def save(cells, path):
    for c in cells:
        c["source"] = _split(c["source"]) if isinstance(c["source"], str) else c["source"]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
            "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("saved", path, f"({len(cells)} cells)")
