"""
统一复杂度：表达式树节点数，运算符、变量、常数各计 1 个节点（含符号的系数计 1）。
口径同 EditFlowSR 附录 D：Complexity(f) = |nodes(f)|。

支持两种格式：
- 中缀（Operon/FFX/ITEA）：Python ast 解析后数节点
- 前缀括号形式（MRGP，如 div(mul(X1, X2), X3)）：递归下降数节点
"""
import ast
import re


def _count_infix(expr):
    tree = ast.parse(expr, mode='eval')
    # Call 的函数名 Name 已随 Call 计 1 个节点，避免重复计数
    func_name_ids = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    cnt = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.Name) and id(node) in func_name_ids:
            continue
        if isinstance(node, ast.BinOp):
            cnt += 1
        elif isinstance(node, ast.UnaryOp):
            # 负常数 -(c) 只计 1 个节点，负号并入系数
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
                continue
            cnt += 1
        elif isinstance(node, (ast.Call, ast.Name, ast.Constant)):
            cnt += 1
    return cnt


def _count_prefix(expr):
    tokens = re.findall(
        r'[A-Za-z_][A-Za-z_0-9]*|'
        r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|'
        r'[(),]',
        expr)
    pos = 0

    def parse_node():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        cnt = 1  # 函数名/变量/常数自身 1 个节点
        if pos < len(tokens) and tokens[pos] == '(':
            pos += 1
            while True:
                cnt += parse_node()
                if tokens[pos] == ',':
                    pos += 1
                elif tokens[pos] == ')':
                    pos += 1
                    break
        return cnt

    n = parse_node()
    assert pos == len(tokens), f'前缀表达式解析不完整: {expr[:80]}'
    return n


def unified_complexity(expr):
    if not isinstance(expr, str) or not expr.strip():
        return None
    infix = not re.match(r'^\s*[A-Za-z_][A-Za-z_0-9]*\s*\(', expr)
    if infix:
        return _count_infix(expr)
    return _count_prefix(expr)
