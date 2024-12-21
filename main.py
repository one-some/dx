exp = "3x^2 + 9x"
exp = "6x^3 -9x + 4"
# exp = "x^-4-9y-3+8y-2+12"
# kexp = "12*2/44^(2+1)"

class Operator:
    def __init__(self, precedence: int, left_associativity: bool):
        self.precedence = precedence
        self.left_associativity = left_associativity

pemdas = {
    "^": Operator(3, False),
    "*": Operator(2, True),
    "/": Operator(2, True),
    "+": Operator(1, True),
    "-": Operator(1, True),
}
functions = ["sin", "max"]

tokens = [""]

for char in exp.replace(" ", ""):
    if char in ["(", ")", ","] or char in pemdas:
        tokens.append(char)
        tokens.append("")
        continue

    if (
        tokens[-1].isdecimal() and not (char.isdecimal() or char == ".")
    ):
        tokens.append("")

    tokens[-1] += char

# Add mult
tok = []
for t in tokens:
    if not t: continue

    if (
        t in pemdas
        or t in functions
        or not tok
        or tok[-1] in pemdas
        or tok[-1] in functions
    ):
        tok.append(t)
        continue

    tok.append("*")
    tok.append(t)

tokens = tok

print(exp)
print(tokens)

output = []
stack = []

for token in tokens:
    if token in functions:
        stack.insert(0, token)
    elif token == "(":
        stack.insert(0, token)
    elif token == ")":
        while stack and stack[0] != "(":
            output.append(stack.pop(0))

        x = stack.pop(0)
    elif token == ",":
        while stack[0] != "(":
            output.append(stack.pop(0))
    elif token in pemdas:
        while (
            stack
            and stack[0] != "("
            and (
                stack[0] in functions
                or pemdas[stack[0]].precedence > pemdas[token].precedence
                or (
                    pemdas[stack[0]].precedence == pemdas[token].precedence
                    and pemdas[token].left_associativity
                )
            )
        ):
            output.append(stack.pop(0))
        stack.insert(0, token)
    else:
        output.append(token)

while stack:
    assert stack[0] != "("
    output.append(stack.pop(0))

print("===")
print(output)

output = [float(x) if x.isdecimal() else x for x in output]

# Eval

stack = []

class Operation:
    operator = "?"

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"({type(self).__name__}: {self.lhs}, {self.rhs})"

    def eval(self):
        # UGLY
        lhs = self.lhs.eval() if isinstance(self.lhs, Operation) else self.lhs
        rhs = self.rhs.eval() if isinstance(self.rhs, Operation) else self.rhs
        return self.i_eval(lhs, rhs)
    
    def to_str(self):
        # UGLY
        lhs = self.lhs.to_str() if isinstance(self.lhs, Operation) else self.lhs
        rhs = self.rhs.to_str() if isinstance(self.rhs, Operation) else self.rhs
        return f"({lhs} {self.operator} {rhs})"

    def optimized(self):
        node = self._optimized()
        if not isinstance(node, Operation): return node

        if isinstance(node.lhs, Operation): node.lhs = node.lhs.optimized()
        if isinstance(node.rhs, Operation): node.rhs = node.rhs.optimized()
        return node

    def _optimized(self):
        return self

    @staticmethod
    def i_eval(lhs, rhs):
        raise NotImplementedError

class AddOperation(Operation):
    operator = "+"

    @staticmethod
    def i_eval(lhs, rhs):
        return lhs + rhs

    def _optimized(self):
        if self.rhs == 0: return self.lhs
        if self.lhs == 0: return self.rhs
        return self

class SubOperation(Operation):
    operator = "-"

    @staticmethod
    def i_eval(lhs, rhs):
        return lhs - rhs

    def _optimized(self):
        # Careful!
        if self.rhs == 0: return self.lhs
        return self

class MultOperation(Operation):
    operator = "*"

    @staticmethod
    def i_eval(lhs, rhs):
        return lhs * rhs

    def _optimized(self):
        arg_pool = []
        node_pool = [self]

        while node_pool:
            node = node_pool.pop()
            hs = node.lhs, node.rhs

            potential_arg_pool = list(arg_pool)

            for s in hs:
                if isinstance(s, MultOperation):
                    node_pool.append(s)
                else:
                    potential_arg_pool.append(s)

            if len([x for x in potential_arg_pool if not isinstance(x, float)]) > 1:
                break
            arg_pool = potential_arg_pool

        node = None

        num = 1
        other = None
        for x in arg_pool:
            if not isinstance(x, float):
                other = x
                continue
            num *= x

        return MultOperation(num, other)


class DivOperation(Operation):
    operator = "/"

    @staticmethod
    def i_eval(lhs, rhs):
        return lhs / rhs

class ExpOperation(Operation):
    operator = "^"

    @staticmethod
    def i_eval(lhs, rhs):
        return lhs ** rhs

    def _optimized(self):
        if self.rhs == 1:
            return self.lhs

        return self

for token in output:
    if token in pemdas:
        rhs, lhs = stack.pop(), stack.pop()

        out = {
            "+": AddOperation,
            "-": SubOperation,
            "*": MultOperation,
            "/": DivOperation,
            "^": ExpOperation,
        }[token](lhs, rhs)
        stack.append(out)
    else:
        stack.append(token)

assert len(stack) == 1
root, = stack

print(root)
print("=== d/dx ===")

def dx(node):
    if isinstance(node, AddOperation):
        if isinstance(node.lhs, float): node.lhs = 0
        if isinstance(node.rhs, float): node.rhs = 0
    elif isinstance(node, SubOperation):
        if isinstance(node.lhs, float): node.lhs = 0
        if isinstance(node.rhs, float): node.rhs = 0
    elif isinstance(node, MultOperation):
        if node.rhs == "x": return node.lhs
        if node.lhs == "x": return node.rhs
    elif isinstance(node, ExpOperation):
        return MultOperation(node.rhs, ExpOperation(node.lhs, node.rhs - 1))

    if isinstance(node.lhs, Operation):
        node.lhs = dx(node.lhs)
    if isinstance(node.rhs, Operation):
        node.rhs = dx(node.rhs)

    return node

print(root.to_str())
dxr = dx(root)
print(dxr.to_str())
print(dxr.optimized().to_str())
