# exp = input("d/dx: ")
#exp = "6x^3-9x+4"
exp = "(1 * (2 * (3 * 4)))"
# exp = "1x^2-2x+1"

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

    if tokens[-1].isdecimal() and not (char.isdecimal() or char == "."):
        tokens.append("")

    tokens[-1] += char

# Add mult
tok = []
for t in tokens:
    if not t: continue

    if (
        t in pemdas
        or t in functions
        or t in "()"
        or not tok
        or tok[-1] in pemdas
        or tok[-1] in functions
        or tok[-1] in "()"
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

class TreeNode:
    def optimized(self):
        return self

class SimpleValue(TreeNode):
    def __init__(self, value):
        self.value = value

    def __eq__(self, val):
        # Will I regret this later....?
        return self.value == val or super().__eq__(val)

    def __repr__(self):
        return str(self.value)

class ImmediateValue(SimpleValue):
    def __repr__(self):
        return "%g" % self.value

class SymbolicValue(SimpleValue): pass

class Operation(TreeNode):
    operator = "?"

    def __init__(self, lhs, rhs):
        lhs = value_token(lhs)
        rhs = value_token(rhs)
        assert isinstance(lhs, TreeNode)
        assert isinstance(rhs, TreeNode)

        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"({type(self).__name__}: {self.lhs}, {self.rhs})"

    def eval(self):
        # UGLY
        lhs = self.lhs.eval() if isinstance(self.lhs, Operation) else self.lhs.value
        rhs = self.rhs.eval() if isinstance(self.rhs, Operation) else self.rhs.value
        return self.i_eval(lhs, rhs)
    
    def __str__(self):
        # UGLY
        # lhs = str(self.lhs.to_str() if isinstance(self.lhs, Operation) else self.lhs
        # rhs = self.rhs.to_str() if isinstance(self.rhs, Operation) else self.rhs
        return f"({self.lhs} {self.operator} {self.rhs})"

    def optimized(self):
        if isinstance(self.lhs, Operation): self.lhs = self.lhs.optimized()
        if isinstance(self.rhs, Operation): self.rhs = self.rhs.optimized()

        opt = self._universal_optimizations()
        if isinstance(opt, Operation): opt = opt._optimized()

        return opt

    def _universal_optimizations(self):
        if isinstance(self.lhs, ImmediateValue) and isinstance(self.rhs, ImmediateValue):
            return ImmediateValue(self.eval())
        return self

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
        # X * 0 == 0
        for s in [self.lhs, self.rhs]:
            if isinstance(s, ImmediateValue) and s.value == 0:
                return ImmediateValue(0)

        # Deep optimization
        immediate_val = ImmediateValue(1)
        root = self

        while True:
            # The structure we want to identify is a constant (immediate) multiplied by a multiplication
            # with another constant and so on. Basically we want to optimize this: (1 * (2 * (3 * (4 * 5)))).
            maybe_immediate, maybe_mult =  sorted([root.lhs, root.rhs], key=lambda x: isinstance(x, ImmediateValue), reverse=True)

            # If we find an immediate, multiply it no matter what. I don't remember why this works
            # but it seems to for now.
            if isinstance(maybe_immediate, ImmediateValue):
                immediate_val.value *= maybe_immediate.value

            # If either candidate is not what we want it to be, call it a day and report our findings.
            if not isinstance(maybe_immediate, ImmediateValue) or not isinstance(maybe_mult, MultOperation):
                return MultOperation(immediate_val, maybe_mult)

            # Otherwise start the next search from what we just found.
            root = maybe_mult


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

def value_token(token):
    if isinstance(token, TreeNode):
        return token
    elif isinstance(token, (float, int)):
        return ImmediateValue(token)

    return SymbolicValue(token)


for token in output:
    if token in pemdas:
        print(stack)
        rhs, lhs = value_token(stack.pop()), value_token(stack.pop())

        out = {
            "+": AddOperation,
            "-": SubOperation,
            "*": MultOperation,
            "/": DivOperation,
            "^": ExpOperation,
        }[token](lhs, rhs)
        stack.append(out)
    else:
        stack.append(value_token(token))


assert len(stack) == 1
root, = stack

print(root)
print("=== d/dx ===")

earmarked_derived_nodes = []

def dx(node) -> TreeNode:
    # Without earmarking we can derive the result of node derivations in calculations
    # which is incorrect and bad and evil
    if node in earmarked_derived_nodes:
        return node
    node = _dx(node)
    earmarked_derived_nodes.append(node)
    return node

def _dx(node) -> TreeNode:
    if isinstance(node, ImmediateValue):
        # A derivitive of a constant is zero
        return ImmediateValue(0)

    if isinstance(node, SymbolicValue):
        # TODO: Handle symbols we're not taking with respect to....

        # The derivitive of any variable with respect to itself is 1.
        return ImmediateValue(1)

    assert isinstance(node, Operation)

    if isinstance(node, AddOperation):
        node.lhs = dx(node.lhs)
        node.rhs = dx(node.rhs)
    # This causes issues. Why?
    elif isinstance(node, SubOperation):
        node.lhs = dx(node.lhs)
        node.rhs = dx(node.rhs)
    elif isinstance(node, MultOperation):
        if node.rhs == "x": return node.lhs
        if node.lhs == "x": return node.rhs
    elif isinstance(node, DivOperation):
        # d/dx(f / g)   -->    ( (g * f') - (f * g') ) / (g ^ 2)
        # girlfriend minus foreground over square ground
        return DivOperation(
            SubOperation(
                MultOperation(node.rhs, dx(node.lhs)),
                MultOperation(node.lhs, dx(node.rhs)),
            ),
            ExpOperation(node.rhs, 2)
        )
    elif isinstance(node, ExpOperation):
        return MultOperation(node.rhs, ExpOperation(node.lhs, SubOperation(node.rhs, 1)))

    if isinstance(node.lhs, Operation):
        node.lhs = dx(node.lhs)
    if isinstance(node.rhs, Operation):
        node.rhs = dx(node.rhs)

    return node

print("\ty =", str(root))
dxr = dx(root)
print("\td/dx =", str(dxr))
print("[opt]\td/dx =", str(dxr.optimized()))
