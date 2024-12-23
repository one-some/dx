# exp = "(1 * (2 * (3 * 4)))"
# exp = "1x^2-2x+1"
# exp = "2x^4 - 10x^2+13x"
# exp = "4x^7-3x^-7+9x"
# exp = "sqrt(x)"
# exp = "y^-4 - 9y^-3 +8y-2 + 12"

exp = "(x + 1) * x"
# exp = "6x^3-9x+4"
exp = "3x^2+6x-4"

try:
    import os
    calc_mode = False
except ImportError:
    print("Calculator Mode")
    calc_mode = True
    exp = None


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
functions = ["sqrt"]

def isdecimal(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def tokenize(exp):
    tokens = [""]

    for char in exp.replace(" ", ""):
        if char in ["(", ")", ","] or char in pemdas:
            tokens.append(char)
            tokens.append("")
            continue

        if isdecimal(tokens[-1]) and not (char.isdigit() or char == "."):
            tokens.append("")

        tokens[-1] += char

    # Cull empty tokens
    tokens = [x for x in tokens if x]

    # Process negatives
    for i in range(len(tokens)):
        if i + 2 >= len(tokens): continue
        if tokens[i] not in pemdas: continue
        if tokens[i + 1] != "-": continue
        if not isdecimal(tokens[i + 2]): continue

        tokens[i + 2] = "-" + tokens[i + 2]
        del tokens[i + 1]

    # Add implicit multiplication like coeffecients or whatever
    tok = []
    for t in tokens:
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
    print(tok)
    return tok


def dijkstra(tokens):
    output = []
    stack = []

    # TODO: We could modify this to not use RPN as an intermediary step and instead assemmble the AST here

    for token in tokens:
        if token in functions:
            stack.insert(0, token)
        elif token == "(":
            stack.insert(0, token)
        elif token == ")":
            while stack and stack[0] != "(":
                output.append(stack.pop(0))

            stack.pop(0)
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

    output = [float(x) if isdecimal(x.lstrip("-")) else x for x in output]

    return output

# Eval

class TreeNode:
    def __init__(self):
        self.dx_earmarker = None

    def dx_earmark(self, respect_to):
        self.dx_earmarker = respect_to
        return self

    def optimized(self):
        return self

class SimpleValue(TreeNode):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __eq__(self, val):
        # Will I regret this later....?
        return self.value == val or self is val

    def __repr__(self):
        return str(self.value)

class ImmediateValue(SimpleValue):
    def __repr__(self):
        return "%g" % self.value

class SymbolicValue(SimpleValue): pass

class Operation(TreeNode):
    operator = "?"

    def __init__(self, lhs, rhs):
        super().__init__()

        lhs = value_token(lhs)
        rhs = value_token(rhs)
        assert isinstance(lhs, TreeNode)
        assert isinstance(rhs, TreeNode)

        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return "(%s: %s, %s)" % (type(self).__name__, self.lhs, self.rhs)

    def eval(self):
        # UGLY
        lhs = self.lhs.eval() if isinstance(self.lhs, Operation) else self.lhs.value
        rhs = self.rhs.eval() if isinstance(self.rhs, Operation) else self.rhs.value
        return self.i_eval(lhs, rhs)
    
    def __str__(self):
        return "(%s)" % " ".join([str(x) for x in [self.lhs, self.operator, self.rhs]])

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

    def __str__(self):
        maybe_immediate, maybe_symbolic =  sorted([self.lhs, self.rhs], key=lambda x: isinstance(x, ImmediateValue), reverse=True)

        if not isinstance(maybe_immediate, ImmediateValue): return super().__str__()
        if not isinstance(maybe_symbolic, SymbolicValue): return super().__str__()

        return str(maybe_immediate) + str(maybe_symbolic)

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

def build_tree(rpn):
    stack = []

    for token in rpn:
        if token in functions:
            # UNARY OPERATORS
            if token == "sqrt":
                # I AM LAZY
                term = value_token(stack.pop())
                out = ExpOperation(term, 1/2)
            else:
                assert False
        elif token in pemdas or token in functions:
            # BINARY OPERATORS
            rhs, lhs = value_token(stack.pop()), value_token(stack.pop())
            out = {
                "+": AddOperation,
                "-": SubOperation,
                "*": MultOperation,
                "/": DivOperation,
                "^": ExpOperation,
            }[token](lhs, rhs)
        else:
            out = value_token(token)

        stack.append(out)

    assert len(stack) == 1
    root, = stack

    print(root)
    return root

def dx(node, respect_to="x") -> TreeNode:
    # Without earmarking we can derive the result of node derivations in calculations
    # which is incorrect and bad and evil
    if node.dx_earmarker == respect_to:
        return node
    node = _dx(node, respect_to).dx_earmark(respect_to)
    # TODO: Earmarked with respect to what??
    return node

def _dx(node, respect_to="x") -> TreeNode:
    if isinstance(node, ImmediateValue):
        # A derivitive of a constant is zero
        return ImmediateValue(0)

    if isinstance(node, SymbolicValue):
        if node.value == respect_to:
            # The derivitive of any variable with respect to itself is 1.
            return ImmediateValue(1)

        # TODO: Handle symbols we're not taking with respect to....
        assert node.value == respect_to

    assert isinstance(node, Operation)

    if isinstance(node, AddOperation):
        node.lhs = dx(node.lhs, respect_to=respect_to)
        node.rhs = dx(node.rhs, respect_to=respect_to)
    elif isinstance(node, SubOperation):
        node.lhs = dx(node.lhs, respect_to=respect_to)
        # print("lhs", type(node.lhs), node.lhs, "rhs", type(node.rhs), node.rhs)
        node.rhs = dx(node.rhs, respect_to=respect_to)
    elif isinstance(node, MultOperation):
        # d/dx(f * g)    -->    (g * f') + (f * g')
        return AddOperation(
            MultOperation(node.rhs, dx(node.lhs, respect_to=respect_to)),
            MultOperation(node.lhs, dx(node.rhs, respect_to=respect_to)),
        )
    elif isinstance(node, DivOperation):
        # d/dx(f / g)   -->    ( (g * f') - (f * g') ) / (g ^ 2)
        # girlfriend minus foreground over square ground
        return DivOperation(
            SubOperation(
                MultOperation(node.rhs, dx(node.lhs, respect_to=respect_to)),
                MultOperation(node.lhs, dx(node.rhs, respect_to=respect_to)),
            ),
            ExpOperation(node.rhs, 2)
        )
    elif isinstance(node, ExpOperation):
        return MultOperation(node.rhs, ExpOperation(node.lhs, SubOperation(node.rhs, 1)))

    if isinstance(node.lhs, Operation):
        node.lhs = dx(node.lhs, respect_to=respect_to)
    if isinstance(node.rhs, Operation):
        node.rhs = dx(node.rhs, respect_to=respect_to)

    return node

def derive_string(string):
    # Calc input
    string = string.lower()
    string = string.replace("**", "^")

    tokens = tokenize(string)
    rpn = dijkstra(tokens)
    root_node =  build_tree(rpn)

    print("=== d/dx ===")
    print("\ty =", str(root_node))
    dxr = dx(root_node)
    print(".....\td/dx =", str(dxr))
    print("[opt]\td/dx =", str(dxr.optimized()))

if calc_mode:
    while True:
        exp = input("d/dx: ")
        if not exp: break
        derive_string(exp)
else:
    derive_string(exp)
