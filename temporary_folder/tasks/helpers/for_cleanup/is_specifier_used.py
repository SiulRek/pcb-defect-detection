import ast


def is_specifier_used(name, source, include_imports=False):
    tree = ast.parse(source)

    class Visitor(ast.NodeVisitor):
        def __init__(self, include_imports=include_imports):
            self.result = False
            self.include_imports = include_imports

        def visit_Name(self, node):
            if node.id == name:
                self.result = True

        def visit_Attribute(self, node):
            if node.attr == name:
                self.result = True
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if node.name == name:
                self.result = True
            self.generic_visit(node)

        def visit_Import(self, node):
            if self.include_imports:
                for alias in node.names:
                    if alias.asname == name or alias.name == name:
                        self.result = True

        def visit_ImportFrom(self, node):
            if self.include_imports:
                for alias in node.names:
                    if alias.asname == name or alias.name == name:
                        self.result = True

    visitor = Visitor()
    visitor.visit(tree)
    return visitor.result
