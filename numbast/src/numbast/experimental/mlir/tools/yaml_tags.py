def string_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return "".join(seq)
