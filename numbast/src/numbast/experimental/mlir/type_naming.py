"""Helpers for constructing readable, collision-resistant Numba type names."""


def qualified_type_class_name(type_obj: object) -> str:
    cls = type_obj if isinstance(type_obj, type) else type(type_obj)
    return f"{cls.__module__}.{cls.__qualname__}@{id(cls):x}"


def make_unique_type_name(type_obj: object, base_name: str) -> str:
    """Build a readable type name with class-qualified disambiguation."""
    return f"{base_name}::{qualified_type_class_name(type_obj)}"
