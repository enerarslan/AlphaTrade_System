# Contributing to AlphaTrade

## Code Style Guidelines

### Docstrings - Google Style

All docstrings in this project MUST follow Google Python Style Guide format.

#### Module Docstrings

```python
"""
Brief description of the module.

Detailed description if needed, explaining the purpose and main functionality.

Example:
    >>> from module import function
    >>> function()

Attributes:
    MODULE_CONSTANT: Description of constant.

Note:
    Any important notes about the module.
"""
```

#### Function/Method Docstrings

```python
def function_name(param1: str, param2: int, param3: list[str] | None = None) -> dict[str, Any]:
    """Brief description of function (one line, imperative mood).

    Extended description if needed. Can span multiple paragraphs.
    Explain the algorithm, important details, or edge cases.

    Args:
        param1: Description of param1. No type info (already in signature).
        param2: Description of param2.
        param3: Description of param3. Defaults to None.

    Returns:
        Description of return value. For complex types, describe structure:
        {
            "key1": Description of key1,
            "key2": Description of key2,
        }

    Raises:
        ValueError: When param1 is empty.
        TypeError: When param2 is negative.

    Example:
        >>> result = function_name("hello", 42)
        >>> print(result)
        {"status": "ok"}

    Note:
        Any important caveats or usage notes.

    See Also:
        related_function: For related functionality.
    """
```

#### Class Docstrings

```python
class ClassName:
    """Brief description of the class.

    Extended description explaining purpose, behavior, and usage patterns.

    Attributes:
        attr1: Description of instance attribute.
        attr2: Description of instance attribute.

    Example:
        >>> obj = ClassName(param1="value")
        >>> obj.method()

    Note:
        Important implementation notes or caveats.
    """

    def __init__(self, param1: str) -> None:
        """Initialize the class.

        Args:
            param1: Description of constructor parameter.
        """
```

### Type Hints

- ALL public functions and methods MUST have type hints
- Use modern Python 3.11+ syntax (`list[str]` not `List[str]`)
- Use `|` for unions (`str | None` not `Optional[str]`)
- Use `Any` sparingly; prefer specific types

### Naming Conventions

- `snake_case` for functions, methods, variables
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants
- `_private_method` for internal methods
- `__dunder__` only for magic methods

### Code Organization

1. Imports (stdlib, third-party, local)
2. Constants
3. Module-level functions
4. Classes
5. Main entry point (if applicable)

## Testing

### Test Naming

```python
def test_function_name_when_condition_should_behavior():
    """Test that function_name returns expected result when condition."""
```

### Test Organization

- Unit tests: `tests/unit/test_<module>.py`
- Integration tests: `tests/integration/test_<feature>.py`
- Fixtures: `tests/conftest.py`

## Pull Request Process

1. Create feature branch from `develop`
2. Write tests first (TDD preferred)
3. Implement feature
4. Run linting: `ruff check quant_trading_system/`
5. Run type checking: `mypy quant_trading_system/ --strict`
6. Run tests: `pytest tests/`
7. Update documentation if needed
8. Submit PR with clear description

## Commit Messages

Follow conventional commits:

```
type(scope): brief description

Longer description if needed.

Refs: #issue-number
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
