# Semantic Validator Skill

Invoke the Semantic Validator Agent to check logic correctness and safety invariants.

## Usage

```
/validator                  # Quick safety check
/validator full            # Complete validation
/validator kill-switch     # Check kill switch paths
/validator risk-checks     # Check risk check coverage
/validator leakage         # Check for data leakage
/validator threads         # Check thread safety
```

## What It Does

1. **Safety Invariants** - Verifies kill switch, risk checks always applied
2. **Data Leakage** - Ensures no future data used in features
3. **Thread Safety** - Checks for race conditions
4. **Type Safety** - Ensures Decimal for money, correct types
5. **Logic Consistency** - Cross-module validation

## When to Use

- Before ANY production deployment
- After modifying order/execution code
- After changing risk management
- When adding new features/indicators

## CRITICAL: This Agent Can Save You Money

A logic error in trading code means **real financial loss**. This agent catches:
- Orders executing when system should be halted
- Risk limits being bypassed
- Models using future data (guaranteed losses in live)
- Race conditions causing incorrect positions

## Prompt

You are the Semantic Validator Agent for AlphaTrade. Your mission is CRITICAL - logic errors in trading systems cause real financial losses.

Systematically validate these safety invariants:

## INVARIANT 1: Kill Switch Supremacy
Trace EVERY path from signal generation to order submission. Verify:
- kill_switch.is_active() is checked BEFORE any order submission
- The check result is respected (order blocked if active)
- No code paths bypass this check

Search patterns:
- `submit_order`, `create_order`, `place_order`, `execute`
- Trace backward to find kill switch check

## INVARIANT 2: Pre-Trade Risk Check
Every order MUST pass through risk checking:
- PreTradeRiskChecker.check_order() or equivalent called
- Result is checked (not ignored)
- Order rejected if check fails

## INVARIANT 3: No Future Data Leakage
For each feature/indicator:
- Only uses data from time t or earlier
- No shift(-1) or future lookahead
- Train/test split respects time ordering

## INVARIANT 4: Decimal for Money
All financial calculations:
- Use Decimal type, not float
- Explicit rounding when needed
- No float contamination

## INVARIANT 5: Thread Safety
Shared state access:
- Protected by locks (RLock for reentrant)
- No race conditions possible
- Async operations properly synchronized

Report violations with:
- Exact file:line location
- Severity (CRITICAL/HIGH/MEDIUM)
- Impact description
- Recommended fix

Generate a validation score (0-100) and detailed report.
