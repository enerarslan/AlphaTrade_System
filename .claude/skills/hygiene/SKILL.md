# Code Hygiene Skill

Invoke the Code Hygiene Agent to scan, clean, and maintain code quality.

## Usage

```
/hygiene                    # Quick scan for issues
/hygiene scan              # Full codebase scan
/hygiene clean             # Auto-fix safe issues
/hygiene report            # Generate detailed report
```

## What It Does

1. **Dead Code Detection** - Finds unused functions, classes, imports
2. **Duplicate Elimination** - Identifies copy-pasted code
3. **Deprecation Cleanup** - Removes old deprecated code
4. **Import Hygiene** - Cleans and sorts imports
5. **Consistency Check** - Enforces naming conventions

## When to Use

- After major feature completion
- Before PR review
- Weekly maintenance
- When codebase feels "messy"

## Prompt

You are the Code Hygiene Agent. Your task is to systematically scan the AlphaTrade codebase and identify:

1. **Dead Code**: Functions, classes, methods that are never called
2. **Duplicates**: Similar code blocks that should be consolidated
3. **Deprecated**: Code marked for removal
4. **Import Issues**: Unused or unorganized imports
5. **Consistency**: Naming convention violations

Start by scanning the quant_trading_system directory. For each module:
- List all exports
- Find all usages
- Identify orphan code
- Report duplicates

Provide a structured report with actionable recommendations. Prioritize by impact:
- CRITICAL: Dead code hiding bugs
- HIGH: Duplicate maintenance burden
- MEDIUM: Import cleanliness
- LOW: Style consistency

After analysis, offer to auto-fix safe issues (unused imports, sorting).
