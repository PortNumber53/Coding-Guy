# Git Hooks

This directory contains git hooks for the Coding-Guy project.

## Pre-commit Hook

The `pre-commit` hook automatically checks Python syntax on all staged Python files before allowing a commit to proceed.

### What it does

- Runs `python -m py_compile` on all staged `.py` files
- Prevents the commit if any file has syntax errors
- Shows the actual error message for debugging

### Installation

To use these hooks, configure git to use the version-controlled hooks directory:

```bash
git config core.hooksPath .githooks
```

Or set it globally for this repository:

```bash
git config --local core.hooksPath .githooks
```

### Testing the hook

To verify the hook is working:

1. Make a small change to any Python file
2. Stage the file: `git add <filename>`
3. Attempt to commit: `git commit -m "test"`

The hook will output syntax check results for all staged Python files.

### Bypassing the hook (not recommended)

In rare cases, you can bypass the pre-commit hook with:

```bash
git commit --no-verify -m "Your message"
```

Only use `--no-verify` when absolutely necessary.
