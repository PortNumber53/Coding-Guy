# Changelog

## Unreleased

- Hot-reload: avoid restart loops by tightening watchdog event filtering, ignoring noisy paths, and improving restart diagnostics.
- Hot-reload: default watch path is now the current directory instead of `.git`.
