# IDE & AI Tool Configuration

This project uses multiple IDE and AI coding assistants. Configuration folders are organized as follows:

## AI Coding Assistants (Tracked in Git)

These configurations are shared across the team for consistent AI behavior:

| Folder | Tool | Purpose |
|--------|------|---------|
| `.cursor/` | Cursor IDE | Plans, rules, agent configs |
| `.continue/` | Continue | MCP server configs |
| `.junie/` | Junie AI | Guidelines and context |
| `.kilocode/` | Kilocode | MCP configs, ignore patterns |

### Key Files

- `.cursor/rules.json` - Cursor agent behavior rules
- `.cursor/plans/` - Implementation plans (ALL_PHASES_INDEX.md is master)
- `.junie/guidelines.md` - Junie AI context
- `.kilocode/mcp.json` - MCP server configuration
- `AGENTS.md` - Master AI agent guidelines (root)
- `AI_GUIDELINES.md` - AI behavior rules (root)

## IDE Configs (Git Ignored)

Local IDE preferences not shared:

| Folder | Tool | Purpose |
|--------|------|---------|
| `.vscode/` | VS Code | Workspace settings, launch configs |
| `.idea/` | JetBrains | PyCharm/IntelliJ project settings |

## Code Quality Tools

| Folder/File | Tool | Purpose |
|-------------|------|---------|
| `.qodana/` | Qodana | Code quality reports (JetBrains) |
| `qodana.yaml` | Qodana | Configuration |
| `.pre-commit-config.yaml` | pre-commit | Git hooks |

## Build/Runtime (Git Ignored)

| Folder | Purpose |
|--------|---------|
| `.venv/` | Python virtual environment |
| `.pytest_cache/` | Pytest cache |
| `.trigger/` | Trigger.dev local state |

## Setup

1. **Cursor**: Open project, rules auto-load from `.cursor/rules.json`
2. **VS Code**: Install recommended extensions from `.vscode/extensions.json`
3. **JetBrains**: Open as Python project, configs in `.idea/`
4. **Continue**: MCP servers configured in `.continue/mcpServers/`
