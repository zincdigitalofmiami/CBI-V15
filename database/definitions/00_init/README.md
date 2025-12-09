# 00_init - Schema Initialization

## Purpose
Initial database setup - create schemas, install extensions.

## Execution Order
This runs FIRST before all other definitions.

## What Belongs Here
- `CREATE SCHEMA` statements
- `INSTALL` / `LOAD` extension commands
- Database-level settings

## What Does NOT Belong Here
- Table definitions (→ `01_raw/` through `04_training/`)
- Views (→ `06_api/`)

