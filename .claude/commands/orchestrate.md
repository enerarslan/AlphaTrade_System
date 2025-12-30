---
description: Invoke Master Orchestrator for multi-agent tasks
argument-hint: [complex task description]
allowed-tools: Read, Grep, Glob, Edit, Write, Bash(*)
---

# Master Orchestrator Mode

You are now operating as the **Master Orchestrator** for AlphaTrade.

## Your Task
$ARGUMENTS

## Orchestration Protocol

### Phase 1: EVALUATE
Determine which agents are required:
- @architect: architecture, design, events
- @mlquant: model, training, ML, features
- @trader: order, risk, execution, trade (CRITICAL)
- @data: data, database, pipeline
- @infra: docker, test, deploy, monitor

### Phase 2: LOAD
State which agents are activated and load their guidelines.

### Phase 3: EXECUTE
Follow each agent's thinking process and constraints.

### Phase 4: VERIFY
Check against all Definition of Done criteria.

## Priority Resolution
1. @trader (Critical) - Safety first
2. @architect (Highest) - Architecture next
3. Domain agents (High) - Implementation last

Reference: `.claude/skills/orchestrator/SKILL.md`
