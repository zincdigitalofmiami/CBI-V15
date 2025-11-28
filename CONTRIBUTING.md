# Contributing to CBI-V15

Thank you for your interest in contributing to CBI-V15!

## Development Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/zincdigitalofmiami/cbi-v15.git
   cd cbi-v15
   ```

2. **Setup Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Setup Dataform**
   ```bash
   cd dataform
   npm install -g @dataform/cli
   npm install
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add API keys to macOS Keychain
   ```

## Code Standards

### Python
- Follow PEP 8 style guide
- Use type hints
- Document all functions with docstrings
- Run `black` formatter before committing

### SQL (Dataform)
- Use consistent naming (`{asset}_{function}_{scope}_{regime}_{horizon}`)
- Add comments for complex logic
- Test with `dataform compile` before committing

### Documentation
- Update README files when adding features
- Document complex calculations
- Cite sources for formulas

## Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow existing patterns
   - Add tests if applicable
   - Update documentation

3. **Test Changes**
   ```bash
   # Python
   python -m pytest tests/
   
   # Dataform
   cd dataform && dataform compile && dataform test
   ```

4. **Commit Changes**
   ```bash
   git commit -m "feat: description of changes"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Critical Rules

1. **NO FAKE DATA** - Only real, verified data
2. **us-central1 ONLY** - All GCP resources
3. **NO COSTLY RESOURCES** - Approval required >$5/month
4. **API KEYS** - Keychain (Mac) or Secret Manager (GCP)
5. **Dataform First** - All ETL in Dataform
6. **Mac Training Only** - No cloud training

## Questions?

- Open an issue for questions or discussions
- See [docs/reference/AI_ASSISTANT_GUIDE.md](docs/reference/AI_ASSISTANT_GUIDE.md) for common tasks

---

**Last Updated**: November 28, 2025

