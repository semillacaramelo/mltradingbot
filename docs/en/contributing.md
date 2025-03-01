# Contributing Guidelines

## Code Standards

### Python Style
- Follow PEP 8
- Use type hints
- Maximum line length: 100
- Use docstrings for all functions

### Documentation
- Update relevant docs
- Include code examples
- Document API changes
- Keep README current

## Development Process

1. **Fork & Clone**
2. **Setup Environment**
```python
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
3. **Create Branch**

```bash
git checkout -b feature/your-feature
```
4. **Testing**
* Add unit tests
* Run existing tests
* Test in demo mode
5. **Submit PR**
* Clear description
* Reference issues
####Pull Request Guidelines
* One feature per PR
* Keep changes focused
* Include tests
* Update documentation