# Contributing to Mochi-Moo

<div align="center">

<picture>
  <img width="100%" alt="Header"
       src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,4,8,12,16,20&height=260&section=header&text=Contributing%20to%20Mochi%E2%80%91Moo&fontSize=48&animation=fadeIn&fontAlignY=40&desc=In%20a%20world%20of%20harsh%20primaries%2C%20be%20the%20gentle%20gradient.&descAlignY=70&descSize=16&fontColor=FFF8FD" />
</picture>

<picture>
  <img alt="Typing subtitle"
       src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&duration=3200&pause=900&color=E6C6FF&center=true&vCenter=true&multiline=true&width=920&height=60&lines=Technical%20excellence%20meets%20pastel%20aesthetics;Kindness%20in%20tone%2C%20rigor%20in%20tests%2C%20clarity%20in%20design" />
</picture>

<p>
  <img src="https://img.shields.io/badge/Style-Black%20%7C%20isort%20%7C%20Ruff-FFE0F5?style=for-the-badge&labelColor=E6E0FF" alt="Style">
  <img src="https://img.shields.io/badge/Types-Mypy-DCF9F0?style=for-the-badge&labelColor=E6E6FA" alt="Mypy">
  <img src="https://img.shields.io/badge/Tests-Pytest-FDE1C9?style=for-the-badge&labelColor=E6E0FF" alt="Pytest">
  <img src="https://img.shields.io/badge/CI-GitHub%20Actions-99D1FF?style=for-the-badge&labelColor=FFE0F5" alt="CI">
</p>

<picture>
  <img alt="Soft Spacer" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=1,6,11,16&height=70&text=&fontSize=16" />
</picture>

</div>

## Welcome

Thank you for considering a contribution. Mochi-Moo lives where **technical rigor** meets **gentle design**. This guide ensures your ideas land softly and ship confidently.

<picture>
  <img alt="Divider" width="100%"
       src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=20,16,12,8,4,0&height=3" />
</picture>

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=0,5,10,15,20&height=110&text=Philosophy&fontSize=30&fontColor=4A4A4A" />
</picture>

- **Pastel first.** Visuals should be calm, legible, and consistent with the project palette.  
- **Clarity over cleverness.** Code reads like a careful explanation.  
- **Kindness in collaboration.** Reviews are constructive, decisions documented.  
- **Tests as confidence.** If it matters, it's tested. If it's public, it's typed.  

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=2,8,14,20&height=110&text=Getting%20Started&fontSize=30&fontColor=4A4A4A" />
</picture>

```bash
# 1) Fork and clone
git clone https://github.com/<you>/Mochi-Moo.git
cd Mochi-Moo

# 2) Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"  # if provided

# 4) Run tests to verify your setup
pytest -q
```

**Optional (pre-commit hooks):**

```bash
pip install pre-commit
pre-commit install
```

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=1,7,13,19&height=110&text=Branching%20Model&fontSize=30&fontColor=4A4A4A" />
</picture>

| Action       | Branch                  |
| ------------ | ----------------------- |
| Ongoing work | `feature/<short-topic>` |
| Bug fix      | `fix/<short-issue>`     |
| Release prep | `release/x.y.z`         |
| Mainline     | `main` (protected)      |
| Integration  | `develop` (if used)     |

> Keep PRs focused and small; reference related issues in the description.

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=3,9,15&height=110&text=Commit%20Convention&fontSize=30&fontColor=4A4A4A" />
</picture>

**Format**

```
type(scope): brief description

Longer explanation if useful.
Refs #123
```

| Type     | Purpose            | Examples                                  |
| -------- | ------------------ | ----------------------------------------- |
| feat     | New capability     | `feat(core): add whisper mode thresholds` |
| fix      | Bug fix            | `fix(privacy): redact SSN patterns`       |
| docs     | Documentation      | `docs(readme): add pastel banners`        |
| style    | Formatting only    | `style: black/isort pass`                 |
| refactor | No behavior change | `refactor(viz): simplify palette map`     |
| test     | Add/adjust tests   | `test(core): coverage for trace save`     |
| chore    | CI, deps, build    | `chore(ci): add codecov upload`           |

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=4,10,16,20&height=110&text=Style%20%26%20Quality%20Gates&fontSize=30&fontColor=4A4A4A" />
</picture>

```bash
# Format and sort
black mochi_moo/ tests/
isort mochi_moo/ tests/

# Lint and type-check
ruff check mochi_moo/ tests/
mypy mochi_moo/

# Run tests with coverage
pytest -v --cov=mochi_moo --cov-report=term-missing
```

| Gate        | Target                                    |
| ----------- | ----------------------------------------- |
| Black/isort | No diffs                                  |
| Ruff        | No errors (warnings allowed if justified) |
| MyPy        | No errors on public APIs                  |
| Coverage    | ≥ 90% on touched files                    |

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=0,6,12,18&height=110&text=Testing%20Guidelines&fontSize=30&fontColor=4A4A4A" />
</picture>

* **Unit tests** for pure logic; **integration tests** for IO and pipelines.
* Test **edge cases**, not just the happy path.
* Include **property-based tests** where invariants matter.
* For visuals, prefer **deterministic render checks** (hashes, SVG text) over screenshots.

**Quick examples**

```bash
pytest tests/test_core.py::TestMochiCore::test_basic_processing -q
pytest -k "palette or privacy" -q
```

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=5,11,17&height=110&text=Docs%20%26%20Examples&fontSize=30&fontColor=4A4A4A" />
</picture>

* Update `README.md` when user behavior changes.
* New modules need a **docstring overview** and **example snippet**.
* Keep HTML blocks **GitHub-safe**: encode query `&` in URLs.
* Prefer **tables** for option matrices and **Mermaid** for flows.

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=2,10,18&height=110&text=Pastel%20Aesthetic%20Guide&fontSize=30&fontColor=4A4A4A" />
</picture>

| Element  | Guidance                                                    |
| -------- | ----------------------------------------------------------- |
| Palette  | Use soft ombré gradients (rose→lavender→mint→sky).          |
| Contrast | Text #4A4A4A on light pastels; avoid low-contrast combos.   |
| Motion   | Subtle, reversible animations; nothing that blocks reading. |
| Badges   | `style=for-the-badge` with pastel `labelColor`.             |
| Dividers | Use capsule-render `rect` 2–4px height with gradient.       |

**Examples**

```html
<picture>
  <img width="100%" alt="Divider"
       src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=20,16,12,8,4,0&height=3" />
</picture>
```

```html
<picture>
  <img width="100%" alt="Soft Banner"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=1,6,11,16&height=100&text=Section&fontSize=28&fontColor=4A4A4A" />
</picture>
```

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=6,12,18&height=110&text=Pull%20Request%20Checklist&fontSize=30&fontColor=4A4A4A" />
</picture>

* [ ] Small, focused PR (ideally < 300 lines net)
* [ ] Tests added/updated and passing locally
* [ ] Lint, type-check, and format pass
* [ ] Docs updated (README, examples, or docstrings)
* [ ] No secrets or PII leaked in code or logs
* [ ] Screenshots or short demo for UX-visible changes
* [ ] Linked issues referenced in description

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=7,13,19&height=110&text=Issue%20Labels&fontSize=30&fontColor=4A4A4A" />
</picture>

| Label              | Use                                |
| ------------------ | ---------------------------------- |
| `good first issue` | Guided, low-risk tasks             |
| `help wanted`      | Community assistance welcome       |
| `bug`              | Repro steps, expected vs. actual   |
| `enhancement`      | Feature requests and improvements  |
| `docs`             | Documentation issues               |
| `design`           | Aesthetic or UX concerns           |
| `security`         | Vulnerabilities or hardening tasks |

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=8,12,16,20&height=110&text=Security%20%26%20Disclosure&fontSize=30&fontColor=4A4A4A" />
</picture>

If you discover a vulnerability, **do not** open a public issue.
Email **security contact** at: `becaziam@gmail.com` with steps to reproduce.
We will coordinate a fix and disclose responsibly.

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=0,5,10,15,20&height=110&text=Community%20Conduct&fontSize=30&fontColor=4A4A4A" />
</picture>

We practice patience, curiosity, and clarity. Debate the idea, care for the person.
Harassment, personal attacks, or exclusionary behavior are not tolerated.

<picture>
  <img alt="Banner" width="100%"
       src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=1,6,11,16&height=110&text=License%20Note&fontSize=30&fontColor=4A4A4A" />
</picture>

Mochi-Moo is released under **MIT** with a **Pastel Clause**: derivatives must maintain aesthetic integrity (gentle gradients, calm motion, accessible contrast).

<picture>
  <img alt="Footer"
       width="100%"
       src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,6,12,18&height=140&section=footer&text=Thank%20you%20for%20helping%20Mochi%E2%80%91Moo%20dream%20better.&fontSize=20&fontColor=FFF8FD" />
</picture>
