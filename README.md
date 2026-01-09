# Simulation for EPFL course CIVIL-324: Urban Public Transport Systems

This repository contains a simulation environment and backend API for experimenting with urban public transport systems, designed for the EPFL course **CIVIL-324**.  

---

## 1. Prerequisites

- **Python** 3.12 (recommended via Conda)
- **Poetry** for dependency management
- (Optional but recommended) **conda** for virtual environments

---

## 2. Setup & Installation

### 2.1 Create and activate a Conda environment

```bash
conda create -n civ324-sim python=3.12
conda activate civ324-sim
```

### 2.2 Install Poetry

If you don’t already have Poetry, refer to this [page](https://python-poetry.org/docs/) for installation.

### 2.3 Install project dependencies

From the repository root:

```bash
poetry install
```

This will:

* Create a virtual environment managed by Poetry
* Install all dependencies defined in `pyproject.toml`
* Generate / update the `poetry.lock` file

This will NOT:

* Install the dependencies necessary to run the test server, but only the ones that are necessary to run the `minitransit_simulation` package.

To also install the dependencies necessary to run the test server, run :
```bash
poetry install --extras "test_interface"
```

### 2.4 Adding new dependencies

To add a new package:

```bash
poetry add <package-name>
```

This updates both `pyproject.toml` and `poetry.lock`.

---

## 3. Running the Backend Server

**!!! Make sure to have installed the packages in the `test_interface` extra as such :**
```bash
poetry install --extras "test_interface"
```

The FastAPI app entrypoint is `main:app`.

From the project root:

```bash
poetry run uvicorn main:app --reload
```

* The server will start at `http://localhost:8000` by default. You can open this URL in browser for a visualization page.
* `--reload` enables auto-reload on code changes (useful for development).

---

## 4. API Documentation (Swagger UI)

Once the backend server is running, you can access:

* **Swagger UI**:
  `http://localhost:8000/docs`
* **ReDoc** (alternative docs):
  `http://localhost:8000/redoc`

These interfaces let you explore and test the simulation API interactively.

---

## 5. Code Style & Formatting

Before committing, please format the code:

```bash
black .
isort .
```

This ensures consistent style across the project.

---

## 6. Running Tests

To run the test suite:

```bash
pytest
```

---

## 7. Install the `minitransit_simulation` as a package in your project

With poetry :
```bash
poetry add git+https://github.com/{REPO_NAME}.git
```

With uv :
```bash
uv pip install git+https://github.com/{REPO_NAME}.git
```
