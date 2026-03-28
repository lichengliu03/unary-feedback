# Webshop Minimal

A minimalistic webshop environment for faster agentic system development. Adapted from full WebShop https://github.com/princeton-nlp/WebShop/tree/master.

Authors for this adaptation: Xing Jin, Zihan Wang

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/webshop-minimal.git
    ```
2. Navigate to the project directory:
    ```bash
    cd webshop-minimal
    ```
3. Install dependencies:
    ```bash
    bash setup.py
    ```
4. Install the package using pip:
    ```bash
    pip install --no-build-isolation --no-deps .
    ```

## Usage

## Usage

### 1. Import in a Python Session

To use the package, start by importing it in your Python session:

```python
import webshop_minimal
```

### 2. Setup Base Directory

Set up the base directory for your project to ensure proper file handling:

```python
import os

base_dir = "/path/to/your/basedir"
webshop_minimal.init_basedir(base_dir)
```

Replace `/path/to/your/basedir` with the actual path to your project directory. If you don't call init_basedir, the env will default to initialize with the provided data under this repo.


## Base Directory Structure

The base directory (`basedir`) should be organized as follows:

```
basedir/
├── data/
│   ├── items_ins_v2_1000.json
│   ├── items_shuffle_1000.json
│   ├── items_human_ins.json
├── templates/
│   ├── template1.html
│   ├── template2.html
├── search_engine/
│   ├── index1/
│   ├── index2/
```

### Explanation of Subdirectories

1. **`data/`**  
    Contains JSON files that define various attributes and configurations:
    - `items_ins_v2_1000.json`: Stores attribute-related data.
    - `items_shuffle_1000.json`: Contains file-related metadata.
    - `items_human_ins.json`: Includes human-readable attributes.

2. **`templates/`**  
    Houses HTML templates used for rendering web pages or other UI components.

3. **`search_engine/`**  
    Contains directories for different search indices, enabling efficient data retrieval.

Ensure that these subdirectories and files are correctly set up for the application to function as expected. See example data generation in setup.sh
