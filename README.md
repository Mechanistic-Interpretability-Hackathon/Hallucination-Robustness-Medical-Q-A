# Gradients Analysis 

Created by: 
- Diego Sabajo: https://www.linkedin.com/in/diego-sabajo
- Eitan Sprejer: https://www.linkedin.com/in/eitan-sprejer-574380204
- Matias Zabaljauregui: https://www.linkedin.com/in/zabaljauregui
- Oliver Morris: https://www.linkedin.com/in/olimoz

---

## Overview

**Explainable Diagnostic Assistant** is an interactive tool designed to refine and analyze model outputs in a transparent and interpretable way. Built on top of **Goodfire's API** for advanced model evaluation and steering, this project focuses on improving trust and usability in AI systems, particularly in healthcare diagnostics. 

### Objectives:
- Find features that reduce hallucinations 
- Build a medical hallucinations classifier that could also provide explainable results.
- Build a framework that demonstrates the use of feature steering to reduce the hallucination rate, while maintaining accuracy.

### Technologies and Frameworks:
- **Goodfire API:** 
- **Python Libraries:** NumPy, Pandas, Matplotlib.
- **Streamlit:**


## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- `pip` (Python package installer)

### Setting Up the Environment
1. **Clone the repository**
    ```bash
   cd project-folder
   git clone https://github.com/Mechanistic-Interpretability-Hackathon/Mech-Interp.git
   cd Mech-Interp

2. **Create virtual environment**
    ```bash
    python3 -m venv .venv

3. **Activate virtual environment**
    ```bash
    Mac: python3 -m venv .venv
    Windows: .venv\Scripts\activate

4. **Install necessary packages**
    ```bash
    pip install -r requirements.txt

### Run the project
    ```bash
    streamlit run src/app.py