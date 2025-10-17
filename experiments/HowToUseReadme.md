##  How to Run and Analyze a General Experiment

This guide explains how to correctly set up, run, and analyze a custom experiment in your local environment.

---

### **Step 1: Environment Setup**

Before running any experiment, make sure your Python environment and dependencies are properly installed.

Open a terminal **in your project’s root directory**, then run:

```bash
pip install -r requirements.txt
```

This command installs all required libraries such as `torch`, `transformers`, and `pandas`.
Since your `setup.py` script is outdated (it mainly checks data for Task 1), using `requirements.txt` is the most reliable option.

*(Ensure your datasets are stored in the `Dataset/` folder — for example, `Dataset/Sarcasm/`.)*

---

### **Step 2: Running the Experiment**

This is the most important part — pay special attention to the **terminal path**.

#### Terminal Path Warning

All commands **must** be executed from the **project root directory**, which contains `unified_runner.py`, `common_config.py`, etc.

 **Incorrect Path Example:**

```powershell
# Wrong! Do NOT run from the 'experiments' subfolder
PS D:\Users\trhua\PycharmProjects\group8_research_topic\experiments>
```

If you run commands here, Python will raise `ModuleNotFoundError` because it can’t find `common_config.py` or `data_loader.py`.

 **Correct Path Example:**

```powershell
# Correct! Always run from the project root
PS D:\Users\trhua\PycharmProjects\group8_research_topic>
```

Running from the root ensures Python can locate all necessary modules.

---

####  **Example Command**

From the **root directory**, you can launch any experiment combination using the `unified_runner.py` script.

For example, to run the **Sarcasm Detection** task using **GPT-2** and the **Greedy** algorithm:

```bash
python unified_runner.py --task sarcasm --model gpt2 --algorithm greedy
```

* `--task sarcasm` → specifies the task to run (Sarcasm Detection)
* `--model gpt2` → selects the GPT-2 model
* `--algorithm greedy` → uses the Greedy optimization algorithm

After the experiment finishes, a `.json` result file will be saved under `experiments/results/`.

---

### **Step 3: Viewing the Experiment Report**

Once your result files appear in `experiments/results`, you can generate and analyze detailed reports.

#### **1. Universal Result Report (`universal_report.py`)**

This script automatically scans all result files and groups them by task.

* **Generate reports for all tasks:**

  ```bash
  python universal_report.py
  ```

* **Generate a report for one specific task:**

  ```bash
  python universal_report.py --task sarcasm
  ```

* **Export all results to a CSV file:**

  ```bash
  python universal_report.py --export-csv
  ```

---

#### **2. Universal Timing Analysis (`universal_timing.py`)**

This script analyzes the runtime and efficiency of your experiments.

* **Analyze all tasks:**

  ```bash
  python universal_timing.py
  ```

* **Analyze one specific task:**

  ```bash
  python universal_timing.py --task sarcasm
  ```

The report will show average runtime for each configuration and a useful **efficiency metric (Accuracy per minute)**.

