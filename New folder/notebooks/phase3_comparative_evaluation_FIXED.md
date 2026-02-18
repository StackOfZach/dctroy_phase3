# Phase 3 Notebook - Correct Cell Order

## ✅ CORRECT ORDER (matching phase1a):

**Cell 1:** `%cd ..` + autoreload setup
**Cell 2:** # PHASE 3 title (markdown)
**Cell 3:** ## Load config files (markdown)
**Cell 4:** Load configs (python) - **MUST BE BEFORE IP2P**
**Cell 5:** ## Initialize the logger (markdown)
**Cell 6:** Initialize logger (python)
**Cell 7:** ## Load experiment configs (markdown)
**Cell 8:** Load experiment configs (python)
**Cell 9:** ## Load the ip2p pipeline (markdown)
**Cell 10:** Load IP2P pipeline (python) - **AFTER configs**
**Cell 11:** ## Loading the dataset (markdown)
**Cell 12:** Load dataset (python)
**Cell 13:** ## Loading the batch (markdown)
**Cell 14:** Load batch (python)
**Cell 15+:** Protection methods, evaluation, etc.

## ❌ WRONG ORDER (current):

Cell 1: %cd ..
Cell 2: ## Load the ip2p pipeline (markdown) ← WRONG!
Cell 3: # PHASE 3 title (markdown) ← Should be cell 2!
Cell 4: Load IP2P pipeline (python) ← TOO EARLY!
Cell 5: Load batch (python) ← Variables don't exist yet!
Cell 6: ## Loading the batch (markdown)
Cell 7: Load dataset (python)
Cell 8: ## Loading the dataset (markdown)
Cell 9: Load experiment configs (python)
Cell 10: ## Load experiment configs (markdown)
Cell 11: Initialize logger (python)
Cell 12: ## Initialize the logger (markdown)
Cell 13: Load configs (python)
Cell 14: ## Load config files (markdown)

## 🔧 FIX:

**In Jupyter/Colab:**

1. Click and drag cells to reorder them
2. OR: Cut/paste cells in the correct order

**Manual fix:**
Run cells in this sequence:

1. Cell 1 (%cd ..)
2. Cell 13 (Load configs) ← Run this BEFORE IP2P!
3. Cell 11 (Initialize logger)
4. Cell 9 (Load experiment configs)
5. Cell 4 (Load IP2P pipeline) ← Now configs are loaded
6. Cell 7 (Load dataset)
7. Cell 5 (Load batch)
8. Continue with rest...
