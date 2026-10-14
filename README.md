# NLP and Deep Learning Assignment Report

## Project Files
*   task1_numpy_nn.py : Activation functions and XOR network code.
*   task2_pytorch_pipeline.py : PyTorch model training and saving logic.
*   task3_classical_nlp.py : TF-IDF model vs GloVe word math.
*   task4_transformers_fastapi.py : FastAPI live server code.

---

## Completed Tasks and Conditions

### Task 1: Fundamentals
*   Created ReLU, Sigmoid, and Softmax functions using NumPy.
*   Verified manual math gradients against PyTorch Autograd (w.grad = -32.0).
*   Trained a basic 2-4-1 network from scratch to solve XOR data logic.
*   Plotted loss curves for learning rates: 0.5, 0.01, and 5.0.

### Task 2: PyTorch Pipeline
*   Built a Custom Dataset and DataLoader for the breast cancer dataset.
*   Created an MLP model with BatchNorm and Dropout layers to stop overfitting.
*   Trained for 40 epochs, saved weights, and verified identical reloaded accuracy.

### Task 3: Classical NLP
*   Created a dataset of 300 text samples with basic text cleaning.
*   Trained a TF-IDF + Logistic Regression model hitting exactly 0.84 accuracy.
*   Solved the classic word equation space: king - man + woman = queen.

### Task 4: FastAPI Server
*   Extracted data tokens using the pre-trained DistilBERT tokenizer.
*   Built a live production web prediction API server using FastAPI.
*   Added input checking to return a 422 Error code if the text field is empty.

---

##  How to Run the Code

1. Open your terminal and run the main calculation scripts:
python task1_numpy_nn.py
python task2_pytorch_pipeline.py
python task3_classical_nlp.py

2. Start the local API web server instance:
python task4_transformers_fastapi.py

3. Open your Google Chrome browser and test the live server at:
http://127.0.0