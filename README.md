# Pneumonia Detection Streamlit App

## Run the app

```powershell
python -m streamlit run app.py
```

If Windows opens Python 3.14 by default, use Python 3.10:

```powershell
py -3.10 -m streamlit run app.py
```

## Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Required model files

The app needs your trained Google Colab model. You can either upload it inside
the Streamlit page, or add it to this project folder with this exact name:

```text
pneumonia_model.h5
```

The app also needs:

```text
class_names.json
```

Current labels:

```json
["NORMAL", "PNEUMONIA"]
```
