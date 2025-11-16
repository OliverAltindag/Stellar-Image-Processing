# Stellar-Image-Processing

Taking raw CCD stellar data, and performing pre-processing: we remove biases, perform image alignment and modularize the process for reusage. The repo will also include generic graphing fucntions, and magnitude calculation. The reduction portion will need to be adjusted depending on the specific scientific object of interest and your file architecture.

### Clone the repo and run for your data: 
```
git clone https://github.com/OliverAltindag/Stellar-Image-Processing.git
```
```
cd Stellar-Image-Processing
```
### Virtual Environment
For Mac/Linux:
```
py -m venv venv
```
```
source venv/bin/activate
```

For Windows:
```
py -m venv venv
```
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
```
.\venv\Scripts\activate
```

### Install the dependecies:
```
pip install -r requirements.txt
```

### Run the reduction:
```
python run.py
```

The steps follow similarly for graphing and other features. 
