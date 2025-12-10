# Stellar-Image-Processing

Taking raw CCD stellar data, and performing pre-processing: we remove biases, perform image alignment and modularize the process for reusage. The repo will also include generic graphing fucntions, and magnitude calculation. The reduction portion will need to be adjusted depending on the specific path on your computer. The code is made to work by reducing .fit fils, but can be easily change for .fits by changing the wildcards. Additionally, your science images must have an operational header, with emposure time ("EXPTIME") and fiter ("FILTER"). Your file architecture must reflect this for the code to work:

```
datafolder/
├── calibration/
│   ├── biasframes/
│   ├── darks/
│   └── flats/
│       ├── blue/
│       ├── visual/
│       └── red/
├── target/
│   ├── blue/
│   ├── visual/
│   └── red/
└── standard/
    ├── blue/
    ├── visual/
    └── red/
```

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

For graphin and photometric analysis, I recommend donwloading the Jupyter notebook and editing the needed steps. 
Given the emphasis of data visualization the analysis was not done in a .py script.
