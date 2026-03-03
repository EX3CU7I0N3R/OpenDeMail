# OpenDeMail

Simple mail‑processing utility.  
This document shows how to get the repo up and running.

## Prerequisites

* Python 3.9+ installed
* git (optional, if you clone from a repo)

## 1. clone the repo (if you haven’t already)

```bash
git clone https://…/OpenDeMail.git
cd OpenDeMail

2. create & activate a virtual environment
macOS / Linux
# from project root
python3 -m venv .venv
source .venv/bin/activate                # prompt will show `(.venv)`
```

## 2. Windows (PowerShell)
Note: if you have an alias such as alias python=/usr/bin/python3 in
your shell, either unalias it or invokethe venv interpreter explicitly
(.venv/bin/python …), otherwise the system Python will be used


## 3. install dependencies

`python -m pip install --upgrade pip setuptools`

`python -m pip install -r [dev-requirements.txt](http://_vscodecontentref_/0)`

make sure your requirements file does **not** list stdlib modules (e.g. remove `imaplib` – it is built in)

or

`python -m pip install python-dotenv`


## 4. run the application
From the project root directory:

`python -m OpenDeMail`
