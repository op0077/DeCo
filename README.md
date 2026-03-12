# DeCo (Deblurring + Colorization)

DeCo is a web application that enhances images by automatically **removing blur and restoring color**.

Users can upload an image, and the system processes it to:
- **Deblur** the image and recover sharper details
- **Colorize** grayscale or faded images

The processed image can then be downloaded in **PNG or JPG format**.

---

# Setup Instructions

### To install the required dependencies, run:
<br><br/>
**For Bash (Git Bash / MINGW64)**

python -m venv venv

source venv/Scripts/activate

pip install -r requirements.txt

<br><br/>
**For PowerShell**

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

<br><br/>
**For Windows Command Prompt (CMD)**

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

<br><br/>
**For macOS / Linux**

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

---

# Features

- Simple image upload through a web interface
- Automatic image **deblurring**
- Automatic **colorization**
- Preview enhanced images
- Download processed images as **PNG or JPG**

---

# Workflow

1. User uploads an image.
2. The system processes the image to **remove blur**.
3. The system **restores color** where needed.
4. The enhanced image is generated.
5. User downloads the final result.

---

# Tech Stack (Proposed)

**Streamlit** is suggested for the initial version since it allows quick development of both frontend and backend.  
If needed, the frontend can later be replaced with a dedicated web framework.

---

# Project Components

## Backend
- Implement image upload API
- Build the image processing pipeline
- Manage file storage and output generation
- Support **PNG and JPG** downloads

## Frontend
- Build a clean and simple UI
- Image upload component
- Processing status indicator
- Image preview and download functionality

## Model / Processing
- Integrate a **deblurring model**
- Integrate a **colorization model**
- Optimize inference speed
- Handle different image sizes and formats

---

# Model Strategy

There are currently two approaches being considered:

### 1. Custom Trained Models
Train our own models for:
- Image deblurring
- Image colorization

This gives more control over performance and quality.

### 2. Pretrained / Open Source Models
Use existing models or APIs that are already available.

Benefits:
- Faster development
- Less compute required
- Easier experimentation

Users may also be given the option to **choose which model to run**, allowing comparison of:
- Processing time
- Output quality

---

# Project Roles

Roles will be used to **track task ownership and project progress**.


``` I usually use python hope everyone is okay with doing whole project in python frontend we can discuss and agree upon ```

### Backend Development

### Frontend Development

### Data Gathering & Preparation

### Model Training & Testing

### Open Source Model Exploration

---
