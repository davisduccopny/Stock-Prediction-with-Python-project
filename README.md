# Prediction Stock With Python Project
#### Website demo: https://stock-prediction-with-python-project-quocchienduc.streamlit.app/
![Project Image](./asset/image/homepage.png)
<div align="center">
    
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

</div>

## Table of contents
- [Install](#cách-cài-đặt-installation)
- [Project Perpose](#mục-đích-dự-án-project-purpose)
- [Usage](#cách-sử-dụng-usage)
- [Directory Structure](#cấu-trúc-thư-mục-directory-structure)
- [Contribution](#đóng-góp-contribution)
- [Author](#tác-giả-author)
- [License](#giấy-phép-license)

## Project purpose

This project focuses on developing a stock price prediction system using Python. We will use machine learning techniques to analyze historical data and predict future stock price trends.

## Installation

1. Clone the project to your local machine:

    ```bash
    git clone https://github.com/davisduccopny/Stock-Prediction-with-Python-project.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Stock-Prediction-with-Python-project
    ```

3. Create a virtual environment (if needed):

    ```bash
    virtualenv venv
    ```

4. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```
    - For the streamlit environment:
        ```bash
        pip freeze > requirements.txt
        ```
5. Install the necessary libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run notebook `main_train.ipynb` to view the detailed prediction process.

2. Customize parameters and models according to your needs. 

3. Monitor prediction results and adjust the model to improve performance.

## Directory Structure

- `dataset/`: Contains historical stock price data.
- `train_folder/`: Contains Jupyter notebooks for analysis and prediction.
- `asset/image/`: Stores project images
- `info_stock/`: Stores stock information.
- `introduction`: Word draft of the project.
- `app_test.py`: The file contains the algorithm demo interface

## Contribution

If you want to contribute to the project, create a new branch and submit a pull request. We would be delighted to receive your contributions!

## Author

- Name: [Data Team - QuocChienDuc]
- Email: 2156210125@hcmussh.edu.vn
- Project members:
    | Name         | Student Number | Email                          |
    |----------------------|-------------------------------|--------------------------------|
    | Hoàng Xuân Quốc      | 2156210125                    | 2156210125@hcmussh.edu.vn      |
    | Đặng Hoàng Chiến     | 2156210095                    | 2156210095@hcmussh.edu.vn      |
    | Nguyễn Viết Đức      | 2156210100                    | 2156210100@hcmussh.edu.vn      |
- GitHub: [Your GitHub Profile](https://github.com/davisduccopny/)

## license

This project is distributed under license [MIT License](LICENSE).

---
Happy coding!

[contributors-shield]: https://img.shields.io/github/contributors/davisduccopny/Stock-Prediction-with-Python-project?style=for-the-badge&label=Contributors 
[contributors-url]:https://github.com/davisduccopny/Stock-Prediction-with-Python-project/graphs/contributors 
[forks-shield]:https://img.shields.io/github/forks/davisduccopny/Stock-Prediction-with-Python-project?label=Folks&style=for-the-badge
[forks-url]: https://github.com/davisduccopny/Stock-Prediction-with-Python-project/forks
[stars-shield]: https://img.shields.io/github/stars/davisduccopny/Stock-Prediction-with-Python-project?style=for-the-badge&label=Stars
[stars-url]: https://github.com/davisduccopny/Stock-Prediction-with-Python-project/stargazers
[issues-shield]: https://img.shields.io/github/issues/davisduccopny/Stock-Prediction-with-Python-project?style=for-the-badge&label=Issues
[issues-url]: https://github.com/davisduccopny/Stock-Prediction-with-Python-project/issues
