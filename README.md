# SoundMatchAnalyser (SMA)

## Project Description

**SoundMatchAnalyser (SMA)** is a powerful tool designed to analyze and compare audio quality by evaluating the differences between two audio files. It reads and processes audio files to measure key audio parameters such as LUFS (Loudness Units Full Scale), PEAQ (Perceptual Evaluation of Audio Quality), Total Harmonic Distortion (THD), and more. SMA is particularly useful for audio engineers, researchers, developers, and hobbyists who aim to assess the performance of audio processing algorithms and neural network models.

### Key Features:

- **PEAQ Analysis**: Use perceptual models to assess the quality differences between audio signals.
- **LUFS Measurement**: Evaluate the perceived loudness differences between audio signals.
- **SNR Calculation**: Measure the signal-to-noise ratio between audio signals.
- **THD Calculation**: Measure harmonic distortion between audio signals.
- **Gain Adjustment**: Determine the optimal scaling factor to match the audio signals.
- **MSE Calculation**: Compute the mean squared error between audio signals.
- **Combined Scoring**: Aggregate multiple metrics into a comprehensive combined score.
- **Neural Network Evaluation**: Assess the performance of neural network models in audio processing by comparing original and processed audio.
- **Automatic Gain and Latency Adjustment**: Automatically adjust gain and latency between audio files, so there's no need for manual alignment.
- **Output Difference Files**: Generate difference files for further analysis.

### Detailed Descriptions of Key Metrics

- **PEAQ (Perceptual Evaluation of Audio Quality)**:
  PEAQ is a standard algorithm used to evaluate the perceptual quality of audio signals. It models how humans perceive sound quality and provides a score that indicates the degree of degradation between two audio signals. This metric is particularly useful for assessing the impact of audio processing algorithms on perceived audio quality.

- **LUFS (Loudness Units Full Scale)**:
  LUFS measures the perceived loudness of audio signals. Unlike simple volume metrics, LUFS takes into account the frequency and amplitude characteristics that affect human perception of loudness. In the context of SMA, LUFS is used to evaluate the loudness differences between two audio files, helping to ensure consistent audio levels.

- **SNR (Signal-to-Noise Ratio)**:
  SNR quantifies the ratio of the desired signal to the background noise. A higher SNR indicates a cleaner and clearer audio signal. By comparing the SNR of two audio files, SMA helps to determine the effectiveness of noise reduction and other audio enhancement techniques.

- **THD (Total Harmonic Distortion)**:
  THD measures the harmonic distortion present in an audio signal, expressed as a percentage. It indicates how much the signal deviates from a pure sine wave due to harmonic frequencies. Lower THD values suggest higher fidelity and less distortion, making it a crucial metric for audio quality assessment.

- **Gain Adjustment**:
  Gain adjustment determines the optimal scaling factor to match the audio levels of two signals. SMA automatically calculates and applies the best gain to align the loudness of the processed audio with the original, ensuring a fair comparison.

- **MSE (Mean Squared Error)**:
  MSE computes the average squared difference between corresponding samples of two audio signals. It provides a quantitative measure of the error between the original and processed audio, with lower MSE values indicating more accurate reproduction.

## Installation Instructions

### Prerequisites

- Python 3.7 or higher
- `pip` (Python package installer)

### Step 1: Clone the Repository

First, clone the repository from GitHub to your local machine. Open your terminal and run:

git clone https://github.com/CPCCoder/soundmatchanalyser.git

Navigate into the project directory:

cd soundmatchanalyser

### Step 2: Create a Virtual Environment (Optional but Recommended)

It is recommended to use a virtual environment to manage your project dependencies. You can create a virtual environment named "SMA" using `venv`:

python -m venv SMA

Activate the virtual environment:

- On Windows:

  .\SMA\Scripts\activate

- On macOS and Linux:

  source SMA/bin/activate

### Step 3: Upgrade `pip`

Before installing the dependencies, it's a good practice to upgrade `pip` to the latest version. Run the following command:

pip install --upgrade pip

### Step 4: Install Dependencies

Use `pip` to install the required dependencies listed in the `requirements.txt` file:

pip install -r requirements.txt

### Step 5: Run the Application

Once all dependencies are installed, you can run the main application. Use the following command:

python SMA.py

## Usage Instructions

1. **Open the Application**: After running the main application, the GUI will appear.
2. **Select Files**: Use the "Browse" buttons to select the amplifier file (original audio) and the inference file (processed audio). You can also optionally select a file to save the differences. There's no need to manually adjust gain or latency, as the program handles this automatically.
3. **Calculate Metrics**: Click the "Calculate" button to analyze the files. The results will be displayed in the GUI, including metrics like PEAQ, LUFS, SNR, THD, Gain, and MSE.
4. **View Results**: The accuracy score and other metrics will be displayed in the GUI for your review.

### Additional Notes

- If you encounter any issues during installation, please refer to the [troubleshooting section](#troubleshooting) or raise an issue on the GitHub repository.
- For detailed usage instructions, please refer to the [documentation](docs/documentation.md).

## Contributing

If you would like to contribute, please fork the repository, create a new branch, and submit a pull request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Legal Notice

This project uses the following technologies:
- **PEAQ (Perceptual Evaluation of Audio Quality)**: PEAQ is an international standard for evaluating audio quality. Please ensure you have the appropriate licenses if you plan to use this technology beyond non-commercial purposes.
- **LUFS (Loudness Units Full Scale)**: LUFS measures the perceived loudness of audio signals. This method is freely usable, but please adhere to relevant standards and guidelines.

For further information, please consult the respective documentation and licensing information for each technology.
