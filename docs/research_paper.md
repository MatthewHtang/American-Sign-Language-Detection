# American Sign Language Detection: Machine Learning Research

**Dallas College, Brookhaven Campus — May 10, 2026**

**Author:** Matthew H Htang, Dr. George Hademenos

## Abstract

Communication barriers between hearing-impaired individuals and those who do not understand sign language remain a major challenge that often goes unaddressed in everyday interaction. This ASL Detection honors research project explores a machine learning system for detecting and recognizing American Sign Language (ASL) using Python. The project consists of three main programs: one for collecting the dataset, one for testing the model, and one for converting text to an audio file that is automatically played after a complete sentence.

The dataset was collected using a computer webcam to capture static images of hand gestures. For greater accuracy, landmark detection was used while collecting the data to train a gesture recognition model capable of identifying ASL alphabet signs. The model itself was trained on Teachable Machine, a web-based tool provided by Google Creative Lab for building models quickly and simply. This experimental research lasted five months and demonstrated, with remarkable accuracy, that the model could effectively detect ASL alphabets across different lighting settings and environments.

## Table of Contents

1. [Libraries](#1-libraries)
2. [Data Collection](#2-data-collection)
3. [Model](#3-model)
4. [Algorithm Implementation](#4-algorithm-implementation)
5. [API Request for Text-to-Speech Conversion](#5-api-request-for-text-to-speech-conversion)
6. [Conclusion & Future Objectives](#6-conclusion--future-objectives)
7. [References](#references)

## 1. Libraries

This ASL detection project worked best on certain versions of the Python environment and its libraries. Development initially began on Python 3.12+, but some libraries — TensorFlow in particular — did not work consistently with others in that environment. After testing compatibility across environments, **Python 3.11.0** proved to be the most stable choice.

Six main Python libraries were used, each with a version chosen for compatibility with the others:

| Library | Version | Purpose |
|---|---|---|
| NumPy | 1.26.4 | Handles numbers and arrays; processes image data and model inputs |
| OpenCV | 4.9.0.80 | Captures images/video and processes frames (e.g., resizing to isolate the hand gesture and discard unnecessary frames) |
| MediaPipe | 0.10.13 | Detects hand landmarks (finger positions) to obtain precise hand coordinates |
| CVZone | — | Built on OpenCV and MediaPipe; provides ready-to-use hand tracking functions |
| TensorFlow | — | Core machine learning framework and backbone of the system |
| Matplotlib / Protocol Buffers | — | Supporting libraries for accuracy visualization and data serialization |

## 2. Data Collection

A dedicated script, `dataCollection.py`, handles collecting, sorting, and assigning images into their respective sub-folders. Images are captured through a computer camera as static frames with hand landmarks and coordinates overlaid, then stored in individual folders per character (e.g., `A`, `B`, `C`, ...).

Model accuracy turned out to depend heavily on the range and distinctiveness of the dataset. Data bias was an early problem: collecting images from only one room produced a model with roughly 10% accuracy. Broadening the dataset to **six different environments** with **at least six different lighting settings** solved this.

- 100 images were collected per environment
- 600 images were collected total per character (e.g., letter "A")
- The letters "J" and "Z" were excluded, since they require motion
- ~12,000 static images were used in total to train the model

> **Tip:** Include the environment where you'll be presenting your project as one of your data collection environments — this maximizes accuracy during the live demo.

## 3. Model

Training the model itself was not the main challenge of this project. After evaluating approaches, the model was trained using **Teachable Machine** by Google, since all the data consisted of static images and the platform offers tools to build a machine learning model quickly without extensive coding.

The training setup involved 25 classes representing the alphabets/characters to classify, with images (captured via `dataCollection.py`, including hand landmark coordinates) uploaded for each — over 500 static images per character.

Key training parameters:

- **Epochs:** One epoch means the model has seen every data point once. Training was set to **2,000 epochs** — the model iterated through all 12,000 images 2,000 times.
- **Batch size:** Set to **32**. With 12,000 samples, this yields 12,000 ÷ 32 = 375 batches (iterations) per epoch, or 2,000 × 375 = 750,000 total iterations across training.
- **Learning rate:** Left at the platform default.

Despite the common assumption that more epochs improve performance, this did not hold here. That relationship tends to apply to massive datasets — with only 12,000 images, pushing epochs too high while the dataset stayed small caused **overfitting**: the model became overly sensitive to small lighting changes, and accuracy could drop to as low as 10%. To find the right balance, models were trained across a range of epoch values from 40 to 2,000, and the best-performing configuration was selected.

## 4. Algorithm Implementation

The harder problem to solve wasn't training the model — it was implementing the logic to spell out alphabets, arrange them, and assemble a complete sentence before sending it to the speech API. The following code performs the collection, sorting, and arrangement of characters into a complete sentence.

```python
# Check if the same label continues
current_label = labels[index]

if current_label == previous_label:
    elapsed = time.time() - start_time

    # Add first character after 0.9 seconds
    if elapsed > 0.9 and not letter_added:
        outPut += current_label
        letter_added = True
        space_added = False

    # Add second character after 1.9 seconds (for double letters)
    elif elapsed > 1.9 and not double_letter_added:
        outPut += current_label
        double_letter_added = True

    else:
        # Reset timer if label changes
        previous_label = current_label
        start_time = time.time()
        elapsed = 0
        letter_added = False
        double_letter_added = False
        space_added = False

# Add space when hand disappears
else:
    if not hands and not space_added and outPut != "":
        outPut += " "
        space_added = True
        letter_added = False
        double_letter_added = False
```

### How it works

**Detecting a held label**
```python
current_label = labels[index]
```
Checks whether the current label is the same as the previous one; the logic below only runs while that condition holds.

**Starting the timer**
```python
elapsed = time.time() - start_time
```
Since Python doesn't track "held" time natively, elapsed time is computed by subtracting the start time from the current time — giving how long the same letter has been held.

**Adding the first character**
```python
# Add first character after 0.9 seconds
if elapsed > 0.9 and not letter_added:
    outPut += current_label
    letter_added = True
    space_added = False
```
If the user holds the same character for 0.9 seconds and it hasn't been added yet:
1. The current label is appended to `outPut`.
2. `letter_added` is set as a flag to prevent the same letter from being added repeatedly.
3. `space_added` is reset to `False`, since no space has been added yet.

**Adding a double letter**
```python
# Add second character after 1.9 seconds (for double letters like TT)
elif elapsed > 1.9 and not double_letter_added:
    outPut += current_label
    double_letter_added = True
```
If elapsed time exceeds 1.9 seconds and a double letter hasn't already been added:
1. The current label is appended again, right after the first instance.
2. `double_letter_added` is set to `True`.

**Resetting the timer**
```python
# Reset timer if label changes
else:
    previous_label = current_label
    start_time = time.time()
    elapsed = 0
    letter_added = False
    double_letter_added = False
    space_added = False
```
Once a new character is detected, the timer resets so the logic above can run again: `previous_label` is updated, `start_time` restarts, `elapsed` resets to 0, and all flags return to `False`.

**Adding spaces**

The initial approach reused the same timer logic — treating 0.9s as a new letter, 1.9s as a double letter, and 2.5s as a space. This failed immediately, since a space (the third state, at 2.5s) could not be reached without first triggering the double-letter state at 1.9s; elapsed time can't skip over an intermediate threshold.

The working solution instead relies on hand detection:

```python
if not hands and not space_added and outPut != "":
    outPut += " "
    space_added = True
    letter_added = False
    double_letter_added = False
```
When the camera no longer detects a hand, that absence is treated as a space. It's not a perfect solution, but it was the most efficient way to add full spelling support to the system.

**Ending a sentence**

A period symbol (`.`) marks a completed sentence and triggers the text-to-speech conversion:

```python
# When label ends with a period
if outPut.endswith("."):
    speak(outPut)
```

## 5. API Request for Text-to-Speech Conversion

With the program able to collect, recognize, and sort characters correctly, the final step is converting the completed sentence into speech. This is handled by a separate `TTS.py` script, which requests the **Inworld AI** API to synthesize natural-sounding speech, saves it as an MP3 file, and plays it back from the main program. Inworld AI supports 15 languages and over 130 pre-built voices, making it well suited to this project.

## 6. Conclusion & Future Objectives

This project successfully translated real-time American Sign Language (ASL) into text and speech. By combining computer vision, a trained image classification model, and a text-to-speech API, the system detects hand gestures, constructs sentences, and converts them into audible output.

Experimental results showed that epoch values in the **40–100** range gave the best balance between accuracy and generalization. Increasing epochs without a correspondingly larger dataset led to overfitting — strong performance on training data, weaker performance under real-world conditions. The system's accuracy also proved sensitive to lighting, background, and hand orientation.

**Future work** could include:
- Expanding the dataset with more diverse samples to improve accuracy
- Supporting real-time dynamic hand gestures, requiring a more advanced training pipeline and a different model architecture for collection, sorting, and conversion
- Extending support to additional languages such as Russian, Spanish, Japanese, and German

Overall, this project demonstrates that combining machine learning, computer vision, and speech synthesis can produce a working ASL translation system — contributing toward more effective and accessible communication technology.

## References

- GeeksforGeeks. (2025, July 23). *Sign language recognition system using TensorFlow in Python*. GeeksforGeeks.
- Inworld Portal. (n.d.). [platform.inworld.ai](https://platform.inworld.ai/v2/workspaces/default-a49gp-7jjr-fz777nzblgg)
- Murtaza's Workshop – Robotics and AI. (2022, July 4). *Easy hand sign detection | American Sign Language ASL | Computer Vision* [Video]. YouTube. [https://www.youtube.com/watch?v=wa2ARoUUdU8](https://www.youtube.com/watch?v=wa2ARoUUdU8)
