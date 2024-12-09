This project automates pill identification and sorting using computer vision, machine learning, and a robotic arm.

ArduinoArm.ino: Controls the robotic arm via serial commands from Python.
Main.py: Core script for identifying pills, detecting containers, and deciding where the pill goes based on color and position.
- Supporting Files
ColorStore.py: Captures pill images for training.
PredictLive.py: Live classification tool for testing.
RFPill.py: Trains the Random Forest model (used in the project).
SVMClassifier.py: Alternative model tested (less effective than Random Forest).
pill_classifier.pkl: Pre-trained Random Forest model for pill classification.
- 3D Models
Container.stl and Pill Holder.3mf: 3D-printed components for holding pills and containers.
