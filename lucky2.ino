#include <Wire.h>
#include <Adafruit_MotorShield.h>

Adafruit_MotorShield AFMS = Adafruit_MotorShield();
Adafruit_DCMotor *myMotor1 = AFMS.getMotor(1);  // Base
Adafruit_DCMotor *myMotor2 = AFMS.getMotor(2);  // Lift
Adafruit_DCMotor *myMotor3 = AFMS.getMotor(3);  // Arrange
Adafruit_DCMotor *myMotor4 = AFMS.getMotor(4);  // Gripper

// Motor positions
// const int M1_INIT = 804;  // Base initial
// const int M2_INIT = 447;  // Lift initial (updated)
// const int M3_INIT = 293;  // Arrange initial (updated)
// const int M4_OPEN = 535;  // Gripper open //488
// const int M4_CLOSE = 570; // Gripper closed //560


// const int M2_Pickup = 523;
// const int M3_Pickup = 385;

const int M1_INIT = 700;  // Base initial
const int M2_INIT = 512;  // Lift initial (updated)
const int M3_INIT = 157;  // Arrange initial (updated)
const int M4_OPEN = 535;  // Gripper open //488
const int M4_CLOSE = 570; // Gripper closed //560


const int M2_Pickup = 580;
const int M3_Pickup = 240;

// Position definitions for different cases (left, right, center)
const int M1_LFINAL = 630;  // Base final for left
const int M2_LFINAL = 545;   // Lift final for left
const int M3_LFINAL = 650;   // Arrange final for left

const int M1_RFINAL = 840;   // Base final for right
const int M2_RFINAL = 545;   // Lift final for right
const int M3_RFINAL = 650;   // Arrange final for right

const int M1_CFINAL = 770;   // Base final for center
const int M2_CFINAL = 545;   // Lift final for center
const int M3_CFINAL = 650;   // Arrange final for center

// Potentiometer pins
const int POT_PIN1 = A12; 
const int POT_PIN2 = A11;
const int POT_PIN3 = A10;
const int POT_PIN4 = A8;
const int TOLERANCE = 5;
 
void setup() {
  Serial.begin(9600);
  
  if (!AFMS.begin()) {
    Serial.println("Could not find Motor Shield. Check wiring.");
    while (1);
  }
  
  // Set initial speeds
  myMotor1->setSpeed(255);
  myMotor2->setSpeed(255);
  myMotor3->setSpeed(255);
  myMotor4->setSpeed(255);
  
  // Move to initial positions
  Serial.println("Moving to initial positions...");
  moveMotor(myMotor1, POT_PIN1, M1_INIT, "Base");
  moveMotor(myMotor2, POT_PIN2, M2_INIT, "Lift");
  moveMotor(myMotor3, POT_PIN3, M3_INIT, "Arrange");
  moveMotor(myMotor4, POT_PIN4, M4_OPEN, "Gripper");
  delay(2000);
}

void loop() {
  Serial.println("\n=== Waiting for Input from Python ===");

  String userInput = getPythonInput();  // Get input from Python

  if (userInput == "L") {
    moveAllMotors(M1_LFINAL, M2_LFINAL, M3_LFINAL);  // Move to final positions for left
    sendStatusToPython("Done");
  } 
  else if (userInput == "R") {
    moveAllMotors(M1_RFINAL, M2_RFINAL, M3_RFINAL);  // Move to final positions for right
    sendStatusToPython("Done");
  } 
  else if (userInput == "U") {
    moveAllMotors(M1_CFINAL, M2_CFINAL, M3_CFINAL);  // Move to final positions for center
    sendStatusToPython("Done");
  } else {
    Serial.println("Invalid input received. Please send 'L', 'R', or 'U'.");
    sendStatusToPython("Invalid");
  }

  Serial.println("\n=== Cycle Complete ===");
  delay(800); 
}

String getPythonInput() {
  while (Serial.available() == 0) {
    // Wait for input from Python script
  }

  String input = Serial.readStringUntil('\n');  // Read input until newline
  input.trim();  // Remove any leading or trailing whitespace
  Serial.print("Received input from Python: ");
  Serial.println(input);  // Print the received input for debugging
  return input;
}

void sendStatusToPython(String status) {
  Serial.print("Status: ");
  Serial.println(status);  // Send the status message back to Python
}


void moveAllMotors(int M1_FINAL, int M2_FINAL, int M3_FINAL) {
  
  // Step 1: M3 moves to final position
  Serial.println("\nStep 2: Moving arrangement");
  moveMotor(myMotor3, POT_PIN3, M3_Pickup, "Arrange");
  delay(100);

 // Step 2: M2 moves to final position
  Serial.println("\nStep 3: Moving lift");
  moveMotor(myMotor2, POT_PIN2, M2_Pickup, "Lift");
  delay(100);

  // Step 3: Gripper closes
  Serial.println("\nStep 1: Closing gripper");
  moveMotor(myMotor4, POT_PIN4, M4_CLOSE, "Gripper");
  delay(100);
  
  // Step 5: M2 moves to final position
  Serial.println("\nStep 3: Moving lift");
  moveMotor(myMotor2, POT_PIN2, M2_INIT, "Lift");
  delay(100);

  // Step 4: M3 moves to final position
  Serial.println("\nStep 2: Moving arrangement");
  moveMotor(myMotor3, POT_PIN3, M3_INIT, "Arrange");
  delay(100);
  
  // Step 6: M1 moves to final position
  Serial.println("\nStep 4: Moving base");
  moveMotor(myMotor1, POT_PIN1, M1_FINAL, "Base");
  delay(100);
  
  // Step 7: M2 moves to initial position
  Serial.println("\nStep 5: Lowering lift");
  moveMotor(myMotor2, POT_PIN2, M2_FINAL, "Lift");
  delay(100);
  
  // Step 8: M3 moves to initial position
  Serial.println("\nStep 6: Moving arrangement back");
  moveMotor(myMotor3, POT_PIN3, M3_FINAL, "Arrange");
  delay(100);
  
  // Step 10: M3 moves to initial position
  Serial.println("\nStep 8: Moving arrangement");
  moveMotor(myMotor3, POT_PIN3, M3_INIT, "Arrange");
  delay(100);
  
  // Step 11: M2 moves to initial position
  Serial.println("\nStep 9: Raising lift");
  moveMotor(myMotor2, POT_PIN2, M2_INIT, "Lift");
  delay(100);
  
  // Step 12: M1 moves to initial position
  Serial.println("\nStep 10: Moving base back");
  moveMotor(myMotor1, POT_PIN1, M1_INIT, "Base");
  delay(100);

    // Step 1: M3 moves to final position
  Serial.println("\nStep 2: Moving arrangement");
  moveMotor(myMotor3, POT_PIN3, M3_Pickup, "Arrange");
  delay(100);

 // Step 2: M2 moves to final position
  Serial.println("\nStep 3: Moving lift");
  moveMotor(myMotor2, POT_PIN2, M2_Pickup, "Lift");
  delay(100);

  // Step : Gripper opens
  Serial.println("\nStep 7: Opening gripper");
  moveMotor(myMotor4, POT_PIN4, M4_OPEN, "Gripper");
  delay(100);
  
  // Step 5: M2 moves to final position
  Serial.println("\nStep 3: Moving lift");
  moveMotor(myMotor2, POT_PIN2, M2_INIT, "Lift");
  delay(100);

  // Step 4: M3 moves to final position
  Serial.println("\nStep 2: Moving arrangement");
  moveMotor(myMotor3, POT_PIN3, M3_INIT, "Arrange");
  delay(100);
}

void moveMotor(Adafruit_DCMotor *motor, int potPin, int targetPos, String motorName) {
  int currentPos = analogRead(potPin);

  while (abs(currentPos - targetPos) > TOLERANCE) {
    currentPos = analogRead(potPin);

    Serial.print(motorName);
    Serial.print(" Position: ");
    Serial.print(currentPos);
    Serial.print(" | Target: ");
    Serial.println(targetPos);

    if (currentPos < targetPos) {
      motor->run(FORWARD);
    } else {
      motor->run(BACKWARD);
    }

    int speed = map(abs(currentPos - targetPos), 0, 512, 50, 255);
    motor->setSpeed(speed);

    delay(50);
  }

  motor->run(RELEASE);
  Serial.println(motorName + " reached position!");
}
