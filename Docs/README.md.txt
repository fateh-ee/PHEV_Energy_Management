Codes are provided according to the following order:
1. Driving cycle code to generate acceleration data from velocity
2. Applying Rule-based method on driving cycles
3. Applying GA on driving cycles and getting best solution for the best segment size
4. Using GA to create training dataset for all different segment sizes of driving cycles
5. DL model for training
6. DL model for predicting and running simulation (using saved best model from tuner)