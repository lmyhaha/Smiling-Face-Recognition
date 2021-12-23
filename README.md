# Smiling Face Recognition

It consists of three parts, namely face detection, smile detection, and real time detection.

Used the HOG features to complete real-time and smiling faces with low latency and high accuracy. Classified smiling faces on the captured images by using SVM algorithms.

## Requirements

opencv-python version 3.4.4.19

pillow version 8.4.0

## Getting Started

### Face Detection

```
python Face_Detection.py
```

### Smile Detection

```
python smile_detection.py
```

### Real Time Detection

```
python real_time_detect.py
```

## Results

### Face Detection

Total picture number: 4000
Recognized picture number: 3756
Accuracy: 0.939
Time cost: 187.77138924598694s

### Smile Detection

|      |         F1         |  TP  |  TN  |  FP  |  FN  |
| :--: | :----------------: | :--: | :--: | :--: | :--: |
|  0   | 0.8704156479217604 | 178  | 144  |  29  |  24  |
|  1   | 0.8740740740740741 | 177  | 140  |  26  |  25  |
|  2   | 0.8704156479217604 | 178  | 129  |  28  |  25  |
|  3   | 0.8329177057356608 | 167  | 133  |  30  |  37  |
|  4   | 0.8626506024096385 | 179  | 137  |  32  |  25  |
|  5   | 0.9046454767726161 | 185  | 151  |  20  |  19  |
|  6   | 0.8735083532219571 | 183  | 140  |  31  |  22  |
|  7   | 0.8823529411764706 | 180  | 149  |  23  |  25  |
|  8   | 0.8801955990220048 | 180  | 144  |  28  |  21  |
|  9   | 0.8693467336683417 | 173  | 148  |  24  |  28  |