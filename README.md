equal_split_class = training a classification alg using efficientnet to distinguish (equal split between GOOD and BAD embryos, made classifcation much more accurate bc we originally had like 86% bad, 14% good so good classification was rlly bad while bad ones were accurate. had very low precision but high recall aka lots of false pos, few false neg)
![image](https://github.com/user-attachments/assets/b2ed0fff-e2fe-4bf3-b77d-8fe616df0a35)

webcam_class = using webcam and doing yolo (pretrained library of stuff like person, lamp etc) and using roi

webcam_classv2 = the above one but we neglect things outside roi and beneath 0.5 confidence in prediction
