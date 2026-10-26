# AI usage

<!-- Replace every comment below with your own words. Short is fine. The grader checks only
     that this file exists and has been changed from this template; your Track Mentor reads it. -->

## Tools I used, and for which functions

I used ChatGPT mainly to clarify the task requirements and explain some Python concepts. I worked on implementing and reviewing rawaj/metrics.py and rawaj/audit.py, then ran the provided tests to check my code. I also wrote the project framing and used AI feedback to improve my understanding of threshold selection.

## One place where the AI was wrong, incomplete, or not what the spec said

While checking my code, I asked the AI to check function and it checked more than one together, but the code produced an error. At first, I could not understand why the error happened. I had to go back to the code, check the error message, and work through the problem step by step. This taught me that reviewing multiple parts together make errors because one of them didn't implement yet, and that I should test the first part only.

## The function I wrote without AI, and why

I worked on implementing `confusion_at()` and `precision_recall_at()` in `rawaj/metrics.py`. I wanted to understand how the model's predictions are evaluated at a given threshold. These functions helped me learn how to count true positives, false positives, true negatives, and false negatives, and how to calculate precision and recall from these values. I also reviewed and tested the other functions to understand how they work and make sure they follow the task requirements. This helped me understand how different thresholds affect the results and why choosing the right threshold matters for Rawaj Pay.