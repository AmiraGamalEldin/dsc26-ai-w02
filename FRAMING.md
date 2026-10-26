# Framing: Rawaj Pay mule detection

<!-- Write a few sentences under each heading about Rawaj Pay's mule detection.
     Keep the four headings exactly as they are. Text inside these comments does not count.
     The grader checks only that each heading has some text under it; your Track Mentor reads it. -->

## Decision

My goal is to help Rawaj Pay identify suspicious transfers without causing too many problems for genuine customers. If the system flags too many normal transfers, customers may face unnecessary delays. But if it misses suspicious transfers, the company could lose money. I need to find a threshold that balances these two risks.

## Metric

I would use expected cost as my main metric because not all mistakes have the same impact. Flagging a legitimate transfer by mistake is a false positive, while missing a fraudulent transfer is a false negative. I would compare the costs of both mistakes to choose a threshold that makes sense for the business. I would also check precision and recall to understand how well the system identifies suspicious transfers.

## Baseline

Before relying on a machine-learning model, I would start with a simple baseline and see how well it performs. Then, I would compare it with the model using the same evaluation data. I would look at expected cost, precision, and recall to check whether the model actually improves the results.

## Classical ML or LLM

I would start with classical machine learning because Rawaj Pay works with structured transfer data, and I need a consistent way to score suspicious transactions. An LLM could help explain the audit results or document the decisions, but I would not rely on it as the main fraud-detection system without testing whether it is suitable for this task.