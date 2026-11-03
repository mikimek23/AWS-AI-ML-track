### Experiment: Testing the Model with Challenging Messages

I tested the trained Multinomial Naive Bayes model using five additional SMS messages designed to challenge its classification ability.

The model classified all five messages as Ham. However, three of them were actually spam, resulting in three False Negatives and two True Negatives.

### Results

| My message | Model said | Should be | Why it failed |
|---|---|---|---|
| "Hey! Just checking if you received the money. Your account has been credited with a special bonus, so please confirm your details." | Not Spam (Ham) | Spam | Used friendly conversational language, making it look like a normal message. |
| "Hi, your mobile number has been selected for a cash reward. You can check it whenever you have time." | Not Spam (Ham) | Spam | Used ordinary language instead of obvious spam keywords. |
| "Congratulations! You finally won the football competition. Let's celebrate this weekend!" | Not Spam (Ham) | Not Spam (Ham) | Correctly identified it as a normal personal message. |
| "Your monthly subscription is ready for renewal. Visit our website to confirm your payment and receive your special bonus." | Not Spam (Ham) | Spam | Looked like a legitimate service notification rather than a spam advertisement. |
| "Hey, I transferred the money to your account. Please check your balance and let me know if you received it." | Not Spam (Ham) | Not Spam (Ham) | Correctly identified normal financial communication as legitimate. |

### Results Summary

- **Total messages tested:** 5
- **Correct predictions:** 2
- **Incorrect predictions:** 3
- **True Positives (TP):** 0
- **True Negatives (TN):** 2
- **False Positives (FP):** 0
- **False Negatives (FN):** 3
- **Accuracy:** 40%

### Why the Model Failed

The Multinomial Naive Bayes model relies on word frequencies and statistical patterns learned from its training dataset rather than understanding the actual meaning or intention of a message.

Some spam messages used ordinary conversational language, making them difficult to distinguish from legitimate messages. The model also struggled with unfamiliar combinations of words.



