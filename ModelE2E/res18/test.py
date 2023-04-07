from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

labels = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]	# 실제 labels
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]	# 에측된 결과

print(f1_score(labels, guesses))	# 0.46