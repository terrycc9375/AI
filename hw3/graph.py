import matplotlib.pyplot as plt

val_acc = [0.8071296296296296, 0.8168518518518518, 0.812962962962963, 0.8162037037037037, 0.8125, 0.8115740740740741, 0.812037037037037, 0.8124074074074074, 0.8106481481481481, 0.8112962962962963, 0.8087037037037037, 0.8085185185185185, 0.8123148148148148, 0.8072222222222222, 0.8060185185185185, 0.8058333333333333]
test_acc = [0.8056666666666666, 0.8166666666666667, 0.8131666666666667, 0.8141666666666667, 0.813, 0.8116666666666666, 0.8105, 0.8143333333333334, 0.8115, 0.8093333333333333, 0.806, 0.8125, 0.8096666666666666, 0.8106666666666666, 0.8095, 0.8006666666666666]

epochs = range(1, len(val_acc) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_acc, 'b-o', label='Validation Accuracy', markersize=4)
plt.plot(epochs, test_acc, 'r-o', label='Test Accuracy', markersize=4)

plt.title('BERT-base-uncased Training Curve', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig('./logs/FacebookAI/roberta-large.png', dpi=300, bbox_inches='tight')
