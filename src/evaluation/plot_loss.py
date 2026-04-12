import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Trich xuat so lieu Val Loss thuc te tu nhat ky huan luyen cua ban
epochs = np.arange(1, 21)

# Val Loss cua Model B (Siamese thuong)
val_loss_b = [0.2184, 0.2003, 0.1885, 0.1898, 0.1907, 0.1753, 0.1799, 0.1811, 0.1672, 0.1695, 
              0.1663, 0.1632, 0.1670, 0.1642, 0.1642, 0.1659, 0.1656, 0.1686, 0.1635, 0.1650]

# Val Loss cua Model C (Hard Negative Mining)
val_loss_c = [0.1513, 0.1551, 0.1567, 0.1523, 0.1535, 0.1559, 0.1440, 0.1475, 0.1494, 0.1461, 
              0.1495, 0.1535, 0.1488, 0.1460, 0.1512, 0.1478, 0.1481, 0.1514, 0.1518, 0.1444]

# 2. Thiet lap bieu do
plt.figure(figsize=(10, 6))

# Ve 2 duong bieu dien
plt.plot(epochs, val_loss_b, marker='o', linestyle='-', color='#2196F3', linewidth=2, label='Model B (Siamese Thuong)')
plt.plot(epochs, val_loss_c, marker='s', linestyle='-', color='#f44336', linewidth=2, label='Model C (Hard Negative)')

# Danh dau diem tot nhat (Lowest Val Loss)
best_epoch_b = 12
best_loss_b = 0.1632
plt.scatter(best_epoch_b, best_loss_b, color='blue', s=100, zorder=5)
plt.annotate(f'Best: {best_loss_b}', (best_epoch_b, best_loss_b), textcoords="offset points", xytext=(0,10), ha='center')

best_epoch_c = 7
best_loss_c = 0.1440
plt.scatter(best_epoch_c, best_loss_c, color='red', s=100, zorder=5)
plt.annotate(f'Best: {best_loss_c}', (best_epoch_c, best_loss_c), textcoords="offset points", xytext=(0,-15), ha='center')

# 3. Trang tri bieu do
plt.title('So sanh Validation Loss: Model B vs Model C (Hard Negative Mining)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Triplet Loss (Validation)', fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 4. Luu file anh
os.makedirs('plots', exist_ok=True)
output_path = 'plots/val_loss_comparison.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Da ve xong! Bieu do duoc luu tai: {output_path}. Ban hay copy anh nay cho vao Bao cao Word nhe!")