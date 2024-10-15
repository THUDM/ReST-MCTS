import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# sns.set_theme(style="darkgrid")
all_data = []
source = './data/ablation_math2_self_training.xlsx'
data = pd.read_excel(source, sheet_name=0)
# print(data)

y_fontsize = 27
y_title_fontsize = 27
legend_fontsize = 15
config = {
    'font.family': 'Times New Roman',
    'font.size': 28,
}
rcParams.update(config)

# Plot the responses for different events and regions
# plt.rc('font', family='Times New Roman', size=24)
plt.figure(figsize=(11, 8))
# sns.lineplot(x="completion_per_question(k)", y="acc", hue="Method", err_style="bars", style="Method", marker="*", linewidth=3, data=data)
sns.lineplot(x="completion_per_question(k)", y="acc", data=data, err_style="bars", hue="Method", style="Method", markers=True, linewidth=4, dashes=False, errorbar=('ci', 50), markersize=15)
plt.xlabel("Completion Tokens (Average Per Question)", fontsize=y_title_fontsize)
# plt.xticks(fontsize=18)
# plt.xticks([0, 2, 4, 6, 8], ['0', '10,000', '20,000', '30,000', '40,000'], fontsize=y_fontsize)
# plt.xticks([0, 2, 4, 6, 8], ['0', '10', '20', '30', '40'], fontsize=y_fontsize)
plt.xticks([0, 10000, 20000, 30000, 40000], ['0', '10,000', '20,000', '30,000', '40,000'], fontsize=y_fontsize)
plt.yticks(fontsize=y_fontsize)
plt.ylabel("Accuracy", fontsize=y_title_fontsize)
# plt.legend(fontsize=legend_fontsize)
handles, labels = plt.gca().get_legend_handles_labels()
order = [4,1,3,0,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=legend_fontsize)
plt.savefig('MATH2_completion_self_train.pdf')
# plt.show()

