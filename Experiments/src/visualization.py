import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from .datasets import load_and_clean_raw_dataset, DEFAULT_VAL_SPLIT_RATIO, DEFAULT_SPLIT_SEED


class DatasetEDAVisualizer:
    """
    Exploratory Data Analysis plots for a registered dataset, built once from
    the same cleaned data (and auto-generated validation split, where needed)
    that training will actually see. Each plot method returns its own
    standalone figure, so you can call only the ones you need instead of
    always rendering a combined multi-panel figure. Use `plot_all()` if you
    do want the combined view.
    """

    def __init__(
        self,
        dataset_name: str,
        max_seq_len: int = 64,
        val_split_ratio: float = DEFAULT_VAL_SPLIT_RATIO,
        seed: int = DEFAULT_SPLIT_SEED,
    ):
        self.dataset_name = dataset_name
        self.max_seq_len = max_seq_len

        ds, info = load_and_clean_raw_dataset(dataset_name, val_split_ratio, seed)
        self.info = info
        self.text_col = info['text_col']
        self.label_col = info['label_col']
        self.label_names = info['label_names']

        train_df = ds['train'].to_pandas()
        val_df = ds['validation'].to_pandas()
        test_df = ds['test'].to_pandas()
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        if self.label_names:
            train_df['label_name'] = train_df[self.label_col].map(self.label_names)
            all_df['label_name'] = all_df[self.label_col].map(self.label_names)
        else:
            train_df['label_name'] = train_df[self.label_col].astype(str)
            all_df['label_name'] = all_df[self.label_col].astype(str)
        all_df['word_count'] = all_df[self.text_col].apply(lambda x: len(str(x).split()))

        self.train_df = train_df
        self.all_df = all_df
        self.num_classes = train_df['label_name'].nunique()
        self.many_classes = self.num_classes >= 15  # e.g. banking77 (77), 20_newsgroups (20) vs emotion/ag_news (<=6)

    def class_distribution(self, ax=None, save_path=None):
        """Class distribution (train split) — horizontal bars when there are too many classes for x-tick labels."""
        standalone = ax is None
        if standalone:
            figsize = (7, max(5, self.num_classes * 0.18)) if self.many_classes else (7, 5)
            fig, ax = plt.subplots(figsize=figsize)

        counts = self.train_df['label_name'].value_counts()
        colors = sns.color_palette('muted', len(counts))
        if self.many_classes:
            counts = counts.sort_values(ascending=True)
            ax.barh(counts.index, counts.values, color=colors)
            ax.set_xlabel('Sample Count')
            ax.tick_params(axis='y', labelsize=7)
        else:
            ax.bar(counts.index, counts.values, color=colors)
            ax.set_xlabel('Class')
            ax.set_ylabel('Sample Count')
            ax.tick_params(axis='x', rotation=30)
            for i, (_, count) in enumerate(counts.items()):
                ax.text(i, count + (counts.max() * 0.01), str(count), ha='center', fontsize=9)
        ax.set_title('Class Distribution (Train)', fontsize=12)

        return self._finish(ax, standalone, save_path)

    def text_length_distribution(self, ax=None, save_path=None):
        """Word-count histogram across all splits, with MAX_SEQ_LEN coverage annotated."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(self.all_df['word_count'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(x=self.max_seq_len, color='red', linestyle='--', linewidth=1.5, label=f'max_len={self.max_seq_len}')
        pct_covered = (self.all_df['word_count'] <= self.max_seq_len).mean() * 100
        ax.set_title(f'Text Length Distribution\n({pct_covered:.1f}% covered by max_len={self.max_seq_len})', fontsize=11)
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Frequency')
        ax.legend()

        return self._finish(ax, standalone, save_path)
    

    def avg_length_by_class(self, ax=None, save_path=None):
        """Average text length per class, across all splits."""
        standalone = ax is None
        if standalone:
            figsize = (7, max(5, self.num_classes * 0.18)) if self.many_classes else (7, 5)
            fig, ax = plt.subplots(figsize=figsize)

        avg_len = self.all_df.groupby('label_name')['word_count'].mean().sort_values(ascending=False)
        ax.barh(avg_len.index, avg_len.values, color=sns.color_palette('muted', len(avg_len)))
        ax.set_title('Average Text Length by Class', fontsize=12)
        ax.set_xlabel('Average Word Count')
        ax.tick_params(axis='y', labelsize=7 if self.many_classes else 10)

        return self._finish(ax, standalone, save_path)

    def plot_all(self, save_path=None):
        """Combined 3-panel figure (class distribution, text length, avg length by class), e.g. for archiving."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5 if not self.many_classes else max(5, self.num_classes * 0.18)))
        fig.suptitle(f"{self.info['hf_path']} — Dataset Exploratory Analysis", fontsize=13, fontweight='bold')

        self.class_distribution(ax=axes[0])
        self.text_length_distribution(ax=axes[1])
        self.avg_length_by_class(ax=axes[2])

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved: {save_path}')
        return fig

    @staticmethod
    def _finish(ax, standalone, save_path):
        if not standalone:
            return ax
        fig = ax.get_figure()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved: {save_path}')
        return fig


class TrainingResultsVisualizer:
    """
    Evaluation plots for a single fine-tuning run. Each plot method returns
    its own standalone figure, so you can call only the ones you need
    (e.g. just the confusion matrix) instead of always rendering a congested
    combined figure. Use `plot_all()` if you do want the combined view.
    """

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        seed: int,
        epoch_logs: list,
        labels,
        preds,
        label_names: dict,
        num_labels: int,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed = seed
        self.epoch_logs = epoch_logs
        self.labels = labels
        self.preds = preds
        self.label_names = label_names
        self.num_labels = num_labels
        self.many_classes = num_labels > 20  # confusion matrix annotations get unreadable past this

    def learning_curves(self, ax=None, save_path=None):
        """Per-epoch macro F1 + accuracy on the validation set, with the best epoch marked."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 5))

        epochs = [e['epoch'] for e in self.epoch_logs]
        f1_scores = [e['eval_macro_f1'] for e in self.epoch_logs]
        acc_scores = [e['eval_accuracy'] for e in self.epoch_logs]

        ax.plot(epochs, f1_scores, 'b-o', linewidth=2, markersize=6, label='Macro F1')
        ax.plot(epochs, acc_scores, 'g--s', linewidth=2, markersize=6, label='Accuracy')
        ax.set_title('Learning Curves (Validation)', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_xticks(epochs)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        best_epoch = f1_scores.index(max(f1_scores)) + 1
        ax.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.7, label=f'Best @ epoch {best_epoch}')
        ax.legend()

        return self._finish(ax, standalone, save_path)

    def confusion_matrix(self, ax=None, save_path=None):
        """Confusion matrix on the test set (annotations/labels auto-scaled down for high class counts)."""
        standalone = ax is None
        if standalone:
            size = max(6, self.num_labels * 0.15) if self.many_classes else 6
            fig, ax = plt.subplots(figsize=(size, size))

        cm = confusion_matrix(self.labels, self.preds)
        label_names_list = (
            [self.label_names[i] for i in range(self.num_labels)]
            if (self.label_names and len(self.label_names) == self.num_labels)
            else [str(i) for i in range(self.num_labels)]
        )
        sns.heatmap(
            cm, annot=not self.many_classes, fmt='d', cmap='Blues',
            xticklabels=label_names_list, yticklabels=label_names_list,
            cbar=True, ax=ax
        )
        ax.set_title('Confusion Matrix (Test Set)', fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=90 if self.many_classes else 30, labelsize=6 if self.many_classes else 10)
        ax.tick_params(axis='y', rotation=0, labelsize=6 if self.many_classes else 10)

        return self._finish(ax, standalone, save_path)

    def training_cost(self, ax=None, save_path=None):
        """Per-epoch wall time and peak VRAM."""
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(7, 5))

        epochs = [e['epoch'] for e in self.epoch_logs]
        epoch_times = [e['epoch_time_s'] for e in self.epoch_logs]
        vrams = [e['peak_vram_mb'] for e in self.epoch_logs]

        ax_vram = ax.twinx()
        bars = ax.bar(epochs, epoch_times, color='steelblue', alpha=0.7, label='Epoch time (s)')
        line = ax_vram.plot(epochs, vrams, 'r-o', linewidth=2, markersize=6, label='Peak VRAM (MB)')
        ax.set_title('Training Cost per Epoch', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)', color='steelblue')
        ax_vram.set_ylabel('Peak VRAM (MB)', color='red')
        ax.set_xticks(epochs)
        ax.legend([bars, line[0]], ['Epoch time (s)', 'Peak VRAM (MB)'], loc='upper right')

        return self._finish(ax, standalone, save_path)

    def plot_all(self, save_path=None):
        """Combined 3-panel figure (learning curves, confusion matrix, training cost), e.g. for archiving."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6 if not self.many_classes else max(6, self.num_labels * 0.15)))
        fig.suptitle(f'{self.model_name} — Results on {self.dataset_name} | Seed {self.seed}', fontsize=13, fontweight='bold')

        self.learning_curves(ax=axes[0])
        self.confusion_matrix(ax=axes[1])
        self.training_cost(ax=axes[2])

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved: {save_path}')
        return fig

    @staticmethod
    def _finish(ax, standalone, save_path):
        if not standalone:
            return ax
        fig = ax.get_figure()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved: {save_path}')
        return fig


def plot_model_comparison(results_df: pd.DataFrame, metric_cols: list = None):
    """
    Bar-chart comparison across multiple saved experiment results (one row per
    model/dataset/seed run, as aggregated by analysis/generate_tables.ipynb).
    Defaults to comparing macro F1, accuracy, energy per 1k queries, and
    latency (ms/sample) side by side, faceted by dataset.
    """
    if metric_cols is None:
        metric_cols = [
            ('final_macro_f1', 'Macro F1'),
            ('final_accuracy', 'Accuracy'),
            ('energy_wh_per_1k_queries', 'Energy / 1k queries (Wh)'),
            ('ms_per_sample', 'Latency (ms/sample)'),
        ]
    metric_cols = [(c, c) if isinstance(c, str) else c for c in metric_cols]

    datasets_present = sorted(results_df['dataset'].unique())
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(6 * len(metric_cols), 5))
    if len(metric_cols) == 1:
        axes = [axes]
    fig.suptitle('Model Comparison Across Datasets', fontsize=13, fontweight='bold')

    for ax, (col, title) in zip(axes, metric_cols):
        pivot = results_df.pivot_table(index='model', columns='dataset', values=col, aggfunc='mean')
        pivot = pivot.reindex(columns=datasets_present)
        pivot.plot(kind='bar', ax=ax, color=sns.color_palette('muted', len(datasets_present)))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Model')
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title='Dataset', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig
