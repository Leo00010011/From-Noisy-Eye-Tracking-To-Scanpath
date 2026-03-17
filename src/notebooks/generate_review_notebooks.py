from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_DIR = Path(__file__).resolve().parent


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text)


def code_cell(text: str):
    return nbf.v4.new_code_cell(text)


def build_recorder_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.x"}
    nb.cells = [
        markdown_cell(
            "# Inference Recorder Review\n"
            "This notebook is focused on visually checking what the inference recorder is saving.\n"
            "It is intentionally exploratory and tolerant to partial runs."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = Path.cwd().resolve().parents[1]\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n"
            "\n"
            "from src.notebooks.review_utils import (\n"
            "    PROJECT_ROOT,\n"
            "    discover_recorder_payloads,\n"
            "    list_recorder_files,\n"
            "    extract_decoder_deformable_attention,\n"
            "    load_recorder_payload,\n"
            "    resolve_image_bank_path,\n"
            "    payload_summary_frame,\n"
            "    activation_inventory_frame,\n"
            "    decoder_attention_summary_frame,\n"
            "    plot_scanpath_overview,\n"
            "    plot_attention_heatmap,\n"
            "    plot_sampling_locations,\n"
            "    plot_decoder_deformable_attention_overlay,\n"
            ")\n"
            "\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "pd.set_option('display.max_colwidth', 120)\n"
            "PROJECT_ROOT"
        ),
        code_cell(
            "all_pt_files = list_recorder_files()\n"
            "display(all_pt_files.head(20))\n"
            "\n"
            "recorder_files = discover_recorder_payloads(limit=50)\n"
            "display(recorder_files)\n"
            "\n"
            "if recorder_files.empty:\n"
            "    print('No inference-recorder payloads were found yet. Once a run produces *.pt payloads, re-run this notebook.')"
        ),
        code_cell(
            "selected_path = None\n"
            "if not recorder_files.empty:\n"
            "    selected_path = PROJECT_ROOT / recorder_files.iloc[0]['file']\n"
            "selected_path"
        ),
        markdown_cell(
            "## Decoder Deformable Attention Overlay\n"
            "This section is focused on `decoder.#.second_cross_attn` only.\n"
            "The origin is reconstructed from teacher-forced fixation coordinates, so query `0` starts at `(0.5, 0.5)` and later queries start at the previous ground-truth fixation.\n"
            "The `eye_decoder.#.cross_attn` branch is intentionally left out for this iteration."
        ),
        code_cell(
            "external_record_dir = Path(r'C:\\Users\\ulloa\\Miooooo\\Master\\thesis\\projectes\\From-Noisy-Eye-Tracking-To-Scanpath\\outputs\\outputs\\2026-03-16\\18-24-45\\inference_records')\n"
            "external_candidates = sorted(external_record_dir.glob('*.pt')) if external_record_dir.exists() else []\n"
            "payload_path = external_candidates[0] if external_candidates else selected_path\n"
            "sample_index = 0\n"
            "decoder_layer = 0\n"
            "query_index = 0\n"
            "aggregate_heads = True\n"
            "head_index = None\n"
            "image_bank_path = None\n"
            "\n"
            "payload_path"
        ),
        code_cell(
            "payload = None\n"
            "if payload_path is not None:\n"
            "    payload = load_recorder_payload(payload_path)\n"
            "    display(payload.get('metadata', {}))\n"
            "    display(payload_summary_frame(payload))\n"
            "    try:\n"
            "        resolved_image_bank_path = resolve_image_bank_path(payload_path=payload_path, image_bank_path=image_bank_path)\n"
            "        print('image_bank_path =', resolved_image_bank_path)\n"
            "    except FileNotFoundError as exc:\n"
            "        resolved_image_bank_path = None\n"
            "        print(exc)"
        ),
        code_cell(
            "decoder_attention_info = None\n"
            "if payload is not None:\n"
            "    decoder_attention_info = extract_decoder_deformable_attention(\n"
            "        payload,\n"
            "        sample_index=sample_index,\n"
            "        decoder_layer=decoder_layer,\n"
            "        query_index=query_index,\n"
            "        aggregate_heads=aggregate_heads,\n"
            "        head_index=head_index,\n"
            "    )\n"
            "    display({\n"
            "        'module_name': decoder_attention_info['module_name'],\n"
            "        'query_count': decoder_attention_info['query_count'],\n"
            "        'num_heads': decoder_attention_info['num_heads'],\n"
            "        'num_points': decoder_attention_info['num_points'],\n"
            "        'valid_fixation_count': decoder_attention_info['valid_fixation_count'],\n"
            "        'is_terminal_or_padded': decoder_attention_info['is_terminal_or_padded'],\n"
            "        'selected_head_index': decoder_attention_info['selected_head_index'],\n"
            "    })\n"
            "    display(decoder_attention_summary_frame(\n"
            "        payload,\n"
            "        sample_index=sample_index,\n"
            "        decoder_layer=decoder_layer,\n"
            "        query_index=query_index,\n"
            "        aggregate_heads=aggregate_heads,\n"
            "        head_index=head_index,\n"
            "    ))"
        ),
        code_cell(
            "if payload is not None:\n"
            "    plot_decoder_deformable_attention_overlay(\n"
            "        payload,\n"
            "        sample_index=sample_index,\n"
            "        decoder_layer=decoder_layer,\n"
            "        query_index=query_index,\n"
            "        aggregate_heads=aggregate_heads,\n"
            "        head_index=head_index,\n"
            "        payload_path=payload_path,\n"
            "        image_bank_path=image_bank_path,\n"
            "    )"
        ),
        code_cell(
            "if payload is not None:\n"
            "    display(activation_inventory_frame(payload).sort_values(['module', 'activation']).reset_index(drop=True))"
        ),
        code_cell(
            "if payload is not None:\n"
            "    plot_scanpath_overview(payload, sample_index=sample_index)"
        ),
        code_cell(
            "if payload is not None:\n"
            "    try:\n"
            "        plot_attention_heatmap(payload, sample_index=sample_index, head_index=0)\n"
            "    except ValueError as exc:\n"
            "        print(exc)"
        ),
        code_cell(
            "if payload is not None:\n"
            "    try:\n"
            "        plot_sampling_locations(payload, sample_index=sample_index, query_index=query_index)\n"
            "    except ValueError as exc:\n"
            "        print(exc)"
        ),
        markdown_cell(
            "## Notes\n"
            "- `data` comes from `InferenceRecorder.record_batch` and currently includes indices plus input/target tensors when present.\n"
            "- `outputs` includes `denoise`, `coord`, `dur`, `reg`, and `cls` under more descriptive names.\n"
            "- `activations` currently come from attention modules in `blocks.py`, mainly cross-attention weights and deformable-attention sampling tensors."
        ),
    ]
    return nb


def build_metrics_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.x"}
    nb.cells = [
        markdown_cell(
            "# Training Metrics Review\n"
            "This notebook groups the metrics saved by `MetricsStorage` and `validate` so we can quickly spot obvious issues."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "if not (PROJECT_ROOT / 'src').exists():\n"
            "    PROJECT_ROOT = Path.cwd().resolve().parents[1]\n"
            "if str(PROJECT_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_ROOT))\n"
            "\n"
            "from src.notebooks.review_utils import (\n"
            "    PROJECT_ROOT,\n"
            "    list_metric_files,\n"
            "    load_metrics,\n"
            "    metrics_to_frame,\n"
            "    plot_metric_groups,\n"
            "    plot_run_comparison,\n"
            ")\n"
            "\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "pd.set_option('display.max_colwidth', 120)\n"
            "PROJECT_ROOT"
        ),
        code_cell(
            "metric_files = list_metric_files()\n"
            "display(metric_files.head(20))"
        ),
        code_cell(
            "selected_metrics_path = None\n"
            "if not metric_files.empty:\n"
            "    selected_metrics_path = PROJECT_ROOT / metric_files.iloc[0]['metrics_file']\n"
            "selected_metrics_path"
        ),
        code_cell(
            "if selected_metrics_path is not None:\n"
            "    metrics = load_metrics(selected_metrics_path)\n"
            "    metric_frame = metrics_to_frame(metrics)\n"
            "    display(metric_frame.head(20))\n"
            "    display(metric_frame.groupby('metric')['value'].agg(['count', 'min', 'max', 'mean']).sort_index())"
        ),
        code_cell(
            "if selected_metrics_path is not None:\n"
            "    plot_metric_groups(metrics)"
        ),
        code_cell(
            "if selected_metrics_path is not None:\n"
            "    epoch_series = metrics.get('epoch', [])\n"
            "    for metric_name in ['reg_loss_train', 'reg_loss_val', 'cls_loss_train', 'cls_loss_val', 'reg_error_val', 'accuracy']:\n"
            "        if metric_name in metrics:\n"
            "            print(metric_name, 'points=', len(metrics[metric_name]), 'epochs=', epoch_series[:len(metrics[metric_name])])"
        ),
        code_cell(
            "comparison_candidates = [PROJECT_ROOT / path for path in metric_files.head(3)['metrics_file'].tolist()]\n"
            "if comparison_candidates:\n"
            "    plot_run_comparison(comparison_candidates, metric_name='reg_error_val')"
        ),
        markdown_cell(
            "## Notes\n"
            "- Validation metrics are appended only on validation epochs, so many training series are denser than validation series.\n"
            "- `outliers_count_val` is stored as a total count over the whole validation pass, while most others are averaged by batch.\n"
            "- This notebook is aimed at sanity-checking trends first, not at final reporting."
        ),
    ]
    return nb


def main() -> None:
    recorder_nb = build_recorder_notebook()
    metrics_nb = build_metrics_notebook()

    with open(NOTEBOOK_DIR / "review_inference_recorder.ipynb", "w", encoding="utf-8") as handle:
        nbf.write(recorder_nb, handle)
    with open(NOTEBOOK_DIR / "review_training_metrics.ipynb", "w", encoding="utf-8") as handle:
        nbf.write(metrics_nb, handle)


if __name__ == "__main__":
    main()
