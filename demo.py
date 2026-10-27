"""Small web demo: pick a database, ask a question, see the SQL and result.

  python demo.py --model models/sft_s1/gptq4
"""

import argparse
from pathlib import Path

import gradio as gr
import pandas as pd
from vllm import LLM, SamplingParams

from data import load
from execution import run_sql
from gpu import has_bf16
from prompts import REFUSE, build_messages, extract_sql
from safety import is_safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    args = p.parse_args()

    dbs = {r["db_id"]: r["db_path"] for r in load("spider_test")}
    bf16 = has_bf16()
    llm = LLM(model=args.model, dtype="bfloat16" if bf16 else "float16",
              max_model_len=4096, gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=300)

    def answer(db_id, question):
        row = {"db_path": dbs[db_id], "question": question}
        out = llm.chat([build_messages(row)], params, use_tqdm=False,
                       chat_template_kwargs={"enable_thinking": False})
        sql = extract_sql(out[0].outputs[0].text)
        if sql == REFUSE:
            return sql, None, "This database can't answer that question."
        if not is_safe(sql):
            return sql, None, "Blocked: only read-only queries can run."
        rows, err = run_sql(dbs[db_id], sql, timeout=10)
        if err:
            return sql, None, f"Query failed: {err}"
        return sql, pd.DataFrame(rows[:200]), f"{len(rows)} rows"

    with gr.Blocks(title="Text to SQL") as ui:
        gr.Markdown("## Ask a database a question")
        with gr.Row():
            db = gr.Dropdown(sorted(dbs), value="concert_singer",
                             label="Database")
            question = gr.Textbox(label="Question",
                                  placeholder="How many singers are there?")
        go = gr.Button("Run")
        sql = gr.Code(label="SQL", language="sql")
        status = gr.Markdown()
        table = gr.Dataframe(label="Result")
        go.click(answer, [db, question], [sql, table, status])

    ui.launch(share=True)


if __name__ == "__main__":
    main()
