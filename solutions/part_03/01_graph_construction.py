"""Step-by-step graph construction exercise.

This version keeps the important graph construction pieces directly in the
script, like the Part 2 RAG preprocessing script does for vector indexing.

The goal is to show the full data flow:
1. Read the source documents.
2. Define the graph schema.
3. Ask the model to extract structured facts.
4. Convert those facts into graph nodes and edges.
5. Save and visualize the graph.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from pdfminer.high_level import extract_text
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.utils.pydantic_utils import get_ollama_model
from graph_rag_workshop.utils.part_03_graph_construction_utils import (
    facts_to_graph,
    save_graph,
    visualize_graph,
)

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"

MY_DOCUMENTS = DATA_DIR / "my_documents"

SERVICE_AGREEMENT_PDF = MY_DOCUMENTS / "Service_Agreement.pdf"
ANNEX_PDF = MY_DOCUMENTS / "Annex_A_Scope_Sheet.pdf"
SOURCE_DOCUMENTS = [SERVICE_AGREEMENT_PDF, ANNEX_PDF]

GRAPH_JSON_PATH = DATA_DIR / "knowledge_graph.json"
GRAPH_HTML_PATH = DATA_DIR / "knowledge_graph.html"

#################################################################
# STEP 1 - Extract the text that will become graph facts
#################################################################

contract_texts = []

for pdf_path in SOURCE_DOCUMENTS:
    text = extract_text(str(pdf_path))
    contract_texts.append(text)
    print(f"✅ Extracted text from {pdf_path.name}")

contract_text = "\n\n------------\n\n".join(contract_texts)

#################################################################
# STEP 2 - Define the small graph schema
#################################################################

# SOLUTION - Graph schema:
# Each schema entry tells the model what kind of contract fact to extract.
SMALL_CONTRACT_SCHEMA = {
    "party": "A person or company with a role in the agreement.",
    "obligation": "Something a party must do.",
    "date": "A due date, event date, or notice date.",
}


class Party(BaseModel):
    role: str = Field(description="Contract role, for example Provider or Organizer.")
    name: str = Field(description="Company name.")


class Obligation(BaseModel):
    party: str = Field(description="The role that must do the action.")
    action: str = Field(description="What the party must do.")
    date: Optional[str] = Field(
        default=None,
        description="Due date or timing. Use null if no date is explicitly stated.",
    )
    source_quote: str = Field(description="Very short quote supporting this fact.")


class ContractGraphExtraction(BaseModel):
    parties: list[Party] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)


print("✅ Defined a small graph schema")

#################################################################
# STEP 3 - Extract structured facts with the language model
#################################################################

schema_text = "\n".join(
    f"- {name}: {description}" for name, description in SMALL_CONTRACT_SCHEMA.items()
)

INSTRUCTIONS = (
    "Extract only the core contract facts. Return valid JSON matching the "
    "requested output type. Do not explain your answer.\n"
    "Use this graph schema:\n"
    f"{schema_text}\n\n"
    "Work in three passes:\n"
    "1. Parties: there are only two parties. Use the roles Organizer and "
    "Provider. Extract their company names from the agreement.\n"
    "2. Obligations: extract only the most important obligations for those "
    "two roles.\n\n"
    "3. For each obligation, if present, include the related date. Confirm that the date is accurate.\n"
    "Return exactly this JSON shape:\n"
    "{\n"
    '  "parties": [\n'
    '    {"role": "Organizer", "name": "..."},\n'
    '    {"role": "Provider", "name": "..."}\n'
    "  ],\n"
    '  "obligations": [\n'
    '    {"party": "Organizer or Provider", "action": "...", "date": "...", "source_quote": "..."}\n'
    "  ]\n"
    "}\n\n"
    "Only use facts explicitly stated in the text. "
    "Every obligation must include party, action, date, and source_quote. "
    "Use null for date when no date is stated.\n"
    "Use a short exact quote for source_quote. Do not invent facts."
)

# SOLUTION - Deterministic extraction:
# Temperature 0 keeps the structured extraction as stable as possible.
extraction_agent = Agent(
    model=get_ollama_model(),
    output_type=ContractGraphExtraction,
    instructions=INSTRUCTIONS,
    model_settings=ModelSettings(
        thinking="minimal",
        temperature=0.0,
        max_tokens=700,
    ),
)

start = time.time()
extracted_facts = extraction_agent.run_sync(contract_text).output
end = time.time()

print(f"✅ Extracted structured facts in {end - start:.1f} seconds")
print(f"OK - Extracted {len(extracted_facts.parties)} parties")
print(f"OK - Extracted {len(extracted_facts.obligations)} obligations")

#################################################################
# STEP 4 - Convert extracted facts into graph nodes and edges
#################################################################

# Convert facts to graph nodes and edges using the shared helper.
source_doc = " and ".join(pdf_path.name for pdf_path in SOURCE_DOCUMENTS)
graph = facts_to_graph(extracted_facts, source_doc)

print("✅ Converted extracted facts into a graph data structure")
print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

print("\nGraph facts preview:")
for edge in graph.edges[:8]:
    source = graph.nodes[edge.source].label
    target = graph.nodes[edge.target].label
    print(f"- {source} --{edge.relation}--> {target}")
    print(f"  Evidence: {edge.evidence}")

#################################################################
# STEP 5 - Save and visualize the constructed graph
#################################################################

graph_path = save_graph(graph, GRAPH_JSON_PATH)
output_path = visualize_graph(graph, GRAPH_HTML_PATH, open_browser=True)

print(f"✅ Graph saved to: {output_path}")
