"""Prompt templates and builders for the Direct pipeline.

Covers query rephrasing, label and relationship selection, Cypher generation,
property and synonym suggestion, query and Neo4j-error correction, and the
answer-synthesis system prompts, plus helpers that render the graph schema as
prompt text.
"""
from nexa.config import params
from nexa.utils.utils import parse_key
from collections import defaultdict

def format_nodes(nodes, descriptions_dict=None, with_properties=True):
    """Render node labels as a bullet list, with descriptions and optionally properties.

    ``nodes`` holds either schema entries (``{"node": {"labels", "properties"}}``)
    or plain labels.
    """
    formatted = []
    is_detailed_format = all(isinstance(node, dict) and 'node' in node and 'labels' in node['node'] for node in nodes)

    for node_data in nodes:
        label_for_display: str
        label_for_lookup: str
        current_properties = []

        if is_detailed_format:
            raw_labels = node_data['node']['labels']
            if isinstance(raw_labels, list):
                label_for_lookup = raw_labels[0] if raw_labels else "UnknownLabel"
                label_for_display = ", ".join(raw_labels)
            else:
                label_for_lookup = str(raw_labels)
                label_for_display = str(raw_labels)
            current_properties = node_data['node'].get('properties', [])
        else:
            label_for_lookup = str(node_data)
            label_for_display = str(node_data)

        description_text = ""
        if descriptions_dict:
            description = descriptions_dict.get(label_for_lookup)
            if description:
                description_text = f": {description}"

        if with_properties and current_properties:
            formatted.append(f"- {label_for_display}{description_text}\n  Properties: {', '.join(current_properties)}")
        else:
            formatted.append(f"- {label_for_display}{description_text}")

    return "\n".join(formatted)


def format_relationships(relationships, descriptions_dict=None, relationship_properties=None):
    """Render relationships as Cypher patterns grouped by type, ready to copy into a MATCH.

    Example output:

        - (:Drug)-[:DRUGTREATSDISEASE]->(:Disease): drug treats disease
          Properties: source, score
        - (:Drug)-[:CHEMICALBINDSGENE]->(:Gene)
          (:Compound)-[:CHEMICALBINDSGENE]->(:Gene)
          Properties: source

    ``relationship_properties`` is the output of
    ``Neo4jHandler.get_relationships_properties``.
    """
    grouped_relationships = defaultdict(lambda: defaultdict(list))
    for rel_data in relationships:
        rel_type = rel_data['relationship']
        source_node = rel_data['source']
        target_nodes = rel_data['target'] if isinstance(rel_data['target'], list) else [rel_data['target']]
        for target_node in target_nodes:
            grouped_relationships[rel_type][source_node].append(target_node)

    properties_by_type = {
        prop['relationship']['type']: prop['relationship']['properties']
        for prop in (relationship_properties or [])
    }

    formatted_output = []
    for rel_type, sources in grouped_relationships.items():
        patterns = []
        for source_node, target_nodes_list in sources.items():
            for target_node in sorted(set(target_nodes_list)):
                patterns.append(f"(:{source_node})-[:{rel_type}]->(:{target_node})")

        description_text = ""
        if descriptions_dict:
            description = descriptions_dict.get(rel_type)
            if description:
                description_text = f": {description}"

        formatted_output.append(f"- {patterns[0]}{description_text}")
        for extra in patterns[1:]:
            formatted_output.append(f"  {extra}")

        props = properties_by_type.get(rel_type, [])
        if props:
            formatted_output.append(f"  Properties: {', '.join(props)}")

        formatted_output.append("")

    return "\n".join(formatted_output).strip()


def format_identifying_properties(identifying_properties):
    """Render the identifying-property mapping as a prompt block, or "" if there is none."""
    if not identifying_properties:
        return ""
    lines = []
    for label in sorted(identifying_properties):
        prop = identifying_properties.get(label)
        if not prop:
            continue
        lines.append(f"- {label} → {prop}")
    if not lines:
        return ""
    return (
        "**Identifying properties (use these for {prop: value} filters and WHERE clauses):**\n"
        + "\n".join(lines)
    )


def format_suggested_properties(suggested_properties):
    """Render the selected replacement values as numbered sentences."""
    if not suggested_properties:
        return ""

    formatted_properties = []
    for idx, (key, value) in enumerate(suggested_properties.items(), start=1):
        element_type, label, property_key, current_value = parse_key(key)
        suggested_values = value.get('suggested_values', [])

        if not suggested_values:
            continue

        last_suggestion = suggested_values[-1]
        selected_value = last_suggestion.get('selectionned_value')

        if not selected_value or selected_value == 'None':
            continue

        formatted_properties.append(
            f"{idx}. For the property '{property_key}' in the element '{label}':\n"
            f"   - The user provided the value '{current_value}'.\n"
            f"   - A suggested replacement is '{selected_value}', which has been found in the graph schema."
        )
    return "\n".join(formatted_properties)


############################
# REPHRASE USER QUERY PROMPTS
############################

REPHRASE_USER_QUERY_SYSTEM_PROMPT = """
You are an expert in biomedical knowledge graphs. Your goal is to rephrase ambiguous or informal user questions into precise, complete, and unambiguous biomedical queries in natural language.

- Use technical biomedical terminology when appropriate.
- Avoid vague words like "thing", "stuff", "info".
- Make implicit concepts explicit (e.g., "cancer drug" → "drug used in cancer treatment").
- Identify the type of entities involved (e.g., disease, gene, drug).
- Clarify relationships if possible.
- If the user query is already precise, complete, and unambiguous, return it as is.
"""

REPHRASE_USER_QUERY_USER_PROMPT = """
**Available Nodes:**
{nodes}

**Available Relationships:**
{relationships}

**User Query to Rephrase:**
{user_query}

Please rephrase this query following the guidelines provided.
"""

def get_rephrase_user_query_prompt(user_query, nodes, relationships, nodes_descriptions_dict=None, relationships_descriptions_dict=None):
    """Return the (system, user) prompts for rephrasing a user query."""
    formatted_nodes = format_nodes(nodes, nodes_descriptions_dict)
    formatted_relationships = format_relationships(relationships, relationships_descriptions_dict)

    user_prompt = REPHRASE_USER_QUERY_USER_PROMPT.format(
        nodes=formatted_nodes,
        relationships=formatted_relationships,
        user_query=user_query
    )

    return REPHRASE_USER_QUERY_SYSTEM_PROMPT, user_prompt


############################
# LABELS SELECTION PROMPTS
############################

DEFAULT_LABELS_SELECTION_EXAMPLES = """Examples — the labels below are illustrative. ONLY emit labels that appear in the **Available Nodes Labels** list above; do not invent labels.

Example 1 — single-hop, two entity types:
Available Nodes Labels: ["Gene", "Disease", "Drug", "SideEffect"]
User query: "What drugs are used to treat <Disease_X>?"
→ Output Selected Labels: ["Drug", "Disease"]

Example 2 — query targets a single entity but its annotation:
Available Nodes Labels: ["Gene", "Protein", "Drug", "Disease"]
User query: "What is the function of <Gene_Y>?"
→ Output Selected Labels: ["Gene", "Protein"]

Example 3 — multi-hop with several entity types in play:
Available Nodes Labels: ["Gene", "Protein", "Drug", "Disease", "SideEffect", "BodyPart", "Compound", "Pathway", "Anatomy", "Symptom", "MolecularFunction", "BiologicalProcess"]
User query: "Which drugs that treat <Disease_Z> target proteins expressed by genes under-expressed in <BodyPart_K>?"
→ Output Selected Labels: ["Drug", "Disease", "Protein", "Gene", "BodyPart"]

Example 4 — negation does NOT change which labels are involved:
Available Nodes Labels: ["Gene", "Disease", "Drug"]
User query: "<Drug_A> does NOT treat <Disease_B>"
→ Output Selected Labels: ["Drug", "Disease"]"""


LABELS_SELECTION_SYSTEM_PROMPT_TEMPLATE = """
You are a biomedical knowledge graph expert.

Given a user query and a list of available node labels in the graph, your goal is to identify the most relevant labels involved in the query.

Instructions:
- Only select nodes labels from the provided list.
- A query may involve more than one nodes label (e.g., both `Gene` and `Disease`).
- If no relevant nodes label is found, output: `None`.
- Think about what kind of biomedical entity each word in the query might refer to.
- Prefer the most specific label when confident; include multiple labels when the query is genuinely ambiguous between them.

{examples}
"""

LABELS_SELECTION_USER_PROMPT = """
**Available Nodes Labels:**
{nodes}

**User Query:**
{user_query}

Identify the relevant nodes labels involved in this query.
"""

def _examples_block(examples, default):
    """Return the profile's examples, else ``default``; "" when NEXA_ABLATE_FEWSHOT_EXAMPLES is set."""
    if params.NEXA_ABLATE_FEWSHOT_EXAMPLES:
        return ""
    return examples if examples else default


def get_labels_selection_prompt(user_query, nodes, nodes_descriptions_dict=None, examples=None):
    """Return the (system, user) prompts for selecting the node labels relevant to a query."""
    formatted_nodes = format_nodes(nodes, nodes_descriptions_dict, with_properties=False)

    examples_block = _examples_block(examples, DEFAULT_LABELS_SELECTION_EXAMPLES)
    system_prompt = LABELS_SELECTION_SYSTEM_PROMPT_TEMPLATE.format(examples=examples_block)

    user_prompt = LABELS_SELECTION_USER_PROMPT.format(
        nodes=formatted_nodes,
        user_query=user_query
    )

    return system_prompt, user_prompt


############################
# RELATIONSHIPS SELECTION PROMPTS
############################

DEFAULT_RELATIONSHIPS_SELECTION_EXAMPLES = """Examples — the names in angle brackets are PLACEHOLDERS for whatever this graph actually calls things. Substitute the real types from the **Available Relationships** list above. ONLY emit names that appear in that list; never invent one, and never emit a placeholder literally.

Suppose the list contained: ["<DRUG_TREATS_DISEASE_REL>", "<DRUG_CAUSES_EFFECT_REL>", "<GENE_CAUSES_DISEASE_REL>", "<GENE_INTERACTS_GENE_REL>", "<GENE_ASSOCIATES_DISEASE_REL>"]

Example 1 — one relationship, because the verb maps onto a single type:
User query: "What drugs are used to treat <Disease_X>?"
→ Output Selected Relationships: ["<DRUG_TREATS_DISEASE_REL>"]

Example 2 — ambiguous question: include every relationship whose semantics could match:
User query: "What diseases are affected by this drug?"
→ Output Selected Relationships: ["<DRUG_TREATS_DISEASE_REL>", "<DRUG_CAUSES_EFFECT_REL>"]

Example 3 — negation does NOT change which relationships are involved:
User query: "<Drug_A> does NOT treat <Disease_B>"
→ Output Selected Relationships: ["<DRUG_TREATS_DISEASE_REL>"]"""


RELATIONSHIPS_SELECTION_SYSTEM_PROMPT_TEMPLATE = """
You are a biomedical knowledge graph expert.

Your task is to select **all relevant relationships** from a biomedical knowledge graph that could correspond to a user query.

Instructions:
- Only select from the list of available relationships.
- A query can involve multiple relationships if their meaning overlaps or if the query is ambiguous.
- If no relationships match, return: `None`.
- Favor specificity when possible. If both a general and a specific relationship match, include both **only if** the query is ambiguous.
- A verb in the question rarely matches a relationship name literally: map it by MEANING onto a relationship that is present in the list (e.g. a question about treating a disease may map to a type named after the indication rather than after "treats"). Never emit a name you have not read in the **Available Relationships** list, even if it is the obvious English word for the concept.

{examples}
"""

RELATIONSHIPS_SELECTION_USER_PROMPT = """
**Available Relationships:**
{relationships}

**Available Nodes:**
{nodes}

**User Query:**
{user_query}

Identify the relationships involved in this query, choosing from the list above.
"""

def get_relationships_selection_prompt(user_query, filtered_relationships, selected_nodes, nodes_descriptions_dict=None, relationships_descriptions_dict=None, examples=None):
    """Return the (system, user) prompts for selecting the relationships relevant to a query."""
    formatted_relationships = format_relationships(filtered_relationships, relationships_descriptions_dict)
    formatted_nodes = format_nodes(selected_nodes, nodes_descriptions_dict, with_properties=False)

    examples_block = _examples_block(examples, DEFAULT_RELATIONSHIPS_SELECTION_EXAMPLES)
    system_prompt = RELATIONSHIPS_SELECTION_SYSTEM_PROMPT_TEMPLATE.format(examples=examples_block)

    user_prompt = RELATIONSHIPS_SELECTION_USER_PROMPT.format(
        relationships=formatted_relationships,
        nodes=formatted_nodes,
        user_query=user_query
    )

    return system_prompt, user_prompt


############################
# QUERY GENERATION PROMPTS
############################

DEFAULT_QUERY_GENERATION_EXAMPLES = """Examples — illustrative only. Replace `<LABEL>`, `<REL_TYPE>`, and property names with the actual labels, relationship types, and properties from the schema and the identifying-properties block.

Example 1 — single-hop filtered by an entity literal:
User Query: "What drugs treat <Disease_X>?"
→ Cypher: MATCH (dr:<DRUG_LABEL>)-[rel:<DRUG_TREATS_DISEASE_REL>]->(d:<DISEASE_LABEL> {<disease_name_prop>: "Disease_X"}) RETURN dr, rel, d

Example 2 — two-hop chain (a → b → c, where b is shared):
User Query: "Which genes interact with <Gene_Y> and are associated with <Disease_Z>?"
→ Cypher: MATCH (g1:<GENE_LABEL> {<gene_name_prop>: "Gene_Y"})-[rel1:<GENE_INTERACTS_GENE_REL>]-(g2:<GENE_LABEL>), (g2)-[rel2:<GENE_ASSOCIATES_DISEASE_REL>]->(d:<DISEASE_LABEL> {<disease_name_prop>: "Disease_Z"}) RETURN g1, rel1, g2, rel2, d

Example 3 — UNION of two relationship types from the same source (use the `|` operator, NOT separate comma-MATCH which causes a Cartesian product):
User Query: "What diseases are affected by <Drug_A>?" (treated OR caused)
→ Cypher: MATCH (dr:<DRUG_LABEL> {<drug_name_prop>: "Drug_A"})-[rel:<DRUG_TREATS_DISEASE_REL>|<DRUG_CAUSES_DISEASE_REL>]->(d:<DISEASE_LABEL>) RETURN dr, rel, d

Example 4 — INTERSECTION (a node connected to BOTH b AND c — use multiple comma-separated patterns sharing the pivot variable):
User Query: "List the drugs that bind <Gene_M> as well as <Gene_N>"
→ Cypher: MATCH (dr:<DRUG_LABEL>)-[rel1:<CHEMICAL_BINDS_GENE_REL>]->(g1:<GENE_LABEL> {<gene_name_prop>: "Gene_M"}), (dr)-[rel2:<CHEMICAL_BINDS_GENE_REL>]->(g2:<GENE_LABEL> {<gene_name_prop>: "Gene_N"}) RETURN dr, rel1, g1, rel2, g2

Example 5 — filter on a list of literals on ONE node, then traverse:
User Query: "Among the genes <Gene_P>, <Gene_Q>, <Gene_R>, which are in pathway <Pathway_K>?"
→ Cypher: MATCH (g:<GENE_LABEL>)-[rel:<GENE_IN_PATHWAY_REL>]->(p:<PATHWAY_LABEL> {<pathway_name_prop>: "Pathway_K"}) WHERE g.<gene_name_prop> IN ["Gene_P", "Gene_Q", "Gene_R"] RETURN g, rel, p
"""


QUERY_GENERATION_SYSTEM_PROMPT_TEMPLATE = """
You are a biomedical knowledge graph expert and Cypher query generator.

Your task is to write a complete Cypher query that answers the given user question, based on the structure of the knowledge graph and the elements extracted from the question.

Guidelines:
- Only use labels and relationships from the provided lists.
- Use `MATCH` clauses to find nodes and relationships.
- Use `WHERE` clauses to apply filters based on entity properties.
- Always alias nodes (e.g., (d:Disease), (g:Gene)) and use consistent variable names.
- Return full nodes and relationships (e.g., RETURN d, g, r) instead of just their properties (e.g., d.name).
- Always return relationships (e.g., RETURN r) if they are used in the query.
- If a relationship is directional (e.g., `(:<SOURCE_LABEL>)-[:<REL_TYPE>]->(:<TARGET_LABEL>)`), respect the direction if known.
- If a relationship is bidirectional or you don't know the direction, use the `-` operator.
- If the query is ambiguous and could correspond to multiple plausible relationships, you may include several of them.
- If no specific node is to be filtered, use generic patterns.

**INTERSECTION VS UNION VS CHAIN VS FILTER:**
- **Intersection** (X connects to BOTH a AND b): repeat the same pivot variable across comma-separated patterns, e.g. `MATCH (x)-[]->(:A {{n:"a"}}), (x)-[]->(:B {{n:"b"}})`.
- **Chain** (a → b → c): patterns share intermediate variables, e.g. `MATCH (a)-[]->(b), (b)-[]->(c)`.
- **Union** (X linked via REL1 OR REL2): use the `|` operator on the relationship, e.g. `MATCH (x)-[:REL1|REL2]->(y)`. Do NOT write two separate comma-MATCH patterns for unions — that produces a Cartesian product.
- **Filter** (a candidate set provided by the user, e.g. "among these: A, B, C"): use `WHERE prop IN [...]`.

The `WHERE prop IN ["A", "B"]` pattern is ONLY for filter-style questions where the user explicitly enumerates a candidate set ("among the following: A, B, C", "from this list: ..."). It expresses UNION semantics ("connected to A OR B") and is WRONG for intersection queries — it will silently return too many results.

**HOW TO TELL A FILTER FROM AN INTERSECTION:**
Count how many of the named entities the answer must satisfy AT ONCE.
- The question offers entities to *choose among* (a candidate list, numbered options,
  "from this list") → FILTER → one pattern plus `WHERE prop IN [...]`.
- The question requires the answer to relate to *every* named entity simultaneously
  (phrasings such as "X that {{verb}} A as well as B", "X in common with A and B",
  "X that share {{relation}} with A and B", "X common to A and B") → INTERSECTION →
  one comma-separated pattern leg per named entity, all sharing the same pivot
  variable, e.g. `MATCH (x)-[r1:REL]->(:T {{prop:"A"}}), (x)-[r2:REL]->(:T {{prop:"B"}})
  RETURN x, r1, r2`.
Writing `IN [...]` for an intersection returns everything linked to A OR B, which is a
superset of the answer — the query looks successful and is wrong.

{examples}
"""

QUERY_GENERATION_USER_PROMPT = """
**Available Nodes Labels:**
{nodes}

**Available Relationships:**
{relationships}
{identifying_properties_block}
**User Query:**
{user_query}

Write a Cypher query using the selected labels and relationships.
"""

def get_query_generation_prompt(user_query, nodes, relationships, rel_properties, nodes_descriptions_dict=None, relationships_descriptions_dict=None, examples=None, identifying_properties=None):
    """Return the (system, user) prompts for generating a Cypher query."""
    formatted_nodes = format_nodes(nodes, nodes_descriptions_dict, with_properties=True)
    formatted_relationships = format_relationships(relationships, relationships_descriptions_dict, relationship_properties=rel_properties)

    examples_block = _examples_block(examples, DEFAULT_QUERY_GENERATION_EXAMPLES)
    system_prompt = QUERY_GENERATION_SYSTEM_PROMPT_TEMPLATE.format(examples=examples_block)

    identifying_block = format_identifying_properties(identifying_properties)
    if identifying_block:
        identifying_block = "\n" + identifying_block + "\n"

    user_prompt = QUERY_GENERATION_USER_PROMPT.format(
        nodes=formatted_nodes,
        relationships=formatted_relationships,
        identifying_properties_block=identifying_block,
        user_query=user_query
    )

    return system_prompt, user_prompt


############################
# PROPERTIES SUGGESTION PROMPTS
############################

PROPERTIES_SUGGESTION_SYSTEM_PROMPT = """
You are an expert in refining graph database queries.

Your task is to match a given input value (from the user query) with the most appropriate value from a list of possible values found in the graph database.

Your selection must reflect real-world semantic equivalence, not just surface similarity.

Guidelines:
1. Do NOT match based on partial string similarity alone — this leads to incorrect results.
2. Accept only exact matches, well-known synonyms, standard identifiers, or widely used alternative names.
3. If no option clearly refers to the same real-world entity, return `None` for the `suggested_property`.

Output only the selected value (copied exactly from the list), or `None` for the `suggested_property`.

Examples:

VALUE: "Alzheimer"
Available candidates: ["Alzheimer's disease", "Alzheimer type dementia", "Parkinson's disease"]
→ Suggested Property: "Alzheimer's disease"

VALUE: "Gleevec"
Available candidates: ["Imatinib", "Paracetamol", "Aspirin"]
→ Suggested Property: "Imatinib"

VALUE: "unknown protein X"
Available candidates: ["TP53", "BRCA1", "EGFR"]
→ Suggested Property: "None"
"""

PROPERTIES_SUGGESTION_USER_PROMPT = """
**VALUE**: {property_value}

Here is the list of possible values in the database for the {element_type} {property_name} of {label_or_rel}:
{top_k_matches}

Select the value FROM THE LIST that best matches {property_value}.
"""

def get_properties_suggestion_prompt(element_type, property_name, label_or_rel, property_value, top_k_matches):
    """Return the (system, user) prompts for matching a value to one of the graph candidates."""
    user_prompt = PROPERTIES_SUGGESTION_USER_PROMPT.format(
        element_type=element_type,
        property_name=property_name,
        label_or_rel=label_or_rel,
        property_value=property_value,
        top_k_matches=top_k_matches
    )

    return PROPERTIES_SUGGESTION_SYSTEM_PROMPT, user_prompt


############################
# SYNONYM SUGGESTION PROMPTS
############################

SYNONYM_SUGGESTION_SYSTEM_PROMPT = """
You are an expert in refining graph database queries.

A Cypher query contains a value that does not match any known value in the database. Your task is to help correct it by suggesting either:

1. A better synonym or alternate string (not in the existing synonym list) that refers to the same real-world entity as the user's value;
2. A new property name (not currently used) that would likely contain this kind of value, based on the context and label.

Instructions:
- Only propose a synonym if you're confident it refers to the same entity.
- Only propose a new property name if the user value looks like a symbol, or alternate naming scheme.
- Never repeat existing synonyms.

Examples:

Property Name: "name"
Label: "Drug"
Value: "Gleevec"
Existing Synonyms: ["Gleevec"]
→ Output: "Imatinib"
"""

SYNONYM_SUGGESTION_USER_PROMPT = """
We could not find any matching property in the database for the following input:

**Property Name**: {property_name}
**Label or Relationship**: {label_or_rel}
**Value**: {property_value}
**Existing Synonyms**: {existing_synonyms}

Please suggest one of the following:
- A new synonym for the value (not already listed), or
- A more appropriate property name for this type of value.
"""

def get_synonym_suggestion_prompt(property_name, label_or_rel, property_value, existing_synonyms):
    """Return the (system, user) prompts for proposing a synonym for an unmatched value."""
    user_prompt = SYNONYM_SUGGESTION_USER_PROMPT.format(
        property_name=property_name,
        label_or_rel=label_or_rel,
        property_value=property_value,
        existing_synonyms=", ".join(existing_synonyms)
    )
    return SYNONYM_SUGGESTION_SYSTEM_PROMPT, user_prompt


############################
# QUERY CORRECTION PROMPTS
############################
DEFAULT_QUERY_CORRECTION_EXAMPLES = """Examples — illustrative only. Use the actual labels, relationships, and properties from the provided schema.

Example 1 — fix a property literal using the suggested synonym mapping:
Initial Failed Query: `MATCH (dr:<DRUG_LABEL>)-[rel:<DRUG_TREATS_DISEASE_REL>]->(d:<DISEASE_LABEL> {<disease_name_prop>: "Alzheimer"}) RETURN dr, rel, d`
Synonym Mapping: "Alzheimer" → "Alzheimer's disease"
→ Output: `MATCH (dr:<DRUG_LABEL>)-[rel:<DRUG_TREATS_DISEASE_REL>]->(d:<DISEASE_LABEL> {<disease_name_prop>: "Alzheimer's disease"}) RETURN dr, rel, d`

Example 2 — drop the directional arrow when direction is uncertain:
Initial Failed Query: `MATCH (g1:<GENE_LABEL> {<gene_name_prop>: "Gene_X"})-[rel:<GENE_INTERACTS_GENE_REL>]->(g2:<GENE_LABEL>) RETURN g1, rel, g2`
→ Output: `MATCH (g1:<GENE_LABEL> {<gene_name_prop>: "Gene_X"})-[rel:<GENE_INTERACTS_GENE_REL>]-(g2:<GENE_LABEL>) RETURN g1, rel, g2`

Example 3 — turn a `WHERE … IN […]` union into a true intersection over a pivot:
Initial Failed Query: `MATCH (dr:<DRUG_LABEL>)-[rel:<CHEMICAL_BINDS_GENE_REL>]->(g:<GENE_LABEL>) WHERE g.<gene_name_prop> IN ["Gene_M", "Gene_N"] RETURN dr, g, rel`
User Question: "List the drugs that bind Gene_M as well as Gene_N" (intersection)
→ Output: `MATCH (dr:<DRUG_LABEL>)-[rel1:<CHEMICAL_BINDS_GENE_REL>]->(g1:<GENE_LABEL> {<gene_name_prop>: "Gene_M"}), (dr)-[rel2:<CHEMICAL_BINDS_GENE_REL>]->(g2:<GENE_LABEL> {<gene_name_prop>: "Gene_N"}) RETURN dr, rel1, g1, rel2, g2`

Example 4 — fix invalid syntax `WHERE … IN […]-[r]->(…)`:
Initial Failed Query: `MATCH (d:<DRUG_LABEL>) WHERE d.<drug_name_prop> IN ["Drug_A", "Drug_B"]-[r:<DRUG_TREATS_DISEASE_REL>]->(dis:<DISEASE_LABEL>) RETURN d, r, dis`
→ Output: `MATCH (d:<DRUG_LABEL>)-[r:<DRUG_TREATS_DISEASE_REL>]->(dis:<DISEASE_LABEL>) WHERE d.<drug_name_prop> IN ["Drug_A", "Drug_B"] RETURN d, r, dis`"""


QUERY_CORRECTION_SYSTEM_PROMPT_TEMPLATE = """
You are an expert in biomedical knowledge graphs and Cypher query generation.

Your task is to correct a Cypher query that is incomplete, invalid, or mismatched with the knowledge graph. Use the structured information below to fix the query while preserving its original semantic intent.

You may:
- Replace property values with corrected synonyms.
- Replace property names with more appropriate ones.
- Fix missing or mismatched labels and relationships.
- Reconstruct the query if needed, based on the user's intent and available elements.

Guidelines:
- Only use node labels, relationships, and properties from the provided lists.
- When replacing values, only use values suggested by the synonym suggestion step.
- The corrected query must be syntactically valid and executable in Neo4j.
- Always return full nodes and relationships (e.g., RETURN d, dr, r) instead of isolated properties (e.g., d.name).
- Many biomedical relationships are semantically symmetric even when the KG
  stores them in only one direction — for instance, interactions ("X interacts
  with Y"), bindings ("X binds Y"), pairings between same-type entities
  (gene–gene, disease–disease, protein–protein), or shared participation
  ("shared between X and Y", "in common with"). If the query returned no
  results and the user question uses such symmetric language, the relationship
  is likely stored in the opposite direction. Try reversing the arrow
  (`<-[r:REL]-`) or using undirected matching (`-[r:REL]-`). For relationships
  that are clearly directional (treats, causes, regulates, increases/decreases
  expression), keep the original direction.
- Ensure all relevant entities and edges used in the MATCH clause are included in the RETURN clause to support downstream interpretability.

{examples}
"""

QUERY_CORRECTION_USER_PROMPT = """
**User Question:**
{question}

**Database Schema Context:**
- **Available Node Labels:**
{nodes}

- **Available Relationship Types:**
{relationships}
{identifying_properties_block}{suggested_properties_section}{notifications_section}
**Current (Possibly Incorrect) Cypher Query:**
{current_cypher_query}

Please revise this Cypher query to accurately answer the user's question, using only the available labels, relationships, and (if provided) the suggested property corrections. Ensure the corrected query is syntactically valid and uses semantically correct labels and properties.

"""


def get_query_correction_prompt(question, nodes, relationships, notifications, cypher_query, parsed_properties, nodes_descriptions_dict=None, relationships_descriptions_dict=None, examples=None, identifying_properties=None):
    """Return the (system, user) prompts for correcting a Cypher query."""
    formatted_nodes = format_nodes(nodes, nodes_descriptions_dict, with_properties=True)
    formatted_relationships = format_relationships(relationships, relationships_descriptions_dict)

    formatted_suggested_properties = format_suggested_properties(parsed_properties)

    suggested_properties_section = ""
    if formatted_suggested_properties:
        suggested_properties_section = f"\n**Suggested Properties**:\n{formatted_suggested_properties}\n"

    notifications_section = ""
    if notifications:
        notifications_section = f"\n**Neo4j Notifications**:\n{notifications}\n"

    identifying_block = format_identifying_properties(identifying_properties)
    identifying_properties_block = f"\n{identifying_block}\n" if identifying_block else ""

    examples_block = _examples_block(examples, DEFAULT_QUERY_CORRECTION_EXAMPLES)
    system_prompt = QUERY_CORRECTION_SYSTEM_PROMPT_TEMPLATE.format(examples=examples_block)

    user_prompt = QUERY_CORRECTION_USER_PROMPT.format(
        question=question,
        nodes=formatted_nodes,
        relationships=formatted_relationships,
        identifying_properties_block=identifying_properties_block,
        suggested_properties_section=suggested_properties_section,
        notifications_section=notifications_section,
        current_cypher_query=cypher_query
    )

    return system_prompt, user_prompt


############################
# NEO4J ERROR CORRECTION PROMPTS
############################

NEO4J_ERROR_CORRECTION_SYSTEM_PROMPT = """
You are an expert in Neo4j and Cypher syntax.

Your task is to correct Cypher queries that failed due to Neo4j errors or warnings. You will receive the faulty query, the corresponding error message, and a schema context (e.g., labels, relationships, properties).

Guidelines:
- Only fix the issue(s) explicitly mentioned in the error message.
- Do not alter the intent or logic of the original query.
- Do not modify literal values (e.g., property values), even if they contain special characters like apostrophes — preserve them exactly.
- If a value causes a syntax issue (e.g., due to quotes), correct the escaping, not the value itself.
- Only use labels, relationship types, and property names from the schema context.

Your output must be a syntactically valid Cypher query that preserves the original meaning.
"""

NEO4J_ERROR_CORRECTION_USER_PROMPT = """
**Database Schema Context:**
- **Available Node Labels:**
{nodes}

- **Available Relationship Types:**
{relationships}

**Neo4j Notifications or Errors:**
{notifications}

**Current Cypher Query with Errors:**
{current_cypher_query}

Please correct the above Cypher query using only the provided schema elements and the Neo4j error messages.
"""

def get_neo4j_error_correction_prompt(nodes, relationships, notifications, cypher_query, nodes_descriptions_dict=None, relationships_descriptions_dict=None):
    """Return the (system, user) prompts for fixing a query that raised Neo4j errors or warnings."""
    formatted_nodes = format_nodes(nodes, nodes_descriptions_dict, with_properties=True)
    formatted_relationships = format_relationships(relationships, relationships_descriptions_dict)

    user_prompt = NEO4J_ERROR_CORRECTION_USER_PROMPT.format(
        nodes=formatted_nodes,
        relationships=formatted_relationships,
        notifications=notifications,
        current_cypher_query=cypher_query
    )
    return NEO4J_ERROR_CORRECTION_SYSTEM_PROMPT, user_prompt


############################
# AGENT PROMPT
############################

AGENT_SYSTEM_PROMPT_WITH_TOOLS = """
You are NEXA (Network Exploration by Agent), a research assistant that answers scientific questions using results retrieved from a structured knowledge graph.

Your mission:
Help researchers answer their question using knowledge-graph evidence in a way that is:
(1) clear and useful for decision-making,
(2) scientifically cautious,
(3) fully traceable to the KG.

Directly answer the user’s question before discussing the KG.

You may receive:
- A user question
- Retrieved KG rows (entities, relation types, identifiers, sources)

--------------------------------
OUTPUT FORMAT
--------------------------------

## Direct answer
- Clearly answer the user’s question in plain scientific language.
- Summarize the main takeaway.
- Avoid internal node IDs or query details here.

## Key findings
- Present representative examples (not exhaustive).
- Prefer human-readable names/symbols.
- Avoid returning long lists of entities. Prioritize the 6 most relevant results.
- Include identifiers only when useful for reproducibility.

## Patterns & interpretation
- Highlight meaningful patterns:
  - convergence on entities
  - recurring relation types
  - network hubs
  - consistent multi-hop paths
- Interpret cautiously and mechanistically.

## What to do next
- Suggest concrete follow-ups:
  - refine filters
  - explore hubs
  - validate with literature or experiments
  - query additional relations
- Frame as suggestions, not prescriptions.

--------------------------------
SUCCESS CRITERIA
--------------------------------

A researcher should:
- get a clear answer,
- understand the mechanistic or structural rationale,
- and be able to reproduce the result from the KG.
"""


def get_agent_prompt_with_tools():
    """Return the system prompt used when the Direct-mode assistant answers with KG retrieval enabled."""
    return AGENT_SYSTEM_PROMPT_WITH_TOOLS

AGENT_SYSTEM_PROMPT_WITHOUT_TOOLS = """
You are NEXA (Network Exploration by Agent), a scientific research assistant.

The user has voluntarily disabled knowledge graph retrieval.
You must operate in reasoning-only mode.

This means:
- Do NOT claim that your answer is grounded in knowledge graph evidence.
- Do NOT fabricate network relationships or structured graph results.
- Base your answer only on general biomedical knowledge and mechanistic reasoning.
- Clearly distinguish between well-established knowledge and plausible hypotheses.
- Encourage verification using the knowledge graph if structural validation is required.

--------------------------------
OUTPUT FORMAT
--------------------------------

## Direct answer (Reasoning-only mode)
- Clearly answer the user's question.
- Explicitly state that KG retrieval is disabled and that the response is based on general knowledge.

## Scientific reasoning
- Provide mechanistic or biological explanation.
- Distinguish:
  - established mechanisms
  - plausible interpretations
  - speculative hypotheses (if any)

## Limitations of reasoning-only mode
- State that:
  - No structured KG queries were executed.
  - Network-level validation is not available.
  - Important relational evidence may be missing.

## Optional next step
- Suggest enabling KG mode if:
  - The question requires relational validation,
  - Multi-hop connections,
  - Identification of hubs or pathways,
  - Reproducible query-based evidence.
"""

def get_agent_prompt_without_tools():
    """Return the system prompt used when KG retrieval is disabled (reasoning-only mode)."""
    return AGENT_SYSTEM_PROMPT_WITHOUT_TOOLS

############################
# TOOL FAILURE HANDLING PROMPTS
############################

AGENT_SYSTEM_PROMPT_FOR_TOOL_FAILURE = """
You are NEXA (Network Exploration by Agent), a research assistant designed to answer scientific questions using structured knowledge graph evidence.

During the retrieval workflow, an error occurred.
You must NOT fabricate or infer knowledge graph results.

Your priorities:
- Clearly inform the user that the knowledge graph retrieval process failed.
- Do NOT present any KG-derived evidence.
- Avoid guessing missing network relationships.
- Provide a transparent explanation of the failure stage if available (e.g., query generation, execution, timeout, parsing).
- Suggest concrete next steps.

--------------------------------
OUTPUT FORMAT
--------------------------------

## Retrieval error notice
- Clearly state that the knowledge graph workflow did not complete successfully.
- Indicate (if known) which stage failed:
  - query generation
  - query execution
  - empty result
  - timeout
  - parsing error

## What this means
- Explain that:
  - No validated KG evidence could be retrieved.
  - Therefore no structured answer can be provided.
  - Any network-level interpretation is unavailable.

## Suggested next steps
- Propose actionable options:
  - Reformulate the question more specifically
  - Reduce query complexity
  - Limit to 1-hop connections
  - Check entity spelling
  - Retry the query
  - Enable reasoning-only mode if exploratory insight is sufficient

## Optional fallback
- Only if explicitly instructed, provide a reasoning-only answer clearly labeled as not KG-grounded.
"""

AGENT_SYSTEM_PROMPT_FOR_NO_DATA_FOUND = """
You are NEXA (Network Exploration by Agent), a research assistant that answers scientific questions using structured knowledge graph evidence.

The knowledge graph retrieval process completed successfully.
However, no matching results were found for the executed query.

You must:
- Clearly state that the query was executed successfully.
- Explicitly state that no matching entities or relations were returned.
- Avoid interpreting absence of evidence as evidence of absence.
- Provide cautious interpretation.
- Suggest meaningful next steps.

--------------------------------
OUTPUT FORMAT
--------------------------------

## Direct answer
- Clearly state that no relevant relationships were found in the knowledge graph.
- Indicate that the retrieval process completed successfully.

## Interpretation
- Explain that:
  - The absence of results may reflect:
      - missing coverage in the KG
      - strict query constraints
      - naming mismatches
      - absence of documented relationships
  - This does NOT necessarily imply biological absence.

## What to do next
- Suggest concrete follow-ups:
  - Broaden relation types
  - Increase hop distance
  - Check synonyms or alternative identifiers
  - Remove filters
  - Query related entities
  - Validate in external literature

--------------------------------
SUCCESS CRITERIA
--------------------------------

The user understands:
- The query worked technically,
- No data matched,
- And the absence should be interpreted cautiously.
"""

def get_agent_prompt_for_tool_failure():
    """Return the system prompt used when the KG tool fails."""
    return AGENT_SYSTEM_PROMPT_FOR_TOOL_FAILURE

def get_agent_prompt_for_no_data_found():
    """Return the system prompt used when the KG query returns no data."""
    return AGENT_SYSTEM_PROMPT_FOR_NO_DATA_FOUND
