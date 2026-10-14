"""Filters that narrow the KG schema (labels, relationship types, relationship properties) to a selection."""
def filter_nodes_by_labels(nodes, labels_to_select):
    """Return the nodes whose labels are in ``labels_to_select``."""
    return [node for node in nodes if node['node']['labels'] in labels_to_select]

def filter_relationships_by_nodes(relationships, selected_nodes):
    """Return the relationships whose source or target node is in ``selected_nodes``."""
    selected_set = set(selected_nodes)
    return [
        rel for rel in relationships
        if rel.get('source') in selected_set or rel.get('target') in selected_set
    ]

def filter_relationships_by_type(relationships, types_to_select):
    """Return the relationships whose type is in ``types_to_select``."""
    return [rel for rel in relationships if rel['relationship'] in types_to_select]

def filter_relationships_props_by_type(relationships, types_to_select):
    """Return the relationship-property entries whose relationship type is in ``types_to_select``."""
    return [rel for rel in relationships if rel['relationship']['type'] in types_to_select]
