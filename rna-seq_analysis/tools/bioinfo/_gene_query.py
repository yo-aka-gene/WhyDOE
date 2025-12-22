import gseapy as gp
import mygene
import numpy as np
import pandas as pd


def gene_query(
    gene_names: list,
    source: list,
    species: str = "human",
    logging: bool = True,
    unique: bool = True
) -> dict:
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        gene_names, 
        scopes="symbol,alias", 
        fields="symbol,alias", 
        species="human",
        as_dataframe=True
    )

    source_set = set(source)
    final_genes = []
    
    found_count = 0

    for query in np.unique(gene_names):
        if query not in res.index:
            continue

        match_rows = res.loc[[query]]
        
        candidates = []
        
        for _, row in match_rows.iterrows():
            if not pd.isna(row.get('symbol')):
                candidates.append(row['symbol'])
            
            aliases = row.get('alias')
            if isinstance(aliases, list):
                candidates.extend(aliases)
            elif isinstance(aliases, str):
                candidates.append(aliases)

            candidates.append(query)

        candidates = list(set(candidates))
        
        match_found = False
        for cand in candidates:
            if cand in source_set:
                final_genes.append(cand)
                match_found = True
                found_count += 1
                break
        
        if not match_found:
            if logging:
                print(f"Not found in source: {query} (Candidates: {candidates})")
            pass

    if logging:
        print(f"[{found_count}/{len(np.unique(gene_names))}] queries mapped to the source.")
        if unique:
             print(f" -> Returning {len(set(final_genes))} unique genes present in data.")

    return sorted(list(set(final_genes))) if unique else final_genes
