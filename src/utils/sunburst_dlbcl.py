import pandas as pd
import plotly.express as px
from utils import data_loader as lo


def plot_sunburst_dynamics():
    ## attention: ICDO3 or ICD10? Granularity is lost through ICDO3, as ICDO3 links concepts to the episode.
    sql = """
        select
        dlbcl.person_id,
        dlbcl.condition_source_concept_id,
        c.concept_name,
        e.episode_start_date,
        e.episode_end_date,
        e.episode_source_value
        
        from cdm.analytics_diffuse_large_b_cell_lymphoma_prepared dlbcl  
        left join cdm.episode e 
        on e.person_id = dlbcl.person_id
        left join cdm.concept c 
        on c.concept_id = e.episode_concept_id 
        
        where e.episode_concept_id in (32948,32949,3294,32946,32947)
    """

    dlbcl_dynamic = lo.execute_sql(sql)

    # sunburst visualisation
    dlbcl_dynamic['episode_start_date'] = pd.to_datetime(dlbcl_dynamic['episode_start_date'])
    # sorting
    df_sorted = dlbcl_dynamic.sort_values(by=['person_id', 'episode_start_date', 'condition_source_concept_id'])
    # dnymaics path sorted
    paths = (
        df_sorted
        .groupby(['person_id','condition_source_concept_id'])['concept_name']
        .apply(lambda x: ' -> '.join(x))
        .reset_index(name='concept_path')
    )

    # calculate paths
    paths['path_length'] = paths['concept_path'].apply(lambda x: len(x.split(' -> ')))

    # splitting paths into levels
    max_depth = paths['concept_path'].str.count('->').max() + 1

    for i in range(int(max_depth)):
        paths[f'level_{i+1}'] = paths['concept_path'].apply(
            lambda x: x.split(' -> ')[i] if i < len(x.split(' -> ')) else None
        )
    print("Paths with levels:")
    print(paths)

    # 2. tree structure
    ids = []
    parents = []
    labels = []

    for _, row in paths.iterrows():
        levels = [row[f'level_{i+1}'] for i in range(max_depth)]
        for i, level in enumerate(levels):
            if pd.isna(level):
                continue
            curr_id = " - ".join(levels[:i+1])
            parent = " - ".join(levels[:i]) if i > 0 else ""
            ids.append(curr_id)
            parents.append(parent)
            labels.append(levels[i])

    # 3. Calculate unique id-parents for dataframe
    n = 100
    df_nodes = pd.DataFrame({'id': ids, 'parent': parents, 'label': labels})
    df_nodes = df_nodes.value_counts().reset_index(name='count')
    df_nodes
    df_nodes= df_nodes[df_nodes['count'] > n]
    # df_nodes = df_nodes[~df_nodes['id'].isin(['Complete Remission - Stable Disease - Complete Remission', 'Complete Remission - Complete Remission', 'Stable Disease - Complete Remission'])]
    print(f"Filtered nodes with count > {n}:")
    print(df_nodes)

    # color formatting
    color_map = {
        'Stable Disease': 'rgba(153, 217, 190, 0.5)',  
        'Complete Remission': 'rgba(158, 202, 225, 0.5)',  
        'Partial Remission': 'rgba(141, 160, 203, 0.5)',  
        'Progression': 'rgba(244, 182, 216, 0.5)'          
    }
    df_nodes['color'] = df_nodes['label'].map(color_map).fillna('#D3D3D380') 
    df_nodes['label_count'] = df_nodes.apply(
        lambda row: f"{row['label']} ({row['count']})", axis=1
    )

    # 4. Sunburst plotten
    fig = px.sunburst(
        df_nodes,
        ids='id',
        names='label_count',
        parents='parent',
        color='label',
        color_discrete_map=color_map,
        values='count',
        title=f"Patient path of the concept_name (length > {n}, ordered chronologically)",
        width=1200,   
        height=1200
    )

    fig.update_traces(
        insidetextfont=dict(size=18),
        outsidetextfont=dict(size=14)
        #textinfo='label_count'
    )

    fig.update_layout(
        uniformtext=dict(minsize=11.5, mode='show'),
        title_font_size=24,
        font=dict(size=16)
    )
    fig.show()
    fig.write_html("src/results/sunburst_dynamics.html")

def plot_sunburst_regimen():

    sql2 = """select
        dlbcl.person_id,
        dlbcl.condition_source_concept_id,
        c.concept_name,
        e.episode_start_date,
        e.episode_end_date,
        e.episode_source_value
        
        from cdm.analytics_diffuse_large_b_cell_lymphoma_prepared dlbcl  
        left join cdm.episode e 
        on e.person_id = dlbcl.person_id
        left join cdm.concept c 
        on c.concept_id = e.episode_object_concept_id 
        
        where c.vocabulary_id = 'HemOnc'"""

                
    dlbcl_regimen = lo.execute_sql(sql2)

    # formatting
    dlbcl_regimen['episode_start_date'] = pd.to_datetime(dlbcl_regimen['episode_start_date'])

    # sorting
    regimen_sorted = dlbcl_regimen.sort_values(by=['person_id', 'episode_start_date', 'condition_source_concept_id'])


    # regimens path sorted
    paths_reg = (
        regimen_sorted
        .groupby(['person_id'])['concept_name']
        .apply(lambda x: ' -> '.join(x))
        .reset_index(name='concept_path')
    )

    # calculate paths
    paths_reg['path_length'] = paths_reg['concept_path'].apply(lambda x: len(x.split(' -> ')))

    # splitting paths into levels
    max_depth_reg = paths_reg['concept_path'].str.count('->').max() + 1

    for i in range(max_depth_reg):
        paths_reg[f'level_{i+1}'] = paths_reg['concept_path'].apply(
            lambda x: x.split(' -> ')[i] if i < len(x.split(' -> ')) else None
        )

    level1_counts_reg = paths_reg['level_1'].value_counts()
    n_reg = 49
    valid_level1_reg = level1_counts_reg[level1_counts_reg >= n_reg].index
    paths_reg = paths_reg[paths_reg['level_1'].isin(valid_level1_reg)]
    print("Paths with levels:")
    print(paths_reg)

    # 2. tree structure
    ids_reg = []
    parents_reg = []
    labels_reg = []

    for _, row in paths_reg.iterrows():
        levels_reg = [row[f'level_{i+1}'] for i in range(max_depth_reg)]
        for i, level_reg in enumerate(levels_reg):
            if pd.isna(level_reg):
                break
            curr_id = " - ".join(levels_reg[:i+1])
            parent = " - ".join(levels_reg[:i]) if i > 0 else ""
            ids_reg.append(curr_id)
            parents_reg.append(parent)
            labels_reg.append(levels_reg[i])

    # 3. Calculate unique id-parents for dataframe
    n2 = 13
    df_nodes_reg = pd.DataFrame({'id': ids_reg, 'parent': parents_reg, 'label': labels_reg})
    df_nodes_reg = df_nodes_reg.value_counts().reset_index(name='count')
    df_nodes_reg
    df_nodes_reg= df_nodes_reg[df_nodes_reg['count'] > n2]
    print(f"Filtered nodes with count > {n2}:")
    print(df_nodes_reg)

    # color formatting with matplotlib
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import colorsys

    # unique Labels of df_nodes_reg
    unique_labels = df_nodes_reg['label'].unique()

    def generate_pastel_colors(n):
        colors = []
        for i in range(n):
            hue = i / n 
            saturation = 0.4            
            lightness = 0.7
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append((r, g, b, 0.6))  # alpha 
        return colors

    num_colors = len(unique_labels)
    colors_rgba = generate_pastel_colors(num_colors)


    # permutation of colors
    indices = list(range(num_colors))
    shuffled_indices = indices[::2] + indices[1::2]

    # resort color
    colors_rgba_shuffled = [colors_rgba[i] for i in shuffled_indices]

    color_map_reg = {
        label: f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})'
        for label, (r, g, b, a) in zip(unique_labels, colors_rgba_shuffled)
    }

    df_nodes_reg['color'] = df_nodes_reg['label'].map(color_map_reg).fillna('#D3D3D380') 
    df_nodes_reg['label_count'] = df_nodes_reg.apply(
        lambda row: f"{row['label']} ({row['count']})", axis=1
    )

    # 4. Sunburst plotten
    fig = px.sunburst(
        df_nodes_reg,
        ids='id',
        names='label_count',
        parents='parent',
        color='label',
        color_discrete_map=color_map_reg,
        values='count',
        #title=f"",
        width=1200,   
        height=1200
    )

    fig.update_traces(
        insidetextfont=dict(size=18),
        outsidetextfont=dict(size=14)
        #textinfo='label_count'
    )

    fig.update_layout(
        uniformtext=dict(minsize=14, mode='show'),
        title_font_size=24,
        font=dict(size=16)
    )
    fig.show()
    fig.write_html("src/results/sunburst_regimen.html")