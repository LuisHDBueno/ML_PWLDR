import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.lines as mlines

# graficos
def gen_time_charts(df_rm, df_md, df_pwldr, name):
    df_1 = df_pwldr[["idx_p","idx_v","nb","displace_func","time"]]
    df_2 = df_md[["idx_p","idx_v","variable"]]
    df_time = pd.merge(df_1, df_2, on=['idx_p', 'idx_v'], how='left')
    df_mean_time = df_time.groupby(['nb', 'displace_func'])['time'].mean().reset_index()

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df_mean_time,
        x='nb',
        y='time',
        hue='displace_func',
        marker='o'
    )

    plt.title(f'Mean Time by Displace Algorithm at {name}')
    plt.xlabel('Breakpoint Number')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.legend(title='Displace Function')
    plt.savefig(f"img/{name}_time_algorithm.png")


    df_local_search = df_time[df_time["displace_func"] == "local_search"]
    df_mean_time_variable = df_local_search.groupby(['nb', 'variable'])['time'].mean().reset_index()

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df_mean_time_variable,
        x='nb',
        y='time',
        hue='variable',
        marker='o'
    )

    plt.title(f'Mean Time by Variable Type at {name}')
    plt.xlabel('Breakpoint Number')
    plt.ylabel('Time (s)')
    plt.grid(True)
    plt.legend(title='Variable')
    plt.savefig(f"img/{name}_time_variable.png")

def gen_dr_gain_charts(df_rm, df_md, df_pwldr, name):
    df_1 = pd.merge(df_pwldr, df_md, on=['idx_p', 'idx_v'], how='left')
    df_1 = df_1[["idx_p","idx_v","nb","metric","displace_func","value_opt", "value_uni", "variable", "EVPI"]]
    df_1["gain"] = - (df_1["value_opt"] - df_1["value_uni"]) / df_1["EVPI"]

    df_1 = df_1[df_1["displace_func"] == "local_search"]
    df_1 = df_1[df_1["metric"] == "obj_pwldr"]
    chart_1 = df_1[df_1["nb"] != 0]

    chart_1 = chart_1.groupby(['nb', 'variable'])['gain'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=chart_1,
        x='nb',
        y='gain',
        hue='variable',
        marker='o'
    )

    plt.title(f'Mean Reduction of Uncertainty For Optimize Displacement by EVPI At {name}')
    plt.xlabel('Breakpoint Number')
    plt.ylabel('Gain (DR Value Opt - DR Value Uni) / EVPI)')
    plt.grid(True)
    plt.legend(title='Variable')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.savefig(f"img/{name}_dr_gain_opt_displacement.png")

    df_baseline = df_1[df_1['nb'] == 0][[
        'idx_p', 'idx_v', 'value_opt'
    ]]

    df_baseline = df_baseline.rename(columns={
        'value_opt': 'value_opt_nb0',
    })
    df_1 = pd.merge(
        df_1,
        df_baseline,
        on=['idx_p', 'idx_v'],
        how='left'
    )
    df_1['gain_vs_nb0'] = (df_1['value_opt_nb0'] - df_1['value_opt']) / (df_1['EVPI'])
    chart_2_data = df_1[df_1["nb"] != 0]
    chart_2 = chart_2_data.groupby(['nb', 'variable'])['gain_vs_nb0'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=chart_2,
        x='nb',
        y='gain_vs_nb0',
        hue='variable',
        marker='o'
    )

    plt.title(f'Mean Reduction of Uncertainty by EVPI At {name}')
    plt.xlabel('Breakpoint Number')
    plt.ylabel('Gain ((LDR - PWLDR) / (EVPI))')
    plt.grid(True)
    plt.legend(title='Variable')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.savefig(f"img/{name}_dr_nb.png")

def gen_metrics_charts(df_rm, df_md, df_pwldr, name):
    df_1 = df_pwldr[["idx_p","idx_v","nb","displace_func","value_uni", "value_opt", "metric", "WS", "EEV", "EVPI", "VSS"]]
    df_1 = df_1[df_1["displace_func"] == "local_search"]
    df_1 = df_1[df_1["metric"] != "obj_pwldr"]
    df_2 = df_md[["idx_p","idx_v","variable"]]

    df_metrics = pd.merge(df_1, df_2, on=['idx_p', 'idx_v'], how='left')

    df_metrics["EVPI_Uni_Relative"] = (df_metrics["value_uni"] - df_metrics["WS"])/df_metrics["EVPI"]
    df_metrics["EVPI_Opt_Relative"] = (df_metrics["value_opt"] - df_metrics["WS"])/df_metrics["EVPI"]

    df_metrics["VSS_Uni_Relative"] = (df_metrics["EEV"] - df_metrics["value_uni"])/df_metrics["VSS"]
    df_metrics["VSS_Opt_Relative"] = (df_metrics["EEV"] - df_metrics["value_opt"])/df_metrics["VSS"]

    df_dr = df_metrics[df_metrics["metric"] == "dr_pwldr"]

    df_evpi = df_dr.groupby(['nb', 'variable'])[["EVPI_Uni_Relative", "EVPI_Opt_Relative"]].mean().reset_index()
    df_evpi = df_evpi[df_evpi["nb"] != 0]
    df_vss = df_dr.groupby(['nb', 'variable'])[["VSS_Uni_Relative", "VSS_Opt_Relative"]].mean().reset_index()
    df_vss = df_vss[df_vss["nb"] != 0]


    colors = plt.cm.tab10.colors
    variables = df_evpi['variable'].unique()
    color_map = {var: colors[i % len(colors)] for i, var in enumerate(variables)}
    fig, ax = plt.subplots(figsize=(8,5))

    for var, data in df_evpi.groupby('variable'):
        ax.plot(
            data['nb'], data['EVPI_Uni_Relative'],
            linestyle='--', color=color_map[var], label=f'{var} (Uni)'
        )
        ax.plot(
            data['nb'], data['EVPI_Opt_Relative'],
            linestyle='-', color=color_map[var], label=f'{var} (Opt)'
        )

    # Add EVPI baseline
    #ax.axhline(1, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('nb')
    ax.set_ylabel('EVPI_DR /EPVI')
    ax.set_title('Comparative EVPI Uni and EVPI Opt by Variable')
    ax.legend()
    ax.grid(True)

    color_handles = [
        mlines.Line2D([], [], color=color_map[var], linestyle='-', label=var)
        for var in variables
    ]

    style_handles = [
        mlines.Line2D([], [], color='black', linestyle='--', label='EVPI_Uni_Relative'),
        mlines.Line2D([], [], color='black', linestyle='-', label='EVPI_Opt_Relative')
    ]

    first_legend = ax.legend(
        handles=color_handles,
        title='Variable',
        bbox_to_anchor=(1.05, 0.6),
        loc='center left'
    )

    ax.add_artist(first_legend)
    ax.legend(
        handles=style_handles,
        title='Metric',
        bbox_to_anchor=(1.05, 0.25),
        loc='center left'
    )

    plt.tight_layout()
    plt.savefig(f"img/{name}_comparative_evpi_variable.png")

    df_mean = df_evpi.groupby('nb')[["EVPI_Uni_Relative", "EVPI_Opt_Relative"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(df_mean['nb'], df_mean['EVPI_Uni_Relative'], '--', label='EVPI Uni/EVPI')
    ax.plot(df_mean['nb'], df_mean['EVPI_Opt_Relative'], '-', label='EVPI Opt/EVPI')

    # área sombreada
    ax.fill_between(
        df_mean['nb'],
        df_mean['EVPI_Uni_Relative'],
        df_mean['EVPI_Opt_Relative'],
        color='gray',
        alpha=0.3,
        label='Diff of Relative EVPI'
    )
    # Add EVPI baseline
    #ax.axhline(1, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('nb')
    ax.set_ylabel('EVPI_DR /EPVI')
    ax.set_title('Comparative EVPI')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"img/{name}_comparative_evpi.png")

    # VSS
    for var, data in df_vss.groupby('variable'):
        ax.plot(
            data['nb'], data['VSS_Uni_Relative'],
            linestyle='--', color=color_map[var], label=f'{var} (Uni)'
        )
        ax.plot(
            data['nb'], data['VSS_Opt_Relative'],
            linestyle='-', color=color_map[var], label=f'{var} (Opt)'
        )

    # Add VSS baseline
    #ax.axhline(1, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('nb')
    ax.set_ylabel('VSS_DR /VSS')
    ax.set_title('Comparative VSS Uni and VSS Opt by Variable')
    ax.legend()
    ax.grid(True)

    color_handles = [
        mlines.Line2D([], [], color=color_map[var], linestyle='-', label=var)
        for var in variables
    ]

    style_handles = [
        mlines.Line2D([], [], color='black', linestyle='--', label='VSS_Uni_Relative'),
        mlines.Line2D([], [], color='black', linestyle='-', label='VSS_Opt_Relative')
    ]

    first_legend = ax.legend(
        handles=color_handles,
        title='Variable',
        bbox_to_anchor=(1.05, 0.6),
        loc='center left'
    )

    ax.add_artist(first_legend)
    ax.legend(
        handles=style_handles,
        title='Metric',
        bbox_to_anchor=(1.05, 0.25),
        loc='center left'
    )

    plt.tight_layout()
    plt.savefig(f"img/{name}_comparative_vss_variable.png")

    df_mean = df_vss.groupby('nb')[["VSS_Uni_Relative", "VSS_Opt_Relative"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(df_mean['nb'], df_mean['VSS_Uni_Relative'], '--', label='VSS Uni/VSS')
    ax.plot(df_mean['nb'], df_mean['VSS_Opt_Relative'], '-', label='VSS Opt/VSS')

    # área sombreada
    ax.fill_between(
        df_mean['nb'],
        df_mean['VSS_Uni_Relative'],
        df_mean['VSS_Opt_Relative'],
        color='gray',
        alpha=0.3,
        label='Diff of Relative VSS'
    )
    # Add VSS baseline
    #ax.axhline(1, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('nb')
    ax.set_ylabel('VSS_DR /VSS')
    ax.set_title('Comparative VSS')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"img/{name}_comparative_vss.png")

def plot_cost_composition(ws, rp, eev, show_annotations=True):

    if not (ws <= rp <= eev):
        print(f"Warning: The relationship WS ({ws:.2f}) <= RP ({rp:.2f}) <= EEV ({eev:.2f}) "
              "is not valid. The plot components may be misleading.")
        
    evpi = max(0, rp - ws)
    vss = max(0, eev - rp)
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))

    x_pos = 'Cost Composition'
    
    ax.bar(x_pos, ws, label=f'WS = {ws:.2f}', color='#440154')
    
    ax.bar(x_pos, evpi, bottom=ws, label=f'EVPI = {evpi:.2f}', color='#21908d')
    
    ax.bar(x_pos, vss, bottom=rp, label=f'VSS = {vss:.2f}', color='#fde725')

    if show_annotations:
        ax.text(x_pos, eev * 1.02, f'Total EEV = {eev:.2f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.text(x_pos, ws / 2, f'{ws:.2f}', ha='center', va='center', color='white', fontweight='bold')
        ax.text(x_pos, ws + (evpi / 2), f'{evpi:.2f}', ha='center', va='center', color='white', fontweight='bold')
        ax.text(x_pos, rp + (vss / 2), f'{vss:.2f}', ha='center', va='center', color='black', fontweight='bold')

    ax.set_title('Solution Cost Composition (EEV)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Total Value / Cost')
    ax.set_xlabel('')
    ax.legend(title="Components", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xticks([]) 
    
    plt.tight_layout()
    plt.savefig(f"img/shipment_planning_composition.png")


def add_metrics_to_df(df_rm, df_md, df_pwldr):
    # Add nb = 0 to pwldr dataframe
    df1_unique = df_pwldr[['idx_p', 'idx_v', 'displace_func']].drop_duplicates()
    df_rm['metric'] = df_rm['metric'].str.replace('_ldr', '_pwldr')
    df_rm_filter = df_rm[df_rm["metric"].isin(["obj_pwldr", "reopt_pwldr", "dr_pwldr"])]
    new_lines = pd.merge(df1_unique, df_rm_filter, on='idx_p')

    new_lines['nb'] = 0
    new_lines['time'] = 0
    new_lines['value_uni'] = new_lines['value']
    new_lines['value_opt'] = new_lines['value']

    new_lines = new_lines.drop(columns=['value'])
    new_lines = new_lines[df_pwldr.columns]

    df_pwldr = pd.concat([df_pwldr, new_lines], ignore_index=True)

    # Add RP, WS, EEV, EVPI and VSS to pwldr dataframe
    df_pivot = df_rm.pivot(
        index='idx_p', 
        columns='metric', 
        values='value'
    )

    df_pwldr["RP"] = df_pwldr['idx_p'].map(df_pivot['reopt_std'])
    df_pwldr["WS"] = df_pwldr['idx_p'].map(df_pivot['ws'])
    df_pwldr['EEV'] = df_pwldr['idx_p'].map(df_pivot['reopt_deterministic'])
    df_pwldr["EVPI"] = df_pwldr["RP"] - df_pwldr["WS"]
    df_pwldr["VSS"] = df_pwldr["EEV"] - df_pwldr["RP"]

    return df_rm, df_md, df_pwldr

def gen_shipment_planing_charts():
    sp_regular_metrics = "shipment_planning_regular_metrics.csv"
    sp_metadata = "shipment_planning_pwldr_metadata.csv"
    sp_pwldr = "shipment_planning_pwldr_metrics.csv"

    df_rm = pd.read_csv(sp_regular_metrics)
    df_md = pd.read_csv(sp_metadata)
    df_pwldr = pd.read_csv(sp_pwldr)

    df_md_fixed = df_md[["idx_p","idx_v","variable", "v5", "v10", "v15"]]
    df_md_fixed['variable'] = df_md_fixed['variable'].str.replace('Truncated{', '')
    df_md_fixed['variable'] = df_md_fixed['variable'].str.split('{').str[0]

    df_md_fixed['v_sum'] = df_md_fixed['v5'] + df_md_fixed['v10'] + df_md_fixed['v15']
    df_md_fixed['max_v_sum'] = df_md_fixed.groupby('idx_p')['v_sum'].transform('max')

    is_normal = df_md_fixed['variable'] == 'Normal'
    is_max_sum = df_md_fixed['v_sum'] == df_md_fixed['max_v_sum']

    df_md_fixed.loc[is_normal & is_max_sum, 'variable'] = 'Normal(50, 40)'
    df_md_fixed.loc[is_normal & ~is_max_sum, 'variable'] = 'Normal(50, 15)'

    df_md["variable"] = df_md_fixed['variable']

    df_rm, df_md, df_pwldr = add_metrics_to_df(df_rm, df_md, df_pwldr)

    gen_time_charts(df_rm, df_md, df_pwldr, "shipment_planning")
    gen_dr_gain_charts(df_rm, df_md, df_pwldr, "shipment_planning")
    gen_metrics_charts(df_rm, df_md, df_pwldr, "shipment_planning")

def gen_capacity_expansion_charts():
    ce_regular_metrics = "capacity_expansion_regular_metrics.csv"
    ce_metadata = "capacity_expansion_pwldr_metadata.csv"
    ce_pwldr = "capacity_expansion_pwldr_metrics.csv"

    df_rm = pd.read_csv(ce_regular_metrics)
    df_md = pd.read_csv(ce_metadata)
    df_pwldr = pd.read_csv(ce_pwldr)

    df_md_fixed = df_md[["idx_p","idx_v","variable", "v5", "v10", "v15"]]
    df_md_fixed['variable'] = df_md_fixed['variable'].str.replace('Truncated{', '')
    df_md_fixed['variable'] = df_md_fixed['variable'].str.split('{').str[0]

    df_md_fixed['v_sum'] = df_md_fixed['v5'] + df_md_fixed['v10'] + df_md_fixed['v15']
    df_md_fixed['v_sum_rank'] = df_md_fixed.groupby('idx_p')['v_sum'].rank(ascending=False, method='first')

    is_normal = df_md_fixed['variable'] == 'Normal'
    is_top_2_sum = df_md_fixed['v_sum_rank'] <= 2 

    df_md_fixed.loc[is_normal & is_top_2_sum, 'variable'] = 'Normal(50, 40)'
    df_md_fixed.loc[is_normal & ~is_top_2_sum, 'variable'] = 'Normal(50, 15)'

    df_md["variable"] = df_md_fixed['variable']

    df_rm, df_md, df_pwldr = add_metrics_to_df(df_rm, df_md, df_pwldr)

    gen_time_charts(df_rm, df_md, df_pwldr, "capacity_expansion")
    gen_dr_gain_charts(df_rm, df_md, df_pwldr, "capacity_expansion")
    gen_metrics_charts(df_rm, df_md, df_pwldr, "capacity_expansion")


#gen_shipment_planing_charts()
#gen_capacity_expansion_charts()
ws_val = 100
rp_val = 120
eev_val = 150

plot_cost_composition(ws_val, rp_val, eev_val, show_annotations=True)