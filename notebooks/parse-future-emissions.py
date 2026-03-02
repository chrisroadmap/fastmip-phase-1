# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Make nice input files for fair

# %%
import os

import pandas as pd

# %%
historical = pd.read_csv('../data/emissions/historical_emissions_1750-2023_cmip7.csv')

# %%
historical

# %%
future = pd.read_excel('../data/emissions/ScenarioMIP_emissions_marker_scenarios_v0.1.xlsx', sheet_name='data')

# %%
future

# %%
future_species = list(future[future.variable.str.startswith('Climate Assessment|Harmonized and Infilled|Emissions')].variable.unique())

# %%
future_species

# %%
future_species.remove('Climate Assessment|Harmonized and Infilled|Emissions|Kyoto GHG AR6GWP100')
future_species.remove('Climate Assessment|Harmonized and Infilled|Emissions|GHG AR6GWP100')
future_species.remove('Climate Assessment|Harmonized and Infilled|Emissions|CO2')

# %%
rename_map = {specie: specie.replace('Climate Assessment|Harmonized and Infilled|Emissions|', '') for specie in future_species}

# %%
rename_map

# %%
rename_map['Climate Assessment|Harmonized and Infilled|Emissions|CO2|AFOLU'] = 'CO2 AFOLU'
rename_map['Climate Assessment|Harmonized and Infilled|Emissions|CO2|Energy and Industrial Processes'] = 'CO2 FFI'
rename_map['Climate Assessment|Harmonized and Infilled|Emissions|cC4F8'] = 'c-C4F8'
rename_map['Climate Assessment|Harmonized and Infilled|Emissions|HFC|HFC43-10'] = 'HFC-4310mee'

for key, value in rename_map.items():
    rename_map[key] = value.replace('HFC|HFC', 'HFC-')
    
for key, value in rename_map.items():
    rename_map[key] = value.replace('Halon', 'Halon-')

for key, value in rename_map.items():
    rename_map[key] = value.replace('CFC', 'CFC-')

# %%
#rename_map

# %%
future_emissions = future[future.variable.str.startswith('Climate Assessment|Harmonized and Infilled|Emissions')]

# %%
future_emissions_renamed = future_emissions.replace(rename_map)

# %%
future_emissions_renamed

# %%
future_emissions_renamed = future_emissions_renamed.drop(columns=[str(year) for year in range(2000, 2023)])

# %%
units = list(future_emissions_renamed.unit.unique())

# %%
units

# %%
units_map = {unit: unit for unit in units}

# %%
units_map['Mt CO2/yr'] = 'Gt CO2/yr'
units_map['MtCO2 / yr'] = 'Gt CO2/yr'
units_map['kt N2O/yr'] = 'Mt N2O/yr'

# %%
future_emissions_renamed_reunited = future_emissions_renamed.replace(units_map)

# %%
future_emissions_renamed_reunited

# %%
future_emissions_renamed_reunited_dedafted = future_emissions_renamed_reunited.copy()

for var_dedaft in ['CO2 FFI', 'CO2 AFOLU', 'N2O']:
    future_emissions_renamed_reunited_dedafted.loc[future_emissions_renamed_reunited.variable == var_dedaft, '2023':] = (
        future_emissions_renamed_reunited.loc[future_emissions_renamed_reunited.variable == var_dedaft, '2023':]
    ) / 1000

# %%
future_emissions_renamed_reunited_dedafted.loc[future_emissions_renamed_reunited_dedafted.variable == 'CO2 FFI']

# %%
# check alignment in 2023
for specie in historical.variable.unique():
    print(historical.loc[historical.variable==specie, '2023'].values[0] - future_emissions_renamed_reunited_dedafted.loc[future_emissions_renamed_reunited_dedafted.variable==specie, '2023'].values[0])

# %%
os.makedirs('../output', exist_ok=True)

# %%
future_emissions_renamed_reunited_dedafted.to_csv('../output/future_emissions_cleaned.csv', index=False)

# %%
