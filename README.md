# Nonequilibrium-Fluctuations
Idea based on paper "Nonequilibrium Fluctuations of Global Warming". 

Data were mainly from ERA5(sea surface, 1000hPa level pressure, 500hPa level pressure). 

## Functions
Some of the functions of the code were as follows:
- `DataAveraging.py`: group data in years.
- `pdfA.py`: plotting probabilistic density function.
- `smooth.py`: smooth data on time scale.
- `mean-var.py`: plotting mean-variance.
- `time-series.py`: dividing data based on 3 latitude, plotting data in time-series.

Most of the data were evaluated using smooth function.

## Reference
- Yin, J., Porporato, A., & Rondoni, L. (2024). Nonequilibrium Fluctuations of Global Warming. Journal of Climate, 37(9), 2809-2819. https://doi.org/10.1175/JCLI-D-23-0273.1
- Laarne P, Amnell E, Zaidan MA, Mikkonen S, Nieminen T. Exploring Non-Linear Dependencies in Atmospheric Data with Mutual Information. Atmosphere. 2022; 13(7):1046. https://doi.org/10.3390/atmos13071046

- PDF models and synthetic model for the wind speed fluctuations based on the resolution of Langevin equation. https://doi.org/10.1016/j.apenergy.2012.05.007
