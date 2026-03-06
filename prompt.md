# Introduction

I have a semesterproject to write in the scope of 12credit points (~300hrs worth of work). The title of the project is Ocean current parameter retrieval using Doppler Centroid Anomalies in spaceborne SAR imagery. The goal right now is to use BIOMASS data and retrieve corrected Doppler centroid anomalies from it. I already have a BIOMASS scene of the eastern coast of South Africa, I want to know which other datasets I need to correctly retrieve the ocean current induced DCA.

# Datasets
## BIOMASS
 
I have a BIOMASS Level-1A SLC scene from 2026/02/04, I have a scene where there is land also visible. The scene still needs to be geocoded, for this I have the GammaRemoteSensing software package available, if zou have anz questions regarding this or need documentation please ask.

## ERA5

From the ERA5 analysis I have:

- the 10 metre U wind component and 10 metre V wind component, so the wind speeds in North and East direction. The resolution is 0.25DEG.

For waves I have (resolution of 0.5DEG):

- Mean wave direction
- Mean wave period
- Significant height of combined wind waves and swell
- Mean direction of wind waves
- Significant height of total swell
- Significant height of wind waves

## Copernicus Global Ocean Physics Analysis and Forecast

From the Global Ocean Physics Analysis and Forecast I have ocean current data for the study area. Following datasets are available in a resolution of 1/12DEG:

- Eastward total velocity (Eulerian + Waves + Tide)
- Northward total velocity (Eulerian + Waves + Tide)
- Eastward tide-induced velocity (Tide current)
- Northward tide-induced velocity (Tide current)
- Eastward Eulerian velocity (Navier-Stokes current)
- Northward Eulerian velocity (Navier-Stokes current)
- Eastward wave-induced velocity (Stokes drift)
- Northward wave-induced velocity (Stokes drift)

# Your tasks

I want you to help me with the following:

- What the best strategy is to move forward with this project
- Finding out which datasets to use additionally to the ones already available
- What datasets of the ones already available are helpful
- What preprocessing steps are necessary (especiall BIOMASS, as in any sort of corrections for the Level-1A data)
- How to coregister all my data, especially my BIOMASS dataset (as it is not geocoded yet)
- How to retrieve ocean current with Doppler Centroid Anomalies

Restrictions/Additional information:

- I am programming in Python
- Use literature for your help
- I have the Gamma remote sensing software package available (maybe helps with Geocoding BIMOASS)
- I can provide you with data specification for each datasets, if you need additional information ask for it