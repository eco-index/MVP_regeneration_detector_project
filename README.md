# Native_plantings
Detection of native plantings in New Zealand

This project aimed to detect native planting efforts through the use of satellite imagery combined with a neural network approach. The native plantings are between 30-50% canopy cover and a trial area of the Waiwhakaiho area within the Taranaki region was used.

This project was a joint effort between Litmus Datatech, Trees that Count, Taranaki Mounga Project and Eco-index. Litmus created the satellite data collection as well as the neural network model (based on Prithvi V2). Trees That Count provided the areas and locations for recorded planting efforts across New Zealand, Taranaki Mounga Project provided the validation of plantings across the Waiwhakaiho and eco-Index organized the efforts from each organization as well as provided data cleaning and handling between each step.

An initial attempt to use the SAM2 model did not work out successfully so a switch to Prithvi was done.

The Satellite data collection is all completed within the eco-index-main directory and the classificaion is done in the eco-index-classifier-master directory which each have thier own set of instructions within the README file.

Finally, eco-Index created a script which georeferences the detections as they are output from the prithvi model in order to validate within a googleearth or GIS environment.

The current model is predicting where native plantings are to a XX percentage by using an IOU calculation. Further refinement to add additional training data to the model would help to improve this accuarcy. 
