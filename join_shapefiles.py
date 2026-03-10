import os
import geopandas as gpd
import pandas as pd

def join_shapefiles(input_dir, files_sorted):

    # Filter only shapefiles (.shp extension)
    shapefiles = [os.path.join(input_dir, file) for file in files_sorted if file.endswith('.shp')]

    output_path = "waiwhakaiho_plantings_joined_clean_200thresh.shp"
    # Read each shapefile into a GeoDataFrame and collect them in a list
    dataframes = []
    for shp_path in shapefiles:
        if shp_path.startswith('.'):
            continue
        if os.path.exists(output_path):
            return
        gdf = gpd.read_file(shp_path)
        dataframes.append(gdf)
    
    # Combine all GeoDataFrames into one
    combined_gdf = pd.concat(dataframes, axis=0)
    
    #dissolve all shapes together to get past overlapping detections
    combined_gdf = combined_gdf.dissolve()

    # Set a consistent CRS (example: WGS84, adjust if needed)
    #combined_gdf = combined_gdf.to_crs("EPSG:4326")
    
    # Write the combined data to an output shapefile
    combined_gdf.to_file(output_path)
    
    print(f"Successfully joined {len(shapefiles)} shapefiles into {output_path}")

if __name__ == "__main__":

    input_directory = 'output_shapes'
    files = os.listdir(input_directory)

    join_shapefiles(input_directory, files)