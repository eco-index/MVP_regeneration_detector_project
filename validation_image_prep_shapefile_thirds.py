#adapted from validation_image_prep to work with the two image layout of latest detections as well as add in location from OCR of the screen grab
from PIL import Image
import os
import numpy as np
import cv2
import easyocr
from pyproj import Transformer
import pyproj
import fiona
import shapely.geometry as geom
from tqdm import tqdm
#from paddleocr import PaddleOCR

def split_image(idx, mask_path, output_folder):
    # Open the input image
    with Image.open(mask_path) as img:
        width, height = img.size
        
        # Calculate the new width for each half
        quarter_width = width // 3
        
        # Split the image into three parts
        images = [img.crop((ind * quarter_width, 0, (ind + 1) * quarter_width, height)) for ind in range(3)]
        
        #grab location from image    3840*2160
        location_img = images[0].crop((3350,2130,3838,2160))
        #location_img.save('location_'+str(i)+'.png')

        reader = easyocr.Reader(['en']) # Specify language(s)
        results = reader.readtext(np.array(location_img.convert('RGB')))
        for result in results:
            if len(result[1]) > 17:
                result_split = result[1].split(' ')
                s_deg = result_split[0][:2]
                s_min = result_split[0][3:5]
                s_sec = result_split[0][-4:-2]

                try: deg_dec = (int(s_min)/60) + (int(s_sec)/3600)
                except:
                    print(result[1])
                    return
                deg_s = int(s_deg)+deg_dec

                n_deg = result_split[1][:3]
                n_min = result_split[1][4:6]
                n_sec = result_split[1][-4:-2]
                try: deg_dec = (int(n_min)/60) + (int(n_sec)/3600)
                except:
                    print(result[1])
                    return

                deg_n = int(n_deg)+deg_dec

                #print(i, result_split[0])
                #print(f'{deg_s:.5f} {deg_n:.5f}')
                #print(s_deg, 'degS',s_min,"'", s_sec,'"', n_deg, 'degN',n_min,"'", n_sec,'"')

                file_name = f'-{deg_s:.5f} {deg_n:.5f}.png'
                #print(file_name)
        scale_image = images[0].crop((3300,2147,3580,2153))
        _, binary_mask = cv2.threshold(np.array(scale_image), 127, 255, cv2.THRESH_BINARY)
        start_point = 0
        for pix in range(0,binary_mask.shape[1]):
            value = binary_mask[3,pix,0]
            if value == 0 and start_point == 0:
                start_point = pix
            elif value == 0 and not start_point == 0:
                end_point = pix
                break
        scale_pix = end_point - start_point

        scale_value = images[0].crop((3450,2130,3580,2151))
        _, scale_value = cv2.threshold(np.array(scale_value), 150, 255, cv2.THRESH_BINARY)
        #scale_value.save('temp/location_'+str(idx)+'.png')
        cv2.imwrite('temp/location_'+str(idx)+'.png', scale_value)
        reader = easyocr.Reader(['en']) # Specify language(s)
        results = reader.readtext(scale_value, allowlist='0123456789')
        #results = reader.readtext(np.array(scale_value.convert('RGB')), allowlist='0123456789')
        if len(results) == 0:
            return
        if int(results[0][1]) > 300 and int(results[0][1]) < 2000:
            scale = int(results[0][1][:-1])
        else:
            return


    #create geodataframe of centre point
    # always_xy=True ensures input is always (longitude, latitude) and output (easting, northing)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    easting, northing = transformer.transform(deg_n, deg_s*-1)

    #create boundary coordinates from image dimensions
    #m_per_px = 30/150 #0.2
    #m_per_px = 40/157 #0.25
    #80/195 = 0.41
    m_per_px = scale/scale_pix
    
    xmin = easting-((quarter_width/2)*m_per_px)
    xmax = easting+((quarter_width/2)*m_per_px)
    ymin = northing-((height/2)*m_per_px)
    ymax = northing+((height/2)*m_per_px)


    threshold_value = 200
    im_np = np.asarray(images[1])
    _, binary_mask = cv2.threshold(im_np, threshold_value, 255, cv2.THRESH_BINARY)
    thresh = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)
        
    #check for no contours detected in empty image
    if len(contours) == 0:
        print('no detections')
        return

    multigon = []

    #loop through contours

    #find all parent contrours and make a list, parent contours all end with -1 with CCOMP since they have no parents
    parent_conts = []
    for i, hier in enumerate(hierarchy[0]):
        if hier[3] == -1:
            parent_conts.append(i)

    children_list = []
    children_parents = []
    for i, hier in enumerate(hierarchy[0]):
        if not hier[3] == -1:
            children_list.append(i)
            children_parents.append(hier[3])

    ##### need to check for holes in the polygon and create if necessary using hierarchy info. To create hole need to pass interior holes list to Polygon: poly_geom = geom.Polygon(multilist, [int hole coords])
    #note interior holes need to be counterclockwise while ext should be clockwise. Can end int holes with [::-1] to change direction if given in CW originally
    #link for more info: https://gis.stackexchange.com/questions/353057/how-to-create-a-shapely-polygon-with-a-hole
    for parent_id in parent_conts:
        #print(parent_id, end='\r')
        #print(parent_id)
        cont = contours[parent_id]

        #within the contour loop through the pixel values
        #check for points or lines in the polygons and skip past
        if len(cont) < 3: continue
        multilist = []

        #create the parent polygon
        for pix in cont:
                x_pix = pix[0][0]
                y_pix = pix[0][1]
                #pixel to long/ lat conversion
                long_pix = x_pix * m_per_px + xmin
                lat_pix = ymax - y_pix * m_per_px
                multilist.append((long_pix, lat_pix))

        holes_list = []
        #check for parent contour having any children children
        if parent_id in children_parents:
            #find the children if they are available
            for child_cont in children_list:
                hole_coords = []
                if hierarchy[0][child_cont][3] == parent_id:
                    for pix in contours[child_cont]:
                        x_pix = pix[0][0]
                        y_pix = pix[0][1]
                        #pixel to long/ lat conversion
                        long_pix = x_pix * m_per_px + xmin
                        lat_pix = ymax - y_pix * m_per_px
                        hole_coords.append((long_pix, lat_pix))
                    #hole_geom = geom.Polygon(hole_coords)
                    holes_list.append(hole_coords)
                    #holes_list.append(hole_coords[::-1])

        #if len(holes_list) > 0: print(holes_list)
        poly_geom = geom.Polygon(multilist, holes_list)
        multigon.append(poly_geom)

    multi_polygon = geom.MultiPolygon(multigon)

    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    save_name = os.path.join(output_folder, str(idx)+'.shp')
    with fiona.open(save_name, 'w', 'ESRI Shapefile', schema, crs = "EPSG:2193") as c:
        for i, polys in enumerate(multi_polygon.geoms):
            c.write({
                'geometry': geom.mapping(polys),
                'properties': {'id': i},
            })



if __name__ == "__main__":
    input_dir = 'waiwhakaiho_joined'
    images = os.listdir(input_dir)
    images.sort()

    masks_dir = 'waiwhakaiho_predictions_prithvi2/updated_predictions'
    masks = os.listdir(masks_dir)
    masks.sort()

    #loop through masks and check if there is a corresponding image to match
    #if so then create a shapefile from the mask image


    #for i, mask in tqdm(enumerate(masks),total=len(masks)):
    #start from mask number 350 since no detections before this were found
    for i, mask in enumerate(masks):
        print(mask)
        if i < 356:
            continue
        output_directory = "output_shapes"  # Replace with the desired output directory
        
        mask_path = os.path.join(masks_dir, mask)
        split_image(i, mask_path, output_directory)
