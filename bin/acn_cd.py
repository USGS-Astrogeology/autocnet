#!/usr/bin/env python

import sys
import os
import argparse
from argparse import RawTextHelpFormatter
import yaml
import tempfile

from plio.io.io_gdal import GeoDataset, array_to_raster

from autocnet.graph.network import CandidateGraph
from autocnet.cg import change_detection as cd
from autocnet.examples import get_path
from autocnet.graph.network import CandidateGraph
from autocnet.graph.edge import Edge
from autocnet.spatial.isis import point_info
from autocnet.utils import hirise
from autocnet.utils.utils import bytescale
from autocnet.examples import get_path
from shapely.geometry import Point, MultiPoint
import numpy as np

from affine import Affine

import pysis
from pysis.exceptions import ProcessError
from pysis import isis

import warnings
warnings.simplefilter("ignore")


_cd_functions_ = {
    "okb" : cd.okubogar_detector,
    "okbm" : cd.okbm_detector,
    "blob" : cd.blob_detector
}

def poly_pixel_to_latlon(poly, affine, coord_transform):
    poly_type = type(poly)

    if poly_type == MultiPoint:
        x = [p.x for p in poly]
        y = [p.y for p in poly]
    elif poly_type == Point:
        x,y = poly.xy
    else:
        x,y = poly.exterior.coords.xy

    lonlats = []
    for xval,yval in zip(x,y):
        lon, lat = Affine.from_gdal(*affine) * (xval, yval)
        lon, lat, _ = coord_transform.TransformPoint(lon, lat)
        lonlats.append([lon, lat])

    if poly_type == Point:
        return Point(lonlats[0][0], lonlats[0][1])

    return poly_type(lonlats)



if __name__ == "__main__":
    cd_function_help_string = ("Change detection algorithm to use.\n"
                               "Okubogar method (okb). Simple method which produces an overlay image of change hotspots (i.e. a 2d histogram image of detected change density). Largely based on a method created by Chris Okubo and Brendon Bogar. Histogram step was added for readability, image1, image2 -> image subtraction/ratio -> feature extraction -> feature histogram.\n\n"
                               "Okubogar modified method (okbm). Experimental feature based change detection algorithm that expands on okobubogar to allow for programmatic change detection."
    )

    parser = argparse.ArgumentParser(description="Registers two image and runs a change detection algorithm on the pair of images. WARNING: Runs bundle adjust with update=yes, make sure you are using copies.")
    parser.add_argument('before', action='store', help='Path to image 1, generally the "before image"')
    parser.add_argument('after', action='store', help='Path to image 2, generally the "after image"')
    parser.add_argument('out', action='store', help='Output image path, csv with geometries are also written as a side cart file as a csv.')
    parser.add_argument('--algorithm', '-a', action='store', choices=_cd_functions_.keys(), help=cd_function_help_string, default='okb')
    parser.add_argument('--config', '-c', action='store', default=get_path('cd_config.yml'), help='path to json or yaml file containing parameters for change detection algorithms')
    parser.add_argument('--map','-m',  action='store', help='path to ISIS map file, determines the projection of the two registered images', default=os.path.join(os.environ["ISISROOT"], "appdata", "templates", "maps", "equirectangular.map"))
    parser.add_argument('--register','-r', action="store_true", default=False, help='Whether or not to register the two images, reccomended to set to false if the two images have been registered before.')
    parser.add_argument('--write-registered-cubes','-w', default=False, action="store_true", help='Pass this flag id you want to write out the projected cubes to disk. Useful if you want to run multiple cd algorithms without having to rerun the registration step.')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # temp path for temp files
    dirpath = tempfile.mkdtemp()

    if args.register:
        # Point to the adjacency Graph
        adjacency = {args.before: [args.after], args.after: [args.before]}
        cg = CandidateGraph.from_adjacency(adjacency)

        # Apply SIFT to extract features
        cg.extract_features(extractor_method='vlfeat', extractor_parameters={"edge_thresh":20})
        cg.match()

        # Apply outlier detection
        cg.apply_func_to_edges(Edge.symmetry_check)
        cg.apply_func_to_edges(Edge.ratio_check)
        cg.minimum_spanning_tree()

        # Compute a homography and apply RANSAC
        cg.apply_func_to_edges(Edge.compute_fundamental_matrix, clean_keys=['ratio', 'symmetry'])

        # Generate ISIS compatible control network
        cg.generate_control_network(clean_keys=["fundamental"])

        # write cnet out to temp file, run it through bundle adjust.
        cnet_path = os.path.join(dirpath, "cnet.net")
        filelist_path = os.path.join(dirpath, "cnet.lis")

        cg.to_isis(cnet_path)

        try:
            output = isis.jigsaw(fromlist=filelist_path, cnet=cnet_path, onet=cnet_path, update="yes", **config['jigsaw'])
            print(output.decode())
        except ProcessError as e:
            print(e.stdout.decode('utf-8'))
            print(e.stderr)
            exit()

        if args.write_registered_cubes:
            before_proj = os.path.splitext(args.before)[0] + ".proj.cub"
            after_proj = os.path.splitext(args.after)[0] + ".proj.cub"
        else: # use the temp directory
            before_proj = os.path.join(dirpath, "before.cub")
            after_proj = os.path.join(dirpath, "after.cub")

        try:
            isis.cam2map(from_=args.before, to=before_proj, map=args.map)
            isis.cam2map(from_=args.after, to=after_proj, patchsize=8, map=before_proj, matchmap=True, warpalgorithm="REVERSEPATCH")
        except ProcessError as e:
            print(e.stderr)
            exit()

        args.before = before_proj
        args.after = after_proj

    before_proj_geo = GeoDataset(args.before)
    after_proj_geo = GeoDataset(args.after)

    if args.algorithm == 'blob':
        # Requires sub solar azmith
        ssa_path = "/tmp/ssa.cub" # os.path.join(dirpath, 'ssa.cub')
        try:
            isis.phocube(from_=args.before, to=ssa_path, subsolargroundazimuth=True)
        except ProcessError as e:
            print(e.stderr)
        print("created: ", ssa_path)
        ssa = GeoDataset(ssa_path).read_array(6)
        ret = _cd_functions_[args.algorithm.strip()](before_proj_geo, after_proj_geo, ssa, **config.get(args.algorithm, {}))
    else:
        ret = _cd_functions_[args.algorithm.strip()](before_proj_geo, after_proj_geo, **config.get(args.algorithm, {}))

    # for now, write out raster files assuming okb
    # make it match one of the projected images
    match_srs = before_proj_geo.dataset.GetProjection()
    match_gt = before_proj_geo.geotransform
    match_coord_trans = before_proj_geo.coordinate_transformation

    if os.path.splitext(args.out)[1] == '':
        args.out = args.out + ".tif"

    print(f"Writing {args.out}")
    array_to_raster(ret[1], args.out, projection=match_srs, geotransform=match_gt, outformat="GTiff")

    print(f"Writing {os.path.splitext(args.out)[0]+'.csv'}")

    geodf = ret[0]
    geodf["geometry"] = geodf['geometry']
    geodf['latlon_geometry'] = geodf['geometry'].apply(lambda x: poly_pixel_to_latlon(x, match_gt, match_coord_trans ))
    geodf['geometry'] = [g.wkt for g in geodf['geometry']]
    geodf['latlon_geometry'] = [g.wkt for g in geodf['latlon_geometry']]

    geodf.to_csv(os.path.splitext(args.out)[0]+".csv", index=False)

