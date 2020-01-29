import os
from glob import glob
import geopandas as gpd

from pysis import isis
from pysis.exceptions import ProcessError

import plio
from plio.io.io_gdal import GeoDataset

from shapely import wkt
import numpy as np
import pvl
from shapely.geometry import Point, MultiPolygon

def segment_hirise(directory, offset=300):
    images = glob(os.path.join(directory, "*RED*.stitched.norm.cub"))
    for image in images:
        label = pvl.loads(isis.catlab(from_=image))

        dims = label["IsisCube"]["Core"]["Dimensions"]
        nlines, nsamples = dims["Lines"], dims["Samples"]
        print("Lines, Samples: ", nlines, nsamples)

        starts = np.arange(1, nlines, nsamples)
        stops = np.append(np.arange(starts[1], nlines, nsamples), [nlines])

        starts[1:] -= offset
        stops[:-1] += offset

        segments = np.asarray([starts, stops]).T

        for i, seg in enumerate(segments):
            start, stop = seg
            output = os.path.splitext(image)[0] + f".{start}_{stop}" + ".cub"
            print("Writing:", output)
            isis.crop(from_=image, to=output, line=start, nlines=stop-start, sample=1, nsamples=nsamples)
            isis.footprintinit(from_=output)

    return load_segments(directory)


def load_segments(directory):
    images = glob(os.path.join(directory, "*RED*.*_*.cub"))
    objs = [GeoDataset(image) for image in images]
    footprints = [o.footprint for o in objs]
    footprints = [wkt.loads(f.ExportToWkt()) for f in footprints]
    return gpd.GeoDataFrame(data=np.asarray([images, objs, footprints]).T, columns=["path", "image", "footprint"], geometry="footprint")


def ingest_hirise(directory):

    l = glob(os.path.join(directory, "*RED*.IMG"))
    l = [os.path.splitext(i)[0] for i in l]
    print(l)
    cube_name = "_".join(os.path.splitext(os.path.basename(l[0]))[0].split("_")[:-2])

    print("Cube Name:", cube_name)

    print(f"Running hi2isis on {l}")
    for i,cube in enumerate(l):
        print(f"{i+1}/{len(l)}")
        try:
            isis.hi2isis(from_=f'{cube}.IMG', to=f"{cube}.cub")
            print(f"finished {cube}")
        except ProcessError as e:
            print(e.stderr)
            return

    print(f"running spiceinit on {l}")
    for i,cube in enumerate(l):
        print(f"{i+1}/{len(l)}")
        try:
            isis.spiceinit(from_=f'{cube}.cub')
        except ProcessError as e:
            print(e.stderr)
            return

    print(f"running hical on {l}")
    for i,cube in enumerate(l):
        print(f"{i}/{len(l)}")
        try:
            isis.hical(from_=f'{cube}.cub', to=f'{cube}.cal.cub')
        except ProcessError as e:
            print(e.stderr)
            return

    cal_list_0 = sorted(glob(os.path.join(directory, "*0.cal*")))
    cal_list_1 = sorted(glob(os.path.join(directory, "*1.cal*")))
    print(f"Channel 0 images: {cal_list_0}")
    print(f"Channel 1 images: {cal_list_1}")

    for i,cubes in enumerate(zip(cal_list_0, cal_list_1)):
        print(f"{i+1}/{len(cal_list_0)}")
        c0, c1 = cubes
        output ="_".join(c0.split("_")[:-1])
        try:
            isis.histitch(from1=c0, from2=c1, to=f"{output}.stitched.cub")
        except ProcessError as e:
            print(e.stderr)
            return

    stitch_list = glob(os.path.join(directory, "*stitched*"))
    for cube in stitch_list:
        output = os.path.splitext(cube)[0] + ".norm.cub"
        try:
            isis.cubenorm(from_=cube, to=output)
        except ProcessError as e:
            print(e.stderr)
            return




